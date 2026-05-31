"""Conversation_Agent（对话 Agent，ReAct，Req 10, 18.1, 19.1）。

本模块实现基于 **ReAct（推理-行动交替）** 模式的对话 Agent，对外提供单一入口
:meth:`ConversationAgent.handle`，编排 Prompt_Module / Retriever / Chat_Model /
Tool_Registry / Memory_Module / Indexer 完成一轮多工具对话应答：

* **历史召回（Req 10.1）**：构造提示词前，先经 Retriever 在当前 Session 范围内召回
  相关历史消息（历史消息 Top-K，1–50），并将其作为上下文拼入初始消息列表。
* **进入循环（Req 10.2）**：用 Prompt_Module 的 ``react_agent`` 模板构造系统提示词，
  绑定 Tool_Registry 的 Function Call schema 后发送给 Chat_Model。
* **思考-行动交替（Req 10.3）**：当模型输出包含工具调用请求时，经 Tool_Registry 调用
  对应工具，并将工具响应作为新的观察结果（``role="tool"`` 消息）送回模型继续循环。
* **退出循环（Req 10.4）**：当模型输出不再包含工具调用请求时，退出循环并将最终内容
  作为应答返回。
* **迭代上限（Req 10.5）**：迭代次数达配置上限（默认 10，范围 1–50）时终止循环，返回
  已生成的内容并附"已达最大迭代次数"提示。
* **工具错误/超时（Req 10.6）**：工具返回错误，或在工具调用超时（默认 30s，范围
  1–300s）内未返回时，终止该次工具调用并将失败原因作为观察结果送回模型，由模型决定
  后续行动（循环继续）。
* **模型错误/超时（Req 10.7）**：循环中模型返回错误，或在模型调用超时（默认 60s，范围
  1–120s）内未返回时，终止循环、保留本轮已生成内容并返回表明应答失败的结果。
* **持久化接线（Req 18.1, 19.1）**：成功产出应答后，最佳努力地写入 Memory_Module 短期
  记忆，并经 Indexer 将本轮消息向量入库；持久化失败仅记录、不影响应答返回。

设计抉择：**采用手写的确定性 ReAct 循环，而非 LangGraph ``create_react_agent``。**
设计文档将 LangGraph 列为 ReAct 的编排便利项，但本属性（Property 13：迭代次数上界）
要求迭代计数"由构造保证"且可在完全离线、无网络环境下被属性化测试覆盖。
``create_react_agent`` 会把迭代记账隐藏在框架内部并引入对底层 LLM/运行时的耦合，
不利于精确断言"模型始终请求工具调用时迭代次数严格不超过上限"。因此这里直接实现
ReAct 循环：循环体显式以 ``range(1, max_iterations + 1)`` 控制上界，使迭代次数严格不
超过配置上限（Property 13 由构造成立）；所有协作者均以依赖注入方式提供，默认实现惰性
构造、构造期不触网，从而支持确定性单元/属性测试。这是一个正当且地道的工程选择，
LangGraph 仅为编排便利、并非该属性的必要条件。

超时实现：工具调用与模型调用均经 :meth:`ConversationAgent._call_with_timeout` 包裹
（基于 ``ThreadPoolExecutor`` 的硬超时）。对于模型调用，:class:`ChatModel` 自身已将
SDK 超时映射为 :class:`GenerationTimeoutError`；此处的线程超时为"双保险"，确保即便
``generate`` 阻塞也能在配置时限内被终止（线程不强杀，仅从调用方视角施加上界）。
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable

from pydantic import BaseModel, Field

from ghost_agent.config import get_settings
from ghost_agent.core.chat_model import ChatMessage, ChatModel, Completion
from ghost_agent.core.indexer import Indexer
from ghost_agent.core.prompt_module import PromptModule
from ghost_agent.core.retriever import Retriever
from ghost_agent.core.tool_registry import ToolRegistry, build_default_registry
from ghost_agent.memory.memory_module import MemoryModule
from ghost_agent.models.errors import GenerationError, GenerationTimeoutError

logger = logging.getLogger(__name__)

__all__ = [
    "ConversationAgent",
    "ConversationResult",
    "MAX_ITERATIONS_NOTICE",
    "GENERATION_FAILED_NOTICE",
]

# ReAct 最大迭代次数的取值范围（Req 10.5）。构造期对越界配置做防御性钳制。
_MIN_ITERATIONS = 1
_MAX_ITERATIONS = 50

#: 达到最大迭代次数上限时附加到应答中的提示（必须包含子串"已达最大迭代次数"，Property 13）。
MAX_ITERATIONS_NOTICE = "[已达最大迭代次数上限，以下为当前已生成的内容，可能尚未完成。]"

#: 模型错误/超时终止循环时附加到应答中的失败提示（Req 10.7）。
GENERATION_FAILED_NOTICE = "[应答生成失败：对话模型调用出错或超时，以上为已生成的内容（如有）。]"

# 三种循环退出原因。
_STOP_COMPLETED = "completed"
_STOP_MAX_ITERATIONS = "max_iterations"
_STOP_GENERATION_ERROR = "generation_error"


# --------------------------------------------------------------------------- #
# 返回结构                                                                      #
# --------------------------------------------------------------------------- #
class ConversationResult(BaseModel):
    """一轮 ReAct 对话的应答结果。

    Attributes:
        answer: 返回给用户的最终应答文本（达上限/失败时含相应提示并保留已生成内容）。
        session_id: 本轮所属会话标识。
        iterations: 实际执行的 ReAct 迭代次数（模型调用次数），恒满足
            ``1 <= iterations <= max_iterations``（Property 13）。
        stop_reason: 循环退出原因，取
            ``"completed" | "max_iterations" | "generation_error"`` 之一。
    """

    answer: str = Field(..., description="返回给用户的最终应答文本。")
    session_id: str = Field(..., description="本轮所属会话标识。")
    iterations: int = Field(..., ge=0, description="实际执行的 ReAct 迭代次数。")
    stop_reason: str = Field(..., description="循环退出原因。")


# --------------------------------------------------------------------------- #
# ConversationAgent                                                             #
# --------------------------------------------------------------------------- #
class ConversationAgent:
    """基于 ReAct 模式的对话 Agent（手写确定性循环，Req 10, 18.1, 19.1）。

    Args:
        chat_model: 对话模型；为 ``None`` 时构造默认 :class:`ChatModel`（惰性连接）。
        tool_registry: 工具集；为 ``None`` 时构造默认注册表
            （:func:`build_default_registry`，含四个内置工具）。
        retriever: 检索器（用于历史消息召回）；为 ``None`` 时构造默认 :class:`Retriever`。
        prompt_module: 提示词模块；为 ``None`` 时构造默认 :class:`PromptModule`。
        memory: 记忆模块；为 ``None`` 时构造默认 :class:`MemoryModule`。
        indexer: 索引器（用于消息向量入库）；为 ``None`` 时构造默认 :class:`Indexer`。
        max_iterations: ReAct 最大迭代次数；为 ``None`` 时取
            ``settings.react_max_iterations``。无论来源如何均被钳制到 [1, 50]（Req 10.5）。
        tool_call_timeout: 单次工具调用超时（秒）；为 ``None`` 时取
            ``settings.tool_call_timeout_seconds``（Req 10.6）。
        model_call_timeout: 单次模型调用超时（秒）；为 ``None`` 时取
            ``settings.model_call_timeout_seconds``（Req 10.7）。
        react_template_name: ReAct 系统提示词模板名（默认 ``"react_agent"``）。
    """

    def __init__(
        self,
        *,
        chat_model: ChatModel | None = None,
        tool_registry: ToolRegistry | None = None,
        retriever: Retriever | None = None,
        prompt_module: PromptModule | None = None,
        memory: MemoryModule | None = None,
        indexer: Indexer | None = None,
        max_iterations: int | None = None,
        tool_call_timeout: float | None = None,
        model_call_timeout: float | None = None,
        react_template_name: str = "react_agent",
    ) -> None:
        settings = get_settings()
        self._chat_model = chat_model if chat_model is not None else ChatModel()
        self._tool_registry = (
            tool_registry if tool_registry is not None else build_default_registry()
        )
        self._retriever = retriever if retriever is not None else Retriever()
        self._prompt_module = (
            prompt_module if prompt_module is not None else PromptModule()
        )
        self._memory = memory if memory is not None else MemoryModule()
        self._indexer = indexer if indexer is not None else Indexer()

        raw_iterations = (
            max_iterations
            if max_iterations is not None
            else settings.react_max_iterations
        )
        # 防御性钳制：即便配置层失效，迭代上限也不会超出 [1, 50]（Req 10.5 / Property 13）。
        self._max_iterations: int = max(
            _MIN_ITERATIONS, min(_MAX_ITERATIONS, int(raw_iterations))
        )
        self._tool_call_timeout: float = (
            tool_call_timeout
            if tool_call_timeout is not None
            else settings.tool_call_timeout_seconds
        )
        self._model_call_timeout: float = (
            model_call_timeout
            if model_call_timeout is not None
            else settings.model_call_timeout_seconds
        )
        self._react_template_name = react_template_name

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def max_iterations(self) -> int:
        """ReAct 最大迭代次数（已钳制到 [1, 50]）。"""
        return self._max_iterations

    # ------------------------------------------------------------------ #
    # 公共 API：handle                                                    #
    # ------------------------------------------------------------------ #
    def handle(self, session_id: str, message: str) -> ConversationResult:
        """处理一轮用户消息，执行 ReAct 循环并返回应答（Req 10.1–10.7）。

        Args:
            session_id: 当前会话标识。
            message: 本轮用户消息文本。

        Returns:
            :class:`ConversationResult`，含最终应答、迭代次数与退出原因。
        """
        history_hits = self._recall_history(session_id, message)
        messages = self._build_initial_messages(message, history_hits)
        chat_model = self._bind_tools(self._chat_model)

        # 本轮已生成的最近一段文本内容（用于达上限/失败时保留，Req 10.5/10.7）。
        last_content = ""

        for iteration in range(1, self._max_iterations + 1):
            try:
                completion = self._generate(chat_model, messages)
            except (GenerationError, GenerationTimeoutError) as exc:
                # Req 10.7：模型返回错误 → 终止循环、保留已生成内容、返回应答失败。
                logger.warning(
                    "ReAct 模型调用失败（session=%s, iteration=%d）：%r",
                    session_id,
                    iteration,
                    exc,
                )
                return self._generation_error_result(
                    session_id, last_content, iteration
                )
            except FuturesTimeoutError:
                # Req 10.7：模型调用超时（线程硬超时兜底）→ 同失败处理。
                logger.warning(
                    "ReAct 模型调用超时（session=%s, iteration=%d, timeout=%ss）",
                    session_id,
                    iteration,
                    self._model_call_timeout,
                )
                return self._generation_error_result(
                    session_id, last_content, iteration
                )

            if completion.content:
                last_content = completion.content

            # 将模型本次输出（含其文本内容）追加为 assistant 消息。
            messages.append(
                ChatMessage(role="assistant", content=completion.content)
            )

            if not completion.tool_calls:
                # Req 10.4：不再含工具调用 → 退出循环并返回最终内容。
                answer = completion.content
                self._persist(session_id, message, answer)
                return ConversationResult(
                    answer=answer,
                    session_id=session_id,
                    iterations=iteration,
                    stop_reason=_STOP_COMPLETED,
                )

            # Req 10.3 / 10.6：逐个执行工具调用，将响应/失败原因作为观察结果送回。
            for tool_call in completion.tool_calls:
                observation = self._invoke_tool(tool_call)
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=observation,
                        tool_call_id=tool_call.id or None,
                        name=tool_call.name or None,
                    )
                )

        # Req 10.5：迭代次数达上限仍未结束 → 终止并返回已生成内容 + 达上限提示。
        answer = self._append_limit_notice(last_content)
        self._persist(session_id, message, answer)
        return ConversationResult(
            answer=answer,
            session_id=session_id,
            iterations=self._max_iterations,
            stop_reason=_STOP_MAX_ITERATIONS,
        )

    # ------------------------------------------------------------------ #
    # 内部：历史召回与提示词构造                                            #
    # ------------------------------------------------------------------ #
    def _recall_history(self, session_id: str, message: str) -> list[Any]:
        """召回当前 Session 的相关历史消息（Req 10.1）。

        召回失败（如空向量库、嵌入不可用）时容错降级为"无历史"，不阻断对话。
        """
        try:
            return list(
                self._retriever.retrieve_messages(message, session_id)
            )
        except Exception as exc:  # noqa: BLE001 - 历史召回失败降级为无历史，不阻断对话
            logger.warning(
                "历史消息召回失败（session=%s），按无历史处理：%r", session_id, exc
            )
            return []

    def _build_initial_messages(
        self, message: str, history_hits: list[Any]
    ) -> list[ChatMessage]:
        """构造初始消息列表：系统提示词 + 召回历史（如有）+ 当前用户消息（Req 10.1/10.2）。"""
        prompt = self._prompt_module.build(self._react_template_name)
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=prompt.text)
        ]
        if history_hits:
            messages.append(
                ChatMessage(role="system", content=self._format_history(history_hits))
            )
        messages.append(ChatMessage(role="user", content=message))
        return messages

    @staticmethod
    def _format_history(history_hits: list[Any]) -> str:
        """将召回的历史消息命中拼装为上下文文本块（纯拼接，brace 安全）。"""
        lines: list[str] = ["## 相关历史消息"]
        for hit in history_hits:
            metadata = getattr(hit, "metadata", {}) or {}
            role = metadata.get("role", "") if isinstance(metadata, dict) else ""
            prefix = f"[{role}] " if role else ""
            lines.append(f"{prefix}{getattr(hit, 'text', '')}")
        return "\n".join(lines)

    def _bind_tools(self, chat_model: ChatModel) -> ChatModel:
        """将 Tool_Registry 的 Function Call schema 绑定到模型（无工具时原样返回）。"""
        try:
            schemas = self._tool_registry.to_openai_schemas()
        except Exception as exc:  # noqa: BLE001 - schema 生成失败降级为不绑定工具
            logger.warning("生成工具 schema 失败，按不绑定工具处理：%r", exc)
            return chat_model
        if schemas and hasattr(chat_model, "bind_tools"):
            return chat_model.bind_tools(schemas)
        return chat_model

    # ------------------------------------------------------------------ #
    # 内部：模型调用与工具调用                                              #
    # ------------------------------------------------------------------ #
    def _generate(
        self, chat_model: ChatModel, messages: list[ChatMessage]
    ) -> Completion:
        """调用模型生成一次输出，并施加模型调用超时上界（Req 10.7）。"""
        return self._call_with_timeout(
            lambda: chat_model.generate(messages), self._model_call_timeout
        )

    def _invoke_tool(self, tool_call: Any) -> str:
        """执行一次工具调用，将结果或失败原因归一化为观察结果文本（Req 10.3/10.6）。

        工具参数解析失败、工具返回错误、或工具调用超时（Req 10.6）均不抛出，而是把
        描述失败原因的文本作为观察结果返回，交由模型决定后续行动。
        """
        name = getattr(tool_call, "name", "") or ""
        try:
            params = self._parse_arguments(getattr(tool_call, "arguments", ""))
        except ValueError as exc:
            return f"工具 {name} 调用失败：参数解析错误（{exc}）"

        try:
            result = self._call_with_timeout(
                lambda: self._tool_registry.invoke(name, params),
                self._tool_call_timeout,
            )
        except FuturesTimeoutError:
            # Req 10.6：工具调用超时 → 失败原因作为观察结果送回。
            return (
                f"工具 {name} 调用超时：在 {self._tool_call_timeout} 秒内未返回响应。"
            )
        except Exception as exc:  # noqa: BLE001 - 工具错误作为观察结果送回（Req 10.6）
            return f"工具 {name} 调用失败：{exc}"

        return self._stringify_observation(result)

    @staticmethod
    def _parse_arguments(arguments: Any) -> dict[str, Any]:
        """将模型给出的工具参数（JSON 字符串）解析为字典；非法/非对象时抛 ValueError。"""
        if isinstance(arguments, dict):
            return arguments
        text = (arguments or "").strip() if isinstance(arguments, str) else ""
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"非法 JSON 参数：{exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("工具参数必须为 JSON 对象")
        return parsed

    @staticmethod
    def _stringify_observation(result: Any) -> str:
        """将工具返回值归一化为可作为观察结果回送模型的文本。"""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    @staticmethod
    def _call_with_timeout(func: Callable[[], Any], timeout: float) -> Any:
        """在独立线程中执行 ``func`` 并施加硬超时。

        超时抛 :class:`concurrent.futures.TimeoutError`。``shutdown(wait=False)`` 不
        阻塞等待潜在挂起线程，从而保证调用方视角的超时上界（线程本身不被强杀）。
        """
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(func)
            return future.result(timeout=timeout)
        finally:
            executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # 内部：结果装配与持久化                                                #
    # ------------------------------------------------------------------ #
    def _generation_error_result(
        self, session_id: str, last_content: str, iteration: int
    ) -> ConversationResult:
        """构造模型错误/超时的应答结果：保留已生成内容并附失败提示（Req 10.7）。"""
        if last_content:
            answer = f"{last_content}\n\n{GENERATION_FAILED_NOTICE}"
        else:
            answer = GENERATION_FAILED_NOTICE
        return ConversationResult(
            answer=answer,
            session_id=session_id,
            iterations=iteration,
            stop_reason=_STOP_GENERATION_ERROR,
        )

    @staticmethod
    def _append_limit_notice(last_content: str) -> str:
        """在已生成内容后附加"已达最大迭代次数"提示（Req 10.5 / Property 13）。"""
        if last_content:
            return f"{last_content}\n\n{MAX_ITERATIONS_NOTICE}"
        return MAX_ITERATIONS_NOTICE

    def _persist(self, session_id: str, message: str, answer: str) -> None:
        """最佳努力地写入记忆与消息向量（Req 18.1, 19.1）；失败仅记录，不影响应答。"""
        try:
            self._memory.append(session_id, message, answer)
        except Exception as exc:  # noqa: BLE001 - 持久化失败不影响应答返回
            logger.warning(
                "写入短期记忆失败（session=%s），不影响应答返回：%r", session_id, exc
            )
        try:
            self._indexer.index_message(session_id, message, answer)
        except Exception as exc:  # noqa: BLE001 - 消息向量入库失败不中断对话（Req 19.2）
            logger.warning(
                "消息向量入库失败（session=%s），不影响应答返回：%r", session_id, exc
            )

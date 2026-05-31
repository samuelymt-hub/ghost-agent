"""Chat_Model（对话模型）封装。

封装对 Doubao（火山方舟）对话模型的调用（Req 9.2, 10.2, 2.2），对上层 Agent
提供与具体 SDK 解耦的稳定接口：

* :meth:`ChatModel.generate` —— 同步生成一次完整应答（``Completion``）。
* :meth:`ChatModel.stream`   —— 异步流式生成增量（``AsyncIterator[Delta]``，
  对应 SSE 增量推送，Req 2.2）。
* :meth:`ChatModel.bind_tools` —— 绑定工具（Function Call）模式（OpenAI 风格
  函数 schema），返回新的 ``ChatModel`` 实例（不可变语义）。

设计要点：
- **惰性连接（Lazy connection）**：构造函数不创建任何 SDK 客户端、不进行任何
  网络调用。真正的 ``volcenginesdkarkruntime.Ark`` 客户端在首次 ``generate`` /
  ``stream`` 调用时才构建并缓存。模块可在无 API Key、无网络环境下被 ``import``
  与单元测试。
- **LangChain 解耦**：此处刻意不强依赖 LangChain 消息类型，而是定义轻量级的
  ``ChatMessage`` / ``ToolCall`` / ``Completion`` / ``Delta`` 结构。LangChain /
  LangGraph 将在 Agent 层包装本组件。
- **错误透传**：超时统一封装为 :class:`GenerationTimeoutError`，其余 SDK / 网络
  异常统一封装为 :class:`GenerationError`，并保留原始异常作为 ``__cause__``，由
  上层（KBA / Conversation_Agent / SSE 层）按 Req 9.5/9.6/10.7/2.5 处理。
- **可测试性 seam**：内部通过 :meth:`_build_client` 构建底层 SDK 客户端，测试可
  monkeypatch 该方法注入假客户端，从而在离线环境验证生成 / 流式行为。
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from ghost_agent.config import get_settings
from ghost_agent.models.errors import GenerationError, GenerationTimeoutError

__all__ = [
    "ChatMessage",
    "ToolCall",
    "Completion",
    "Delta",
    "ChatModel",
]


# --------------------------------------------------------------------------- #
# 轻量级消息 / 结果结构                                                          #
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    """一条对话消息（与具体 SDK / LangChain 解耦的轻量结构）。

    Attributes:
        role: 消息角色，``"system" | "user" | "assistant" | "tool"`` 之一。
        content: 文本内容；工具调用型 assistant 消息可为空串。
        tool_call_id: 当 ``role == "tool"`` 时，标识其回应的工具调用 ID。
        name: 工具消息对应的工具名（可选）。
    """

    role: str
    content: str = ""
    tool_call_id: str | None = None
    name: str | None = None


class ToolCall(BaseModel):
    """模型输出的一次工具调用请求（Function Call）。

    Attributes:
        id: 工具调用唯一标识（由模型生成，用于关联工具结果）。
        name: 待调用的工具 / 函数名。
        arguments: 调用参数，保持模型返回的 JSON 字符串原样。
    """

    id: str
    name: str
    arguments: str = ""


class Completion(BaseModel):
    """一次完整的生成结果。

    Attributes:
        content: 生成的文本内容。
        tool_calls: 模型请求的工具调用列表（无则为空列表）。
        finish_reason: 结束原因（如 ``"stop"`` / ``"tool_calls"`` / ``"length"``）。
    """

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None


class Delta(BaseModel):
    """流式生成中的一个增量片段（聚焦文本内容，服务于 SSE delta 事件，Req 2.2）。

    Attributes:
        content: 本次增量的文本片段。
    """

    content: str = ""


# --------------------------------------------------------------------------- #
# 内部工具：超时识别                                                            #
# --------------------------------------------------------------------------- #
def _is_timeout(exc: BaseException) -> bool:
    """判断异常是否为"超时"语义。

    由于无法稳定导入 Ark/OpenAI SDK 的具体超时异常类型，这里采用宽松判定：
    - 标准库 :class:`TimeoutError`（``concurrent.futures.TimeoutError`` 在
      Python 3.11+ 即为内建 ``TimeoutError`` 的别名）；
    - 或异常类型名中（不区分大小写）包含 ``timeout``（覆盖
      ``APITimeoutError`` / ``ReadTimeout`` 等 SDK 自定义异常）。
    """
    if isinstance(exc, TimeoutError):
        return True
    return "timeout" in type(exc).__name__.lower()


# --------------------------------------------------------------------------- #
# ChatModel                                                                     #
# --------------------------------------------------------------------------- #
class ChatModel:
    """Doubao 对话模型客户端（惰性连接、支持工具调用与流式输出）。

    Args:
        api_key: 火山引擎 Chat API Key；为 ``None`` 时取
            ``settings.effective_chat_api_key``。
        base_url: 火山方舟 API Base URL；为 ``None`` 时取 ``settings.doubao_base_url``。
        model: Chat 模型 ID / 推理接入点；为 ``None`` 时取 ``settings.doubao_chat_model``。
        timeout: 模型调用超时（秒）；为 ``None`` 时取
            ``settings.model_call_timeout_seconds``（Req 10.7）。
        tools: 可选的工具 schema 列表（OpenAI Function Call 风格），用于绑定到模型。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        tools: list[dict] | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key: str = (
            api_key if api_key is not None else settings.effective_chat_api_key
        )
        self._base_url: str = (
            base_url if base_url is not None else settings.doubao_base_url
        )
        self._model: str = model if model is not None else settings.doubao_chat_model
        self._timeout: float = (
            timeout if timeout is not None else settings.model_call_timeout_seconds
        )
        # 复制一份工具列表，避免外部可变引用泄漏到内部状态。
        self._tools: list[dict] | None = list(tools) if tools else None
        # 惰性构建并缓存底层 SDK 客户端；构造期间不触碰网络。
        self._client: Any | None = None

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def model(self) -> str:
        """当前使用的 Chat 模型 ID。"""
        return self._model

    @property
    def tools(self) -> list[dict] | None:
        """当前已绑定的工具 schema 列表（未绑定时为 ``None``）。"""
        return self._tools

    # ------------------------------------------------------------------ #
    # 工具绑定（Function Call）                                            #
    # ------------------------------------------------------------------ #
    def bind_tools(self, tools: list[dict]) -> "ChatModel":
        """返回绑定了给定工具 schema 的新 ``ChatModel`` 实例（不可变语义）。

        工具 schema 采用 OpenAI 风格函数定义：
        ``{"type": "function", "function": {"name", "description", "parameters"}}``。
        Agent 层（Tool_Registry，任务 8）负责生成这些 schema。原实例保持不变。

        Args:
            tools: 待绑定的工具 schema 列表。

        Returns:
            一个新的 ``ChatModel``，沿用原配置但绑定了新的工具列表。
        """
        return ChatModel(
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._model,
            timeout=self._timeout,
            tools=tools,
        )

    # ------------------------------------------------------------------ #
    # 测试 seam：构建底层 SDK 客户端                                       #
    # ------------------------------------------------------------------ #
    def _build_client(self) -> Any:
        """构建底层 volcengine Ark 客户端。

        SDK 在方法内部惰性导入，避免 import 期或无关测试受 SDK 安装 / 配置影响。
        测试可通过 ``monkeypatch.setattr(instance, "_build_client", fake)`` 注入
        假客户端。

        Raises:
            GenerationError: 未配置 API Key，或 SDK 不可导入时抛出。
        """
        if not self._api_key:
            raise GenerationError(
                "未配置 Doubao Chat API Key，无法调用对话模型"
                "（请设置 DOUBAO_CHAT_API_KEY 或 DOUBAO_API_KEY）"
            )
        try:
            from volcenginesdkarkruntime import Ark
        except Exception as exc:  # noqa: BLE001 - SDK 导入失败统一封装
            raise GenerationError(
                "Doubao(volcengine) SDK 导入失败，无法构建 Chat 客户端"
            ) from exc
        return Ark(api_key=self._api_key, base_url=self._base_url)

    def _get_client(self) -> Any:
        """返回缓存的底层客户端，必要时惰性构建。"""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # ------------------------------------------------------------------ #
    # 公共 API：generate                                                  #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> Completion:
        """同步生成一次完整应答。

        Args:
            messages: 对话消息列表（``ChatMessage``）。
            temperature: 采样温度；为 ``None`` 时不传，使用模型默认值。

        Returns:
            ``Completion``，包含文本内容、工具调用列表与结束原因。

        Raises:
            GenerationTimeoutError: 底层调用超时（Req 9.6, 10.7）。
            GenerationError: 未配置 Key、SDK 不可用或其他生成阶段错误
                （保留原始异常为 ``__cause__``，Req 9.5, 10.7）。
        """
        client = self._get_client()
        kwargs = self._build_create_kwargs(messages, temperature=temperature, stream=False)
        try:
            response = client.chat.completions.create(**kwargs)
        except GenerationError:
            # 来自 _get_client/_build_client 的封装异常，原样向上抛出。
            raise
        except Exception as exc:  # noqa: BLE001 - 统一封装 SDK/网络异常供上层处理
            raise self._wrap_exception(exc) from exc

        return self._parse_completion(response)

    # ------------------------------------------------------------------ #
    # 公共 API：stream                                                    #
    # ------------------------------------------------------------------ #
    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[Delta]:
        """异步流式生成，按生成先后顺序逐个产出 ``Delta``（Req 2.2）。

        底层 Ark SDK 的流是同步可迭代对象，这里在 async 生成器中顺序迭代，
        将每个 chunk 的文本增量包装为 ``Delta`` 产出，保持生成顺序。

        Args:
            messages: 对话消息列表（``ChatMessage``）。
            temperature: 采样温度；为 ``None`` 时不传，使用模型默认值。

        Yields:
            ``Delta``，含本次增量文本片段。

        Raises:
            GenerationTimeoutError: 建立流或迭代过程中超时（Req 9.6, 10.7）。
            GenerationError: 未配置 Key、SDK 不可用或流中途出错
                （保留原始异常为 ``__cause__``；SSE 层据此发送 error 事件，Req 2.5）。
        """
        client = self._get_client()
        kwargs = self._build_create_kwargs(messages, temperature=temperature, stream=True)
        try:
            sync_stream = client.chat.completions.create(**kwargs)
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 - 建流阶段异常统一封装
            raise self._wrap_exception(exc) from exc

        iterator = iter(sync_stream)
        while True:
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            except GenerationError:
                raise
            except Exception as exc:  # noqa: BLE001 - 流中途异常统一封装
                raise self._wrap_exception(exc) from exc

            piece = self._extract_delta_content(chunk)
            if piece:
                yield Delta(content=piece)

    # ------------------------------------------------------------------ #
    # 内部工具：请求参数构造与响应解析                                       #
    # ------------------------------------------------------------------ #
    def _build_create_kwargs(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None,
        stream: bool,
    ) -> dict[str, Any]:
        """将 ``ChatMessage`` 列表与选项转换为 SDK ``create()`` 的关键字参数。"""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [self._message_to_dict(m) for m in messages],
            "timeout": self._timeout,
            "stream": stream,
        }
        if self._tools:
            kwargs["tools"] = self._tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

    @staticmethod
    def _message_to_dict(message: ChatMessage) -> dict[str, Any]:
        """将单条 ``ChatMessage`` 转换为 SDK 所需的字典格式（仅含有效字段）。"""
        data: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_call_id is not None:
            data["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            data["name"] = message.name
        return data

    @staticmethod
    def _parse_completion(response: Any) -> Completion:
        """从 SDK 响应对象解析出 ``Completion``。

        预期结构：``response.choices[0].message`` 含 ``.content`` 与可选
        ``.tool_calls``；``response.choices[0].finish_reason`` 为结束原因。

        Raises:
            GenerationError: 响应结构非法（缺少 choices / message 等）。
        """
        try:
            choice = response.choices[0]
            message = choice.message
        except Exception as exc:  # noqa: BLE001
            raise GenerationError("Chat 模型响应结构非法：缺少 choices/message") from exc

        content = getattr(message, "content", None) or ""
        finish_reason = getattr(choice, "finish_reason", None)

        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        for raw in raw_tool_calls:
            function = getattr(raw, "function", None)
            tool_calls.append(
                ToolCall(
                    id=getattr(raw, "id", "") or "",
                    name=(getattr(function, "name", "") or "") if function is not None else "",
                    arguments=(
                        getattr(function, "arguments", "") or ""
                    )
                    if function is not None
                    else "",
                )
            )

        return Completion(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _extract_delta_content(chunk: Any) -> str:
        """从流式 chunk 中提取文本增量；结构异常时返回空串以保证流的稳健性。

        预期结构：``chunk.choices[0].delta.content``。
        """
        try:
            delta = chunk.choices[0].delta
        except Exception:  # noqa: BLE001 - 容忍空/心跳类 chunk
            return ""
        return getattr(delta, "content", None) or ""

    @staticmethod
    def _wrap_exception(exc: Exception) -> GenerationError:
        """将底层 SDK / 网络异常封装为领域异常（区分超时与一般错误）。"""
        if _is_timeout(exc):
            return GenerationTimeoutError("Chat 模型调用超时")
        return GenerationError("调用 Chat 模型失败")

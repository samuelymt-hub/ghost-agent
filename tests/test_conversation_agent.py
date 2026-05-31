"""Conversation_Agent 测试（任务 12.1–12.3，Req 10.1–10.7, 18.1, 19.1）。

覆盖：
- ReAct 循环 handle（Req 10.1–10.7）：
  - happy path：模型立即给出最终内容 → completed、iterations==1、记忆+消息向量入库各一次。
  - 历史召回（10.1）：retriever.retrieve_messages 以 session_id 调用，召回命中拼入提示词。
  - 工具调用观察结果（10.3/10.6）：工具错误 / 超时 → 失败原因作为 tool 观察结果送回模型。
  - 模型错误/超时（10.7）：终止循环、保留已生成内容、返回应答失败结果（不抛出）。
  - 持久化最佳努力：memory.append 抛错不影响应答返回。
  - 迭代上限钳制到 [1, 50]。
- 属性测试（Hypothesis, max_examples>=100, deadline=None）：
  - Property 13（12.2）：始终请求工具的模型行为下，迭代次数不超过配置上限，
    达上限应答携带"已达最大迭代次数"提示；混合脚本下提前完成时迭代计数正确。

测试以离线替身隔离全部外部依赖，无任何网络调用：
- ``ScriptedChatModel``：按脚本返回 Completion 序列，支持 bind_tools，统计 generate 次数；
  可配置为始终请求工具调用（驱动 Property 13）。
- ``FakeToolRegistry``：invoke 返回值 / 抛错 / 睡眠（超时）；to_openai_schemas 返回工具 schema。
- ``StubRetriever``：retrieve_messages 返回预置命中并记录调用参数。
- ``SpyMemory`` / ``SpyIndexer``：记录持久化接线调用。
"""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.agents.conversation_agent import (
    MAX_ITERATIONS_NOTICE,
    ConversationAgent,
    ConversationResult,
)
from ghost_agent.core.chat_model import Completion, ToolCall
from ghost_agent.models.errors import GenerationError, GenerationTimeoutError
from ghost_agent.models.vector_record import VectorType
from ghost_agent.vector_db.vector_store import SearchHit


# --------------------------------------------------------------------------- #
# 离线替身                                                                      #
# --------------------------------------------------------------------------- #
class ScriptedChatModel:
    """按脚本返回 Completion 序列的 Chat_Model 替身。

    * ``script`` —— 依次返回的 Completion 列表；调用次数超出脚本时返回最后一个。
    * ``always_tool`` —— 为 True 时忽略 script，每次都返回带工具调用的 Completion
      （驱动 Property 13：模型始终请求工具调用）。
    * ``error`` —— 非 None 时每次 generate 抛出该异常（驱动 Req 10.7）。
    :attr:`calls` 记录每次 generate 收到的消息列表（用于断言观察结果回送）。
    :attr:`bound_tools` 记录最近一次 bind_tools 绑定的 schema。
    """

    def __init__(
        self,
        *,
        script: list[Completion] | None = None,
        always_tool: bool = False,
        error: Exception | None = None,
    ) -> None:
        self._script = list(script or [])
        self._always_tool = always_tool
        self._error = error
        self.calls: list[list] = []
        self.bound_tools: list[dict] | None = None

    def bind_tools(self, tools: list[dict]) -> "ScriptedChatModel":
        # 返回自身（携带脚本/计数状态），并记录绑定的工具 schema。
        self.bound_tools = list(tools)
        return self

    def generate(self, messages, *, temperature: float | None = None) -> Completion:
        self.calls.append(list(messages))
        if self._error is not None:
            raise self._error
        if self._always_tool:
            return Completion(
                content="思考中，需要调用工具",
                tool_calls=[ToolCall(id=f"tc-{len(self.calls)}", name="query_cls_log", arguments="{}")],
                finish_reason="tool_calls",
            )
        index = len(self.calls) - 1
        if index < len(self._script):
            return self._script[index]
        return self._script[-1] if self._script else Completion(content="最终答案")


class FakeToolRegistry:
    """内存版 Tool_Registry 替身：可配置返回值 / 抛错 / 睡眠超时。

    * ``result`` —— invoke 的返回值。
    * ``error`` —— 非 None 时 invoke 抛出该异常（驱动 Req 10.6 工具错误）。
    * ``sleep`` —— invoke 前睡眠秒数（驱动 Req 10.6 工具超时）。
    :attr:`invocations` 记录 (name, params) 调用。
    """

    def __init__(
        self,
        *,
        result=None,
        error: Exception | None = None,
        sleep: float = 0.0,
        schemas: list[dict] | None = None,
    ) -> None:
        self._result = result if result is not None else {"status": "ok"}
        self._error = error
        self._sleep = sleep
        self._schemas = schemas if schemas is not None else [
            {
                "type": "function",
                "function": {
                    "name": "query_cls_log",
                    "description": "查询日志",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]
        self.invocations: list[tuple[str, dict]] = []

    def to_openai_schemas(self) -> list[dict]:
        return list(self._schemas)

    def invoke(self, name: str, params: dict):
        self.invocations.append((name, params))
        if self._sleep:
            time.sleep(self._sleep)
        if self._error is not None:
            raise self._error
        return self._result


class StubRetriever:
    """retrieve_messages 返回预置命中并记录调用参数的 Retriever 替身。"""

    def __init__(self, hits: list[SearchHit] | None = None, *, raises: Exception | None = None) -> None:
        self._hits = list(hits or [])
        self._raises = raises
        self.calls: list[tuple[str, str]] = []

    def retrieve_messages(self, query: str, session_id: str, top_k: int | None = None):
        self.calls.append((query, session_id))
        if self._raises is not None:
            raise self._raises
        return list(self._hits)


class SpyMemory:
    """记录 append 调用的 Memory_Module 替身；可配置 append 抛错。"""

    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.appends: list[tuple[str, str, str]] = []

    def append(self, session_id: str, user_msg: str, answer: str) -> None:
        self.appends.append((session_id, user_msg, answer))
        if self._error is not None:
            raise self._error


class SpyIndexer:
    """记录 index_message 调用的 Indexer 替身；可配置抛错。"""

    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.indexed: list[tuple[str, str, str]] = []

    def index_message(self, session_id: str, user_msg: str, answer: str) -> None:
        self.indexed.append((session_id, user_msg, answer))
        if self._error is not None:
            raise self._error


# --------------------------------------------------------------------------- #
# 工厂                                                                          #
# --------------------------------------------------------------------------- #
def _make_agent(
    *,
    chat_model,
    tool_registry=None,
    retriever=None,
    memory=None,
    indexer=None,
    max_iterations=None,
    tool_call_timeout=None,
    model_call_timeout=None,
) -> ConversationAgent:
    return ConversationAgent(
        chat_model=chat_model,
        tool_registry=tool_registry if tool_registry is not None else FakeToolRegistry(),
        retriever=retriever if retriever is not None else StubRetriever(),
        memory=memory if memory is not None else SpyMemory(),
        indexer=indexer if indexer is not None else SpyIndexer(),
        max_iterations=max_iterations,
        tool_call_timeout=tool_call_timeout,
        model_call_timeout=model_call_timeout,
    )


def _message_hit(text: str, role: str = "user") -> SearchHit:
    return SearchHit(
        id=f"m-{text}",
        text=text,
        source_id="sess",
        vector_type=VectorType.MESSAGE,
        score=0.9,
        metadata={"role": role},
    )


# =========================================================================== #
# 12.1 — handle 单元测试                                                        #
# =========================================================================== #
def test_handle_happy_path_completed_with_persistence():
    """模型立即给出最终内容（无工具调用）→ completed、iterations==1、持久化各一次（10.4/18.1/19.1）。"""
    chat = ScriptedChatModel(script=[Completion(content="这是最终答案", finish_reason="stop")])
    memory = SpyMemory()
    indexer = SpyIndexer()
    agent = _make_agent(chat_model=chat, memory=memory, indexer=indexer)

    result = agent.handle("sess-1", "你好")

    assert isinstance(result, ConversationResult)
    assert result.stop_reason == "completed"
    assert result.iterations == 1
    assert result.answer == "这是最终答案"
    assert chat.calls and len(chat.calls) == 1
    # Req 18.1 / 19.1：应答完成后写入记忆与消息向量各一次。
    assert memory.appends == [("sess-1", "你好", "这是最终答案")]
    assert indexer.indexed == [("sess-1", "你好", "这是最终答案")]


def test_handle_recalls_history_and_includes_in_prompt():
    """历史召回（10.1）：retrieve_messages 以 session_id 调用，召回文本拼入初始消息。"""
    chat = ScriptedChatModel(script=[Completion(content="ok", finish_reason="stop")])
    retriever = StubRetriever([_message_hit("上轮提到磁盘告警", role="user")])
    agent = _make_agent(chat_model=chat, retriever=retriever)

    agent.handle("sess-hist", "继续排查")

    assert retriever.calls == [("继续排查", "sess-hist")]
    # 首次模型调用的消息中应包含召回的历史文本。
    first_call_text = "\n".join(m.content for m in chat.calls[0])
    assert "上轮提到磁盘告警" in first_call_text


def test_handle_history_recall_failure_degrades_gracefully():
    """历史召回失败（如空向量库）降级为无历史，不阻断对话（10.1 容错）。"""
    chat = ScriptedChatModel(script=[Completion(content="ok", finish_reason="stop")])
    retriever = StubRetriever(raises=RuntimeError("vector store empty"))
    agent = _make_agent(chat_model=chat, retriever=retriever)

    result = agent.handle("sess-2", "提问")

    assert result.stop_reason == "completed"
    assert result.answer == "ok"


def test_handle_tool_error_observation_fed_back_then_completes():
    """工具错误（10.6）：失败原因作为 tool 观察结果送回，下一轮模型给出最终答案 → completed。"""
    chat = ScriptedChatModel(
        script=[
            Completion(
                content="需要查日志",
                tool_calls=[ToolCall(id="tc1", name="query_cls_log", arguments="{}")],
                finish_reason="tool_calls",
            ),
            Completion(content="根据观察得出结论", finish_reason="stop"),
        ]
    )
    registry = FakeToolRegistry(error=RuntimeError("日志服务不可用"))
    agent = _make_agent(chat_model=chat, tool_registry=registry)

    result = agent.handle("sess-tool-err", "排查")

    assert result.stop_reason == "completed"
    assert result.iterations == 2
    assert result.answer == "根据观察得出结论"
    # 第二次模型调用应收到包含失败原因的 tool 观察结果。
    second_call = chat.calls[1]
    tool_msgs = [m for m in second_call if m.role == "tool"]
    assert tool_msgs, "工具失败应作为 tool 观察结果送回模型"
    assert "日志服务不可用" in tool_msgs[-1].content


def test_handle_tool_timeout_observation_fed_back():
    """工具超时（10.6）：工具睡眠超过 tool_call_timeout → 超时失败作为观察结果送回，循环继续。"""
    chat = ScriptedChatModel(
        script=[
            Completion(
                content="调用工具",
                tool_calls=[ToolCall(id="tc1", name="query_cls_log", arguments="{}")],
                finish_reason="tool_calls",
            ),
            Completion(content="超时后的最终答案", finish_reason="stop"),
        ]
    )
    registry = FakeToolRegistry(sleep=0.3)
    agent = _make_agent(chat_model=chat, tool_registry=registry, tool_call_timeout=0.05)

    result = agent.handle("sess-tool-timeout", "排查")

    assert result.stop_reason == "completed"
    assert result.iterations == 2
    second_call = chat.calls[1]
    tool_msgs = [m for m in second_call if m.role == "tool"]
    assert tool_msgs and "超时" in tool_msgs[-1].content


def test_handle_tool_success_observation_fed_back():
    """工具成功（10.3）：工具响应作为观察结果送回模型继续循环。"""
    chat = ScriptedChatModel(
        script=[
            Completion(
                content="调用工具",
                tool_calls=[ToolCall(id="tc1", name="query_cls_log", arguments='{"query": "error"}')],
                finish_reason="tool_calls",
            ),
            Completion(content="完成", finish_reason="stop"),
        ]
    )
    registry = FakeToolRegistry(result={"hits": 3})
    agent = _make_agent(chat_model=chat, tool_registry=registry)

    result = agent.handle("sess-tool-ok", "排查")

    assert result.stop_reason == "completed"
    assert registry.invocations == [("query_cls_log", {"query": "error"})]
    tool_msgs = [m for m in chat.calls[1] if m.role == "tool"]
    assert tool_msgs and "hits" in tool_msgs[-1].content


def test_handle_model_error_terminates_and_preserves_content():
    """模型错误（10.7）：终止循环、保留已生成内容、返回应答失败结果（不抛出）。"""
    # 第一轮请求工具（产生已生成内容），第二轮模型抛错。
    class ErrorOnSecond(ScriptedChatModel):
        def generate(self, messages, *, temperature=None):
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                return Completion(
                    content="已生成的部分内容",
                    tool_calls=[ToolCall(id="tc1", name="query_cls_log", arguments="{}")],
                    finish_reason="tool_calls",
                )
            raise GenerationError("模型炸了")

    chat = ErrorOnSecond()
    agent = _make_agent(chat_model=chat)

    result = agent.handle("sess-model-err", "提问")

    assert result.stop_reason == "generation_error"
    assert result.iterations == 2
    # 保留本轮已生成的内容（Req 10.7）。
    assert "已生成的部分内容" in result.answer


def test_handle_model_error_on_first_iteration_no_partial_content():
    """模型首轮即错（10.7）：无已生成内容，返回失败结果且不抛出。"""
    chat = ScriptedChatModel(error=GenerationError("立刻失败"))
    agent = _make_agent(chat_model=chat)

    result = agent.handle("sess-err-1", "提问")

    assert result.stop_reason == "generation_error"
    assert result.iterations == 1
    assert result.answer  # 含失败提示


def test_handle_model_timeout_terminates_with_generation_error():
    """模型超时（10.7）：GenerationTimeoutError 与模型错误同处理 → generation_error。"""
    chat = ScriptedChatModel(error=GenerationTimeoutError("模型超时"))
    agent = _make_agent(chat_model=chat)

    result = agent.handle("sess-timeout", "提问")

    assert result.stop_reason == "generation_error"


def test_handle_persistence_failure_does_not_break_response():
    """持久化最佳努力：memory.append 抛错不影响应答返回（Req 18.1 容错）。"""
    chat = ScriptedChatModel(script=[Completion(content="答案", finish_reason="stop")])
    memory = SpyMemory(error=RuntimeError("memory down"))
    indexer = SpyIndexer(error=RuntimeError("index down"))
    agent = _make_agent(chat_model=chat, memory=memory, indexer=indexer)

    result = agent.handle("sess-persist", "提问")

    assert result.stop_reason == "completed"
    assert result.answer == "答案"


def test_max_iterations_clamped_to_range():
    """迭代上限钳制到 [1, 50]：越界配置被钳制。"""
    chat = ScriptedChatModel(script=[Completion(content="x", finish_reason="stop")])
    assert _make_agent(chat_model=chat, max_iterations=0).max_iterations == 1
    assert _make_agent(chat_model=chat, max_iterations=999).max_iterations == 50
    assert _make_agent(chat_model=chat, max_iterations=7).max_iterations == 7


def test_handle_max_iterations_notice_when_always_tool():
    """始终请求工具 → 达上限：stop_reason max_iterations，应答含"已达最大迭代次数"。"""
    chat = ScriptedChatModel(always_tool=True)
    agent = _make_agent(chat_model=chat, max_iterations=3)

    result = agent.handle("sess-max", "提问")

    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3
    assert "已达最大迭代次数" in result.answer
    assert len(chat.calls) == 3


# =========================================================================== #
# 12.2 — 属性测试 Property 13                                                    #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 13: 对任意 Chat_Model 行为（包括始终请求工具调用的情形），Conversation_Agent 的 ReAct 循环迭代次数不超过配置的最大迭代次数上限（默认 10，范围 1–50），且达上限时返回的应答携带"已达最大迭代次数"的提示信息。
# Validates: Requirements 10.5


@settings(max_examples=200, deadline=None)
@given(
    max_iterations=st.integers(min_value=1, max_value=50),
    # 额外维度（会话标识与查询文本）确保输入空间足够大，运行 >=100 个样例
    # （仅 max_iterations 取值域为 [1,50]，单独使用会在 50 个样例后被 Hypothesis 提前耗尽）。
    session_id=st.text(min_size=1, max_size=12),
    query=st.text(min_size=1, max_size=24),
)
def test_property_13_always_tool_respects_iteration_upper_bound(
    max_iterations: int, session_id: str, query: str
):
    """Property 13：模型始终请求工具调用时，迭代次数恰为上限，应答携带达上限提示。"""
    chat = ScriptedChatModel(always_tool=True)
    agent = _make_agent(chat_model=chat, max_iterations=max_iterations)

    result = agent.handle(session_id, query)

    # 迭代次数严格不超过配置上限（由构造保证）。
    assert result.iterations <= max_iterations
    # 始终请求工具 → 必然达上限。
    assert result.iterations == max_iterations
    assert result.stop_reason == "max_iterations"
    # 达上限时应答携带"已达最大迭代次数"提示。
    assert "已达最大迭代次数" in result.answer
    assert MAX_ITERATIONS_NOTICE in result.answer
    # 模型调用次数等于迭代次数，不超过上限。
    assert len(chat.calls) == max_iterations


@settings(max_examples=100, deadline=None)
@given(
    max_iterations=st.integers(min_value=1, max_value=50),
    finish_at=st.integers(min_value=1, max_value=50),
)
def test_property_13_mixed_script_completes_within_bound(max_iterations: int, finish_at: int):
    """Property 13（混合脚本）：模型在第 k 轮给出最终内容时迭代次数 == min(k, 上限)，且永不超过上限。"""
    # 构造脚本：前 finish_at-1 轮请求工具，第 finish_at 轮给出最终内容。
    script: list[Completion] = []
    for i in range(finish_at - 1):
        script.append(
            Completion(
                content=f"step-{i}",
                tool_calls=[ToolCall(id=f"tc-{i}", name="query_cls_log", arguments="{}")],
                finish_reason="tool_calls",
            )
        )
    script.append(Completion(content="最终内容", finish_reason="stop"))

    chat = ScriptedChatModel(script=script)
    agent = _make_agent(chat_model=chat, max_iterations=max_iterations)

    result = agent.handle("sess-mixed", "查询")

    assert result.iterations <= max_iterations
    if finish_at <= max_iterations:
        assert result.stop_reason == "completed"
        assert result.iterations == finish_at
        assert result.answer == "最终内容"
    else:
        assert result.stop_reason == "max_iterations"
        assert result.iterations == max_iterations
        assert "已达最大迭代次数" in result.answer

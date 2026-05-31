"""ChatModel 单元测试（任务 7.1）。

覆盖范围（基础设施 / 模型封装，按设计采用示例化测试，不使用 Hypothesis）：
- 构造期不连接（惰性），``model`` 反映 settings 默认值与覆盖值。
- ``generate`` 解析响应内容、工具调用（id/name/arguments），并把绑定工具传入 create()。
- 空 api_key 且不 monkeypatch -> GenerationError。
- SDK 抛超时类异常 -> GenerationTimeoutError；抛一般异常 -> GenerationError（保留 __cause__）。
- ``stream`` 按顺序产出 Delta，拼接后等于完整文本；流中途出错 -> GenerationError。
- ``bind_tools`` 返回携带工具的新实例且不修改原实例。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ghost_agent.core import ChatMessage, ChatModel, Completion, Delta, ToolCall
from ghost_agent.models.errors import GenerationError, GenerationTimeoutError


# --------------------------------------------------------------------------- #
# 假 SDK 客户端                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str | None = ""
    tool_calls: list = field(default_factory=list)


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str | None = "stop"


class _FakeResponse:
    def __init__(self, choice: _FakeChoice) -> None:
        self.choices = [choice]


@dataclass
class _FakeDelta:
    content: str | None = ""


@dataclass
class _FakeStreamChoice:
    delta: _FakeDelta


class _FakeStreamChunk:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeStreamChoice(_FakeDelta(content))]


class _FakeCompletions:
    """模拟 ``client.chat.completions``，记录调用并按配置返回 / 抛错。"""

    def __init__(
        self,
        *,
        response: _FakeResponse | None = None,
        stream_pieces: list[str] | None = None,
        raises: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self._response = response
        self._stream_pieces = stream_pieces
        self._raises = raises
        self._stream_error = stream_error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        if kwargs.get("stream"):
            return self._make_stream()
        return self._response

    def _make_stream(self):
        pieces = self._stream_pieces or []
        error = self._stream_error

        def _gen():
            for piece in pieces:
                yield _FakeStreamChunk(piece)
            if error is not None:
                raise error

        return _gen()


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeArk:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


# 模拟 SDK 超时异常（类型名包含 "Timeout"）。
class APITimeoutError(Exception):
    pass


# --------------------------------------------------------------------------- #
# 构造与属性                                                                    #
# --------------------------------------------------------------------------- #
def test_construction_without_api_key_does_not_raise():
    """无 API Key 构造不应抛错（惰性连接）。"""
    model = ChatModel(api_key="")
    assert model is not None
    assert model._client is None  # 尚未构建底层 SDK 客户端


def test_model_reflects_settings_default():
    from ghost_agent.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    model = ChatModel()
    assert model.model == settings.doubao_chat_model


def test_model_override():
    model = ChatModel(model="custom-chat-model")
    assert model.model == "custom-chat-model"


# --------------------------------------------------------------------------- #
# generate 行为                                                                 #
# --------------------------------------------------------------------------- #
def test_generate_returns_completion_with_content(monkeypatch):
    response = _FakeResponse(
        _FakeChoice(_FakeMessage(content="你好，我是助手"), finish_reason="stop")
    )
    completions = _FakeCompletions(response=response)
    model = ChatModel(api_key="dummy-key")
    monkeypatch.setattr(model, "_build_client", lambda: _FakeArk(completions))

    result = model.generate([ChatMessage(role="user", content="hi")])

    assert isinstance(result, Completion)
    assert result.content == "你好，我是助手"
    assert result.finish_reason == "stop"
    assert result.tool_calls == []
    # 消息已转换为字典格式传给 SDK。
    assert completions.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_generate_parses_tool_calls(monkeypatch):
    message = _FakeMessage(
        content="",
        tool_calls=[
            _FakeToolCall(
                id="call_1",
                function=_FakeFunction(name="query_cls_log", arguments='{"q": "err"}'),
            )
        ],
    )
    response = _FakeResponse(_FakeChoice(message, finish_reason="tool_calls"))
    completions = _FakeCompletions(response=response)
    model = ChatModel(api_key="dummy-key")
    monkeypatch.setattr(model, "_build_client", lambda: _FakeArk(completions))

    result = model.generate([ChatMessage(role="user", content="查日志")])

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.id == "call_1"
    assert call.name == "query_cls_log"
    assert call.arguments == '{"q": "err"}'
    assert result.finish_reason == "tool_calls"


def test_generate_passes_bound_tools_to_create(monkeypatch):
    response = _FakeResponse(_FakeChoice(_FakeMessage(content="ok")))
    completions = _FakeCompletions(response=response)
    tools = [
        {
            "type": "function",
            "function": {"name": "send_msg", "description": "send", "parameters": {}},
        }
    ]
    model = ChatModel(api_key="dummy-key").bind_tools(tools)
    monkeypatch.setattr(model, "_build_client", lambda: _FakeArk(completions))

    model.generate([ChatMessage(role="user", content="hi")])

    assert completions.calls[0]["tools"] == tools


def test_generate_omits_tools_when_not_bound(monkeypatch):
    response = _FakeResponse(_FakeChoice(_FakeMessage(content="ok")))
    completions = _FakeCompletions(response=response)
    model = ChatModel(api_key="dummy-key")
    monkeypatch.setattr(model, "_build_client", lambda: _FakeArk(completions))

    model.generate([ChatMessage(role="user", content="hi")])

    assert "tools" not in completions.calls[0]


def test_generate_with_empty_key_and_no_monkeypatch_raises():
    """空 API Key 且无注入时，generate 必须抛 GenerationError。"""
    model = ChatModel(api_key="")
    with pytest.raises(GenerationError):
        model.generate([ChatMessage(role="user", content="hi")])


def test_generate_timeout_maps_to_generation_timeout(monkeypatch):
    timeout_exc = APITimeoutError("read timed out")
    completions = _FakeCompletions(raises=timeout_exc)
    model = ChatModel(api_key="dummy-key")
    monkeypatch.setattr(model, "_build_client", lambda: _FakeArk(completions))

    with pytest.raises(GenerationTimeoutError) as exc_info:
        model.generate([ChatMessage(role="user", content="hi")])
    assert exc_info.value.__cause__ is timeout_exc


def test_generate_builtin_timeout_maps_to_generation_timeout(monkeypatch):
    timeout_exc = TimeoutError("timed out")
    completions = _FakeCompletions(raises=timeout_exc)
    model = ChatModel(api_key="dummy-key")
    monkeypatch.setattr(model, "_build_client", lambda: _FakeArk(completions))

    with pytest.raises(GenerationTimeoutError):
        model.generate([ChatMessage(role="user", content="hi")])


def test_generate_generic_error_maps_to_generation_error(monkeypatch):
    original = RuntimeError("network down")
    completions = _FakeCompletions(raises=original)
    model = ChatModel(api_key="dummy-key")
    monkeypatch.setattr(model, "_build_client", lambda: _FakeArk(completions))

    with pytest.raises(GenerationError) as exc_info:
        model.generate([ChatMessage(role="user", content="hi")])
    assert not isinstance(exc_info.value, GenerationTimeoutError)
    assert exc_info.value.__cause__ is original


# --------------------------------------------------------------------------- #
# stream 行为                                                                   #
# --------------------------------------------------------------------------- #
async def test_stream_yields_deltas_in_order(monkeypatch):
    pieces = ["你", "好", "，", "世界"]
    completions = _FakeCompletions(stream_pieces=pieces)
    model = ChatModel(api_key="dummy-key")
    monkeypatch.setattr(model, "_build_client", lambda: _FakeArk(completions))

    collected = []
    async for delta in model.stream([ChatMessage(role="user", content="hi")]):
        assert isinstance(delta, Delta)
        collected.append(delta.content)

    assert collected == pieces
    assert "".join(collected) == "你好，世界"
    assert completions.calls[0]["stream"] is True


async def test_stream_error_mid_iteration_raises_generation_error(monkeypatch):
    original = RuntimeError("stream broke")
    completions = _FakeCompletions(stream_pieces=["a", "b"], stream_error=original)
    model = ChatModel(api_key="dummy-key")
    monkeypatch.setattr(model, "_build_client", lambda: _FakeArk(completions))

    collected = []
    with pytest.raises(GenerationError) as exc_info:
        async for delta in model.stream([ChatMessage(role="user", content="hi")]):
            collected.append(delta.content)

    # 出错前已产出的增量保留（不撤回，Req 2.5 上层语义）。
    assert collected == ["a", "b"]
    assert exc_info.value.__cause__ is original


async def test_stream_timeout_maps_to_generation_timeout(monkeypatch):
    completions = _FakeCompletions(
        stream_pieces=["a"], stream_error=APITimeoutError("timed out")
    )
    model = ChatModel(api_key="dummy-key")
    monkeypatch.setattr(model, "_build_client", lambda: _FakeArk(completions))

    with pytest.raises(GenerationTimeoutError):
        async for _ in model.stream([ChatMessage(role="user", content="hi")]):
            pass


# --------------------------------------------------------------------------- #
# bind_tools 不可变语义                                                         #
# --------------------------------------------------------------------------- #
def test_bind_tools_returns_new_model_without_mutating_original():
    tools = [{"type": "function", "function": {"name": "send_msg", "parameters": {}}}]
    original = ChatModel(api_key="dummy-key", model="m")
    bound = original.bind_tools(tools)

    assert bound is not original
    assert bound.tools == tools
    assert original.tools is None  # 原实例未被修改
    assert bound.model == original.model  # 配置沿用

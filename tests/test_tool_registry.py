"""Tool_Registry 测试 (Req 16.1–16.6)。

包含：
- 任务 8.2 属性测试 Property 21（工具参数校验 iff 不变量，Hypothesis,
  ``max_examples>=100``）。
- 单元测试：重名冲突保留既有工具、未知工具、缺失必填、类型不符、bool 不计入
  NUMBER、未知参数忽略、合法放行、内置工具注册与后端可注入、OpenAI schema 结构。
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.core import (
    RegisteredTool,
    ToolRegistry,
    build_default_registry,
    register_builtin_tools,
)
from ghost_agent.models import (
    ParamDef,
    ParamType,
    ToolDefinition,
    ToolNamingConflictError,
    ToolNotFoundError,
    ToolSource,
    ToolValidationError,
)


# --------------------------------------------------------------------------- #
# 测试辅助                                                                      #
# --------------------------------------------------------------------------- #
class Recorder:
    """记录工具是否被执行及其入参，用于断言"不执行"语义。"""

    def __init__(self, result: Any = "OK") -> None:
        self.called = False
        self.params: dict[str, Any] | None = None
        self.result = result

    def __call__(self, params: dict[str, Any]) -> Any:
        self.called = True
        self.params = params
        return self.result


def _simple_tool(name: str = "t", *, params: list[ParamDef] | None = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="测试工具",
        params=params or [],
        source=ToolSource.BUILTIN,
    )


# 独立于实现的类型断言（Property 21 的预言机）：直接依据 spec 类型映射。
def _value_matches_type_oracle(value: Any, expected: ParamType) -> bool:
    if expected is ParamType.STRING:
        return isinstance(value, str)
    if expected is ParamType.NUMBER:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is ParamType.BOOLEAN:
        return isinstance(value, bool)
    if expected is ParamType.OBJECT:
        return isinstance(value, dict)
    if expected is ParamType.ARRAY:
        return isinstance(value, (list, tuple))
    return False


def _expected_valid(definition: ToolDefinition, params: dict[str, Any]) -> bool:
    """独立计算调用是否应当放行：必填齐备 且 已提供的同名参数类型匹配。"""
    for p in definition.params:
        if p.name not in params:
            if p.required:
                return False
            continue
        if not _value_matches_type_oracle(params[p.name], p.type):
            return False
    return True


# --------------------------------------------------------------------------- #
# Property 21（任务 8.2）：工具参数校验 iff 不变量                                #
# --------------------------------------------------------------------------- #
_NAME_POOL = ["a", "b", "c", "d", "e"]
# 调用参数键空间：包含已定义名 + 额外未知名，以覆盖"多余参数被忽略"分支。
_KEY_POOL = _NAME_POOL + ["x", "y", "z"]

_PY_VALUES = st.one_of(
    st.text(max_size=5),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.booleans(),
    st.dictionaries(st.text(max_size=3), st.integers(), max_size=3),
    st.lists(st.integers(), max_size=3),
    st.none(),
)


@st.composite
def _tool_definitions(draw: st.DrawFn) -> ToolDefinition:
    names = draw(st.lists(st.sampled_from(_NAME_POOL), unique=True, max_size=len(_NAME_POOL)))
    params = [
        ParamDef(
            name=n,
            type=draw(st.sampled_from(list(ParamType))),
            required=draw(st.booleans()),
        )
        for n in names
    ]
    return ToolDefinition(
        name="tool_under_test",
        description="随机生成的测试工具",
        params=params,
        source=ToolSource.BUILTIN,
    )


@settings(max_examples=200, deadline=None)
@given(
    definition=_tool_definitions(),
    call_params=st.dictionaries(st.sampled_from(_KEY_POOL), _PY_VALUES, max_size=len(_KEY_POOL)),
)
def test_property_21_validation_iff_required_present_and_types_match(
    definition: ToolDefinition,
    call_params: dict[str, Any],
) -> None:
    """Property 21：当且仅当所有必填参数齐备且各参数类型匹配时放行执行；
    否则拒绝调用、不执行工具并返回指明具体不符合项的校验错误。

    **Validates: Requirements 16.2, 16.3, 16.5**
    """
    registry = ToolRegistry()
    recorder = Recorder(result={"ok": True})
    registry.register(definition, recorder)

    expected = _expected_valid(definition, call_params)

    if expected:
        result = registry.invoke(definition.name, call_params)
        assert recorder.called is True
        assert result == {"ok": True}
    else:
        with pytest.raises(ToolValidationError) as exc_info:
            registry.invoke(definition.name, call_params)
        # 不执行工具 (Req 16.3, 16.5)。
        assert recorder.called is False
        # 错误指明具体不符合项 (Req 16.3)。
        details = exc_info.value.details
        assert details is not None
        assert details["tool"] == definition.name
        offending = set(details["missing"]) | {m["name"] for m in details["type_mismatch"]}
        assert offending, "校验错误必须列出至少一个具体不符合项"
        # 每个被指明的不符合项确实在定义中且确实不符合。
        defined = {p.name: p for p in definition.params}
        for missing_name in details["missing"]:
            assert defined[missing_name].required is True
            assert missing_name not in call_params
        for m in details["type_mismatch"]:
            assert m["name"] in defined
            assert not _value_matches_type_oracle(
                call_params[m["name"]], defined[m["name"]].type
            )


# --------------------------------------------------------------------------- #
# 单元测试：注册与冲突 (Req 16.1, 17.3)                                          #
# --------------------------------------------------------------------------- #
def test_register_duplicate_name_raises_and_preserves_original() -> None:
    registry = ToolRegistry()
    original = _simple_tool("dup", params=[ParamDef(name="q", type=ParamType.STRING, required=True)])
    original_handler = Recorder()
    registry.register(original, original_handler)

    conflicting = _simple_tool("dup", params=[])
    with pytest.raises(ToolNamingConflictError):
        registry.register(conflicting, Recorder())

    # 既有工具保持不变。
    kept = registry.get("dup")
    assert isinstance(kept, RegisteredTool)
    assert kept.definition is original
    assert kept.handler is original_handler
    assert len(registry.list_definitions()) == 1


# --------------------------------------------------------------------------- #
# 单元测试：未知工具 (Req 16.6)                                                  #
# --------------------------------------------------------------------------- #
def test_invoke_unknown_tool_raises_not_found() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError) as exc_info:
        registry.invoke("nope", {"q": "x"})
    assert exc_info.value.details == {"name": "nope"}


def test_get_unknown_tool_raises_not_found() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("missing")


# --------------------------------------------------------------------------- #
# 单元测试：缺失必填 (Req 16.3)                                                  #
# --------------------------------------------------------------------------- #
def test_missing_required_param_rejected_and_not_executed() -> None:
    registry = ToolRegistry()
    rec = Recorder()
    registry.register(
        _simple_tool("t", params=[ParamDef(name="q", type=ParamType.STRING, required=True)]),
        rec,
    )
    with pytest.raises(ToolValidationError) as exc_info:
        registry.invoke("t", {})
    assert rec.called is False
    assert "q" in exc_info.value.details["missing"]


# --------------------------------------------------------------------------- #
# 单元测试：类型不符 (Req 16.2, 16.3)                                            #
# --------------------------------------------------------------------------- #
def test_wrong_type_param_rejected_and_names_offender() -> None:
    registry = ToolRegistry()
    rec = Recorder()
    registry.register(
        _simple_tool("t", params=[ParamDef(name="q", type=ParamType.STRING, required=True)]),
        rec,
    )
    with pytest.raises(ToolValidationError) as exc_info:
        registry.invoke("t", {"q": 123})
    assert rec.called is False
    offenders = {m["name"] for m in exc_info.value.details["type_mismatch"]}
    assert "q" in offenders


def test_bool_not_accepted_as_number_but_int_and_float_are() -> None:
    registry = ToolRegistry()
    rec = Recorder()
    registry.register(
        _simple_tool("t", params=[ParamDef(name="n", type=ParamType.NUMBER, required=True)]),
        rec,
    )
    # bool 被拒绝。
    with pytest.raises(ToolValidationError):
        registry.invoke("t", {"n": True})
    assert rec.called is False

    # int 放行。
    assert registry.invoke("t", {"n": 7}) == "OK"
    # float 放行。
    rec.called = False
    assert registry.invoke("t", {"n": 3.14}) == "OK"


@pytest.mark.parametrize(
    "ptype, good, bad",
    [
        (ParamType.STRING, "s", 1),
        (ParamType.BOOLEAN, True, "true"),
        (ParamType.OBJECT, {"k": 1}, [1, 2]),
        (ParamType.ARRAY, [1, 2], {"k": 1}),
    ],
)
def test_type_matching_per_param_type(ptype: ParamType, good: Any, bad: Any) -> None:
    registry = ToolRegistry()
    registry.register(
        _simple_tool("t", params=[ParamDef(name="p", type=ptype, required=True)]),
        Recorder(),
    )
    assert registry.invoke("t", {"p": good}) == "OK"
    with pytest.raises(ToolValidationError):
        registry.invoke("t", {"p": bad})


# --------------------------------------------------------------------------- #
# 单元测试：未知参数被忽略；合法放行 (Req 16.5)                                   #
# --------------------------------------------------------------------------- #
def test_extra_unknown_params_are_ignored() -> None:
    registry = ToolRegistry()
    rec = Recorder(result={"done": 1})
    registry.register(
        _simple_tool("t", params=[ParamDef(name="q", type=ParamType.STRING, required=True)]),
        rec,
    )
    result = registry.invoke("t", {"q": "hi", "unexpected": 999})
    assert rec.called is True
    assert result == {"done": 1}
    # 句柄收到的是原始 params（含多余键）。
    assert rec.params == {"q": "hi", "unexpected": 999}


def test_optional_param_absent_is_valid() -> None:
    registry = ToolRegistry()
    rec = Recorder()
    registry.register(
        _simple_tool(
            "t",
            params=[
                ParamDef(name="q", type=ParamType.STRING, required=True),
                ParamDef(name="top_k", type=ParamType.NUMBER, required=False),
            ],
        ),
        rec,
    )
    assert registry.invoke("t", {"q": "hello"}) == "OK"
    assert rec.called is True


# --------------------------------------------------------------------------- #
# 单元测试：内置工具 (Req 16.4)                                                  #
# --------------------------------------------------------------------------- #
def test_build_default_registry_registers_four_builtins() -> None:
    registry = build_default_registry()
    names = sorted(t.name for t in registry.list_definitions())
    assert names == [
        "query_cls_log",
        "query_internal_docs",
        "query_prometheus_alarm",
        "send_msg",
    ]
    for definition in registry.list_definitions():
        assert definition.source is ToolSource.BUILTIN


def test_builtin_default_stub_handler_returns_structured_dict() -> None:
    registry = build_default_registry()
    out = registry.invoke("query_cls_log", {"query": "error"})
    assert out["status"] == "stub"
    assert out["tool"] == "query_cls_log"
    assert out["params"] == {"query": "error"}


def test_builtin_backend_is_injectable() -> None:
    sent: list[dict[str, Any]] = []

    def fake_send(params: dict[str, Any]) -> dict[str, Any]:
        sent.append(params)
        return {"delivered": True}

    registry = build_default_registry(send_msg=fake_send)
    result = registry.invoke("send_msg", {"target": "group-1", "message": "hi"})
    assert result == {"delivered": True}
    assert sent == [{"target": "group-1", "message": "hi"}]


def test_register_builtin_tools_conflicts_with_existing_name() -> None:
    registry = ToolRegistry()
    registry.register(_simple_tool("send_msg"), Recorder())
    with pytest.raises(ToolNamingConflictError):
        register_builtin_tools(registry)


def test_builtin_send_msg_validates_required_params() -> None:
    registry = build_default_registry()
    with pytest.raises(ToolValidationError) as exc_info:
        registry.invoke("send_msg", {"target": "g"})
    assert "message" in exc_info.value.details["missing"]


# --------------------------------------------------------------------------- #
# 单元测试：OpenAI Function Call schema                                          #
# --------------------------------------------------------------------------- #
def test_to_openai_schemas_well_formed_for_builtins() -> None:
    registry = build_default_registry()
    schemas = registry.to_openai_schemas()
    by_name = {s["function"]["name"]: s for s in schemas}

    assert set(by_name) == {
        "query_internal_docs",
        "query_cls_log",
        "query_prometheus_alarm",
        "send_msg",
    }

    send = by_name["send_msg"]
    assert send["type"] == "function"
    fn = send["function"]
    assert fn["description"]
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["target"] == {"type": "string"}
    assert params["properties"]["message"] == {"type": "string"}
    assert sorted(params["required"]) == ["message", "target"]


def test_to_openai_schemas_maps_all_param_types() -> None:
    registry = ToolRegistry()
    registry.register(
        _simple_tool(
            "t",
            params=[
                ParamDef(name="s", type=ParamType.STRING, required=True),
                ParamDef(name="n", type=ParamType.NUMBER, required=False),
                ParamDef(name="b", type=ParamType.BOOLEAN, required=False),
                ParamDef(name="o", type=ParamType.OBJECT, required=False),
                ParamDef(name="a", type=ParamType.ARRAY, required=True),
            ],
        ),
        Recorder(),
    )
    schema = registry.to_openai_schemas()[0]
    props = schema["function"]["parameters"]["properties"]
    assert props["s"] == {"type": "string"}
    assert props["n"] == {"type": "number"}
    assert props["b"] == {"type": "boolean"}
    assert props["o"] == {"type": "object"}
    assert props["a"] == {"type": "array"}
    assert sorted(schema["function"]["parameters"]["required"]) == ["a", "s"]

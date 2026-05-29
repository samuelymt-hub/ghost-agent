"""ToolDefinition / ParamDef 单元测试 (Req 16.1, 16.4, 17.1, 17.3)。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ghost_agent.models import ParamDef, ParamType, ToolDefinition, ToolSource


# ---------------------------------------------------------------------------
# ParamDef
# ---------------------------------------------------------------------------


def test_param_def_happy_path() -> None:
    p = ParamDef(name="query", type=ParamType.STRING, required=True)
    assert p.name == "query"
    assert p.type is ParamType.STRING
    assert p.required is True


def test_param_def_empty_name_rejected() -> None:
    with pytest.raises(ValidationError):
        ParamDef(name="", type=ParamType.STRING, required=True)


def test_param_def_invalid_type_rejected() -> None:
    with pytest.raises(ValidationError):
        ParamDef(name="x", type="UNKNOWN", required=True)


@pytest.mark.parametrize("missing", ["name", "type", "required"])
def test_param_def_missing_required_field_rejected(missing: str) -> None:
    payload = {"name": "x", "type": ParamType.STRING, "required": True}
    payload.pop(missing)
    with pytest.raises(ValidationError):
        ParamDef(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


def _builtin(**overrides: object) -> dict:
    base = {
        "name": "query_internal_docs",
        "description": "查询内部文档",
        "params": [
            ParamDef(name="query", type=ParamType.STRING, required=True),
            ParamDef(name="top_k", type=ParamType.NUMBER, required=False),
        ],
        "source": ToolSource.BUILTIN,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def test_tool_definition_happy_path_builtin() -> None:
    tool = ToolDefinition(**_builtin())
    assert tool.name == "query_internal_docs"
    assert tool.source is ToolSource.BUILTIN
    assert len(tool.params) == 2


def test_tool_definition_happy_path_no_params() -> None:
    tool = ToolDefinition(
        name="ping",
        description="健康检查",
        source=ToolSource.BUILTIN,
    )
    assert tool.params == []


def test_tool_definition_mcp_source_ok() -> None:
    tool = ToolDefinition(
        name="external.search",
        description="外部检索",
        source=ToolSource.MCP,
    )
    assert tool.source is ToolSource.MCP


def test_tool_definition_empty_name_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(**_builtin(name=""))


def test_tool_definition_empty_description_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(**_builtin(description=""))


def test_tool_definition_invalid_source_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(**_builtin(source="UNKNOWN"))


def test_tool_definition_rejects_duplicate_param_names() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolDefinition(
            **_builtin(
                params=[
                    ParamDef(name="q", type=ParamType.STRING, required=True),
                    ParamDef(name="q", type=ParamType.NUMBER, required=False),
                ]
            )
        )
    assert "重复" in str(exc_info.value) or "params" in str(exc_info.value).lower()


def test_tool_definition_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ToolDefinition(**_builtin(unknown="x"))


def test_tool_definition_json_round_trip_serializes_enums_as_strings() -> None:
    original = ToolDefinition(**_builtin())
    raw = original.model_dump_json()
    payload = json.loads(raw)
    assert payload["source"] == "BUILTIN"
    assert payload["params"][0]["type"] == "STRING"

    restored = ToolDefinition.model_validate_json(raw)
    assert restored == original

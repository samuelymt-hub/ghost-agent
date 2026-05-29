"""工具集相关数据模型 (Req 16.1, 16.2, 17.1)。

包含:

* :class:`ParamType`       —— 工具参数类型枚举 (Req 16.1, 16.2)。
* :class:`ToolSource`      —— 工具来源枚举（内置 / MCP, Req 16.4, 17.1）。
* :class:`ParamDef`        —— 单个参数定义。
* :class:`ToolDefinition`  —— 工具完整定义。

工具参数运行时校验（必填、类型）由 :mod:`ghost_agent.core.tool_registry`
负责 (Req 16.2, 16.3)；本模型只承载 DTO 结构与 ``ParamDef.name`` 在工具内
唯一的结构性约束。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParamType(str, Enum):
    """工具参数类型枚举 (Req 16.1, 16.2)。"""

    STRING = "STRING"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    OBJECT = "OBJECT"
    ARRAY = "ARRAY"


class ToolSource(str, Enum):
    """工具来源 (Req 16.4, 17.1)。"""

    BUILTIN = "BUILTIN"
    MCP = "MCP"


class ParamDef(BaseModel):
    """单个工具参数定义 (Req 16.1, 16.2)。"""

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        ...,
        min_length=1,
        description="参数名 (Req 16.1)。",
    )
    type: ParamType = Field(
        ...,
        description="参数类型 (Req 16.1, 16.2)。",
    )
    required: bool = Field(
        ...,
        description="是否必填 (Req 16.1, 16.2)。",
    )


class ToolDefinition(BaseModel):
    """工具完整定义 (Req 16.1, 16.4, 17.1, 17.3)。

    ``name`` 在 :class:`Tool_Registry` 全局唯一 (Req 17.3)，由注册表在 register
    时校验；本模型在结构上仅保证 ``params`` 内部 ``name`` 不重复。
    """

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        ...,
        min_length=1,
        description="工具名，工具集内唯一 (Req 16.1, 17.3)。",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="功能描述 (Req 16.1)。",
    )
    params: list[ParamDef] = Field(
        default_factory=list,
        description="参数定义列表。",
    )
    source: ToolSource = Field(
        ...,
        description="工具来源：内置或 MCP (Req 16.4, 17.1)。",
    )

    @model_validator(mode="after")
    def _check_unique_param_names(self) -> "ToolDefinition":
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            raise ValueError(
                f"ToolDefinition({self.name!r}).params 中存在重复参数名: {names}"
            )
        return self


__all__ = [
    "ParamType",
    "ToolSource",
    "ParamDef",
    "ToolDefinition",
]

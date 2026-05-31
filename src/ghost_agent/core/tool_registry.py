"""Tool_Registry（工具集）实现 (Req 16.1–16.6)。

本模块提供:

* :class:`RegisteredTool` —— 工具定义 (:class:`ToolDefinition`) 与其执行
  句柄 (``handler``) 的绑定。
* :class:`ToolRegistry`   —— 维护工具集合，负责注册、参数校验、调用分发，
  并能产出 OpenAI 风格 Function Call schema 供 ``ChatModel.bind_tools`` 使用。
* :func:`register_builtin_tools` / :func:`build_default_registry` —— 注册四个
  内置工具（query_internal_docs、query_cls_log、query_prometheus_alarm、
  send_msg, Req 16.4），后端实现以可注入回调形式提供，默认返回轻量 stub，
  便于当前可测试、后续（任务 16）接线真实外部系统。

设计要点：
- **唯一名称 (Req 16.1, 17.3)**：``register`` 遇到重名直接抛
  :class:`ToolNamingConflictError`，已注册同名工具保持不变。
- **参数校验 (Req 16.2, 16.3)**：``invoke`` 调用前按工具的 :class:`ParamDef`
  列表校验——必填齐备 + 已提供参数类型匹配；任一不符即抛
  :class:`ToolValidationError`，并在 ``details`` 中给出具体不符合项；**不执行**
  工具句柄。未知/多余参数采取宽松策略（忽略），与常见 Function Call 行为一致。
- **不存在的工具 (Req 16.6)**：``invoke`` / ``get`` 命中不存在工具名时抛
  :class:`ToolNotFoundError`。
- **放行执行 (Req 16.5)**：工具存在且参数通过校验时执行句柄并返回其结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ghost_agent.models.errors import (
    ToolNamingConflictError,
    ToolNotFoundError,
    ToolValidationError,
)
from ghost_agent.models.tool import (
    ParamDef,
    ParamType,
    ToolDefinition,
    ToolSource,
)

__all__ = [
    "RegisteredTool",
    "ToolRegistry",
    "register_builtin_tools",
    "build_default_registry",
]

#: 工具执行句柄类型：接收已通过校验的参数字典，返回任意结果。
ToolHandler = Callable[[dict[str, Any]], Any]


# --------------------------------------------------------------------------- #
# 类型校验映射 (Req 16.2)                                                       #
# --------------------------------------------------------------------------- #
def _matches_type(value: Any, expected: ParamType) -> bool:
    """判断 ``value`` 的 Python 运行时类型是否匹配声明的 :class:`ParamType`。

    映射规则：
        STRING  -> ``str``
        NUMBER  -> ``int`` 或 ``float``，但 **排除** ``bool``（``bool`` 是 ``int``
                   的子类，语义上不应被当作数值）
        BOOLEAN -> ``bool``
        OBJECT  -> ``dict``
        ARRAY   -> ``list`` 或 ``tuple``
    """
    if expected is ParamType.STRING:
        return isinstance(value, str)
    if expected is ParamType.NUMBER:
        # bool 是 int 子类，需显式排除。
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is ParamType.BOOLEAN:
        return isinstance(value, bool)
    if expected is ParamType.OBJECT:
        return isinstance(value, dict)
    if expected is ParamType.ARRAY:
        return isinstance(value, (list, tuple))
    # 理论不可达：ParamType 已穷举。保守返回 False。
    return False  # pragma: no cover


#: ParamType -> JSON Schema 类型字符串，用于 OpenAI Function Call schema。
_JSON_SCHEMA_TYPES: dict[ParamType, str] = {
    ParamType.STRING: "string",
    ParamType.NUMBER: "number",
    ParamType.BOOLEAN: "boolean",
    ParamType.OBJECT: "object",
    ParamType.ARRAY: "array",
}


# --------------------------------------------------------------------------- #
# RegisteredTool                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegisteredTool:
    """一个已注册工具：定义 + 执行句柄。"""

    definition: ToolDefinition
    handler: ToolHandler


# --------------------------------------------------------------------------- #
# ToolRegistry                                                                  #
# --------------------------------------------------------------------------- #
class ToolRegistry:
    """工具集：注册、校验、调用分发 (Req 16)。"""

    def __init__(self) -> None:
        # 名称 -> 已注册工具；插入顺序即注册顺序。
        self._tools: dict[str, RegisteredTool] = {}

    # ------------------------------------------------------------------ #
    # 注册 (Req 16.1, 17.3)                                               #
    # ------------------------------------------------------------------ #
    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        """注册一个工具：维护唯一名称、参数定义与功能描述 (Req 16.1)。

        Args:
            definition: 工具定义（名称在工具集内唯一）。
            handler: 工具执行句柄，接收已通过校验的参数字典。

        Raises:
            ToolNamingConflictError: 工具名已存在 (Req 17.3)；此时既有工具保持
                不变，新工具不被注册。
        """
        name = definition.name
        if name in self._tools:
            raise ToolNamingConflictError(
                f"工具名 {name!r} 已注册，拒绝重复注册并保留既有工具",
                details={"name": name},
            )
        self._tools[name] = RegisteredTool(definition=definition, handler=handler)

    # ------------------------------------------------------------------ #
    # 查询                                                                #
    # ------------------------------------------------------------------ #
    def has(self, name: str) -> bool:
        """工具名是否已注册。"""
        return name in self._tools

    def get(self, name: str) -> RegisteredTool:
        """按名称返回已注册工具。

        Raises:
            ToolNotFoundError: 工具名不存在 (Req 16.6)。
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                f"工具 {name!r} 不存在",
                details={"name": name},
            )
        return tool

    def list_definitions(self) -> list[ToolDefinition]:
        """按注册顺序返回所有工具定义。"""
        return [t.definition for t in self._tools.values()]

    # ------------------------------------------------------------------ #
    # 参数校验 (Req 16.2, 16.3)                                           #
    # ------------------------------------------------------------------ #
    def validate_params(self, name: str, params: dict[str, Any]) -> None:
        """按工具参数定义校验调用参数；不符合则抛错并指明具体不符合项。

        校验内容：
            * 每个 ``required=True`` 的参数必须出现在 ``params`` 中 (Req 16.2)。
            * 每个出现在 ``params`` 中且与某定义参数同名的值，其运行时类型必须
              匹配该参数声明的 :class:`ParamType` (Req 16.2)。
            * 未在定义中出现的多余参数：宽松忽略（不视为不符合）。

        Args:
            name: 工具名。
            params: 调用参数字典。

        Raises:
            ToolNotFoundError: 工具名不存在 (Req 16.6)。
            ToolValidationError: 存在缺失必填或类型不匹配；``details`` 含
                ``missing``（缺失必填参数名列表）与 ``type_mismatch``（含
                ``name`` / ``expected`` / ``actual`` 的列表）(Req 16.3)。
        """
        tool = self.get(name)
        defs = tool.definition.params

        missing: list[str] = []
        type_mismatch: list[dict[str, str]] = []

        for param in defs:
            present = param.name in params
            if not present:
                if param.required:
                    missing.append(param.name)
                # 非必填且缺省：跳过类型校验。
                continue
            value = params[param.name]
            if not _matches_type(value, param.type):
                type_mismatch.append(
                    {
                        "name": param.name,
                        "expected": param.type.value,
                        "actual": type(value).__name__,
                    }
                )

        if missing or type_mismatch:
            raise ToolValidationError(
                self._format_validation_message(name, missing, type_mismatch),
                details={
                    "tool": name,
                    "missing": missing,
                    "type_mismatch": type_mismatch,
                },
            )

    @staticmethod
    def _format_validation_message(
        name: str,
        missing: list[str],
        type_mismatch: list[dict[str, str]],
    ) -> str:
        """构造指明具体不符合项的人类可读校验错误消息 (Req 16.3)。"""
        parts: list[str] = []
        if missing:
            parts.append(f"缺少必填参数: {', '.join(missing)}")
        if type_mismatch:
            details = ", ".join(
                f"{m['name']}(期望 {m['expected']}, 实际 {m['actual']})"
                for m in type_mismatch
            )
            parts.append(f"参数类型不符: {details}")
        return f"工具 {name!r} 参数校验失败：" + "；".join(parts)

    # ------------------------------------------------------------------ #
    # 调用分发 (Req 16.5, 16.6)                                           #
    # ------------------------------------------------------------------ #
    def invoke(self, name: str, params: dict[str, Any]) -> Any:
        """校验参数后执行工具并返回其响应。

        Args:
            name: 工具名。
            params: 调用参数字典。

        Returns:
            工具句柄的返回值 (Req 16.5)。

        Raises:
            ToolNotFoundError: 工具名不存在；不执行任何工具 (Req 16.6)。
            ToolValidationError: 参数不符合定义；拒绝调用、**不执行**该工具
                (Req 16.3)。
        """
        tool = self.get(name)
        # 校验失败将抛 ToolValidationError，从而不会执行下方句柄。
        self.validate_params(name, params)
        return tool.handler(params)

    # ------------------------------------------------------------------ #
    # OpenAI Function Call schema (供 ChatModel.bind_tools 使用)           #
    # ------------------------------------------------------------------ #
    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """产出 OpenAI 风格 Function Call schema 列表。

        形如::

            {
              "type": "function",
              "function": {
                "name": ...,
                "description": ...,
                "parameters": {
                  "type": "object",
                  "properties": {<param>: {"type": <json_type>}},
                  "required": [<required param names>]
                }
              }
            }

        可直接传给 ``ChatModel.bind_tools(registry.to_openai_schemas())``。
        """
        schemas: list[dict[str, Any]] = []
        for tool in self._tools.values():
            definition = tool.definition
            properties: dict[str, Any] = {}
            required: list[str] = []
            for param in definition.params:
                properties[param.name] = {"type": _JSON_SCHEMA_TYPES[param.type]}
                if param.required:
                    required.append(param.name)
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return schemas


# --------------------------------------------------------------------------- #
# 内置工具 (Req 16.4)                                                           #
# --------------------------------------------------------------------------- #
def _default_stub(tool_name: str) -> ToolHandler:
    """构造一个轻量、无副作用的占位句柄。

    真实集成（内部文档检索 / CLS 日志 / Prometheus 告警 / IM 群消息上报）
    属于外部系统，此处不可用；占位句柄返回结构化字典，便于当前测试，且
    可在任务 16 通过注入真实后端替换。
    """

    def _handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"status": "stub", "tool": tool_name, "params": params}

    return _handler


def _wrap_backend(tool_name: str, backend: ToolHandler | None) -> ToolHandler:
    """若提供了后端回调则使用之，否则回退到占位句柄。"""
    if backend is None:
        return _default_stub(tool_name)
    return backend


# 四个内置工具的参数 schema (Req 16.4)。
_BUILTIN_DEFINITIONS: dict[str, ToolDefinition] = {
    "query_internal_docs": ToolDefinition(
        name="query_internal_docs",
        description="检索内部文档/处理手册，返回与查询相关的内容片段。",
        params=[
            ParamDef(name="query", type=ParamType.STRING, required=True),
            ParamDef(name="top_k", type=ParamType.NUMBER, required=False),
        ],
        source=ToolSource.BUILTIN,
    ),
    "query_cls_log": ToolDefinition(
        name="query_cls_log",
        description="查询 CLS 日志，按查询语句与时间范围返回匹配日志。",
        params=[
            ParamDef(name="query", type=ParamType.STRING, required=True),
            ParamDef(name="time_range", type=ParamType.STRING, required=False),
            ParamDef(name="limit", type=ParamType.NUMBER, required=False),
        ],
        source=ToolSource.BUILTIN,
    ),
    "query_prometheus_alarm": ToolDefinition(
        name="query_prometheus_alarm",
        description="查询 Prometheus 告警/指标，按查询语句与时间范围返回结果。",
        params=[
            ParamDef(name="query", type=ParamType.STRING, required=True),
            ParamDef(name="time_range", type=ParamType.STRING, required=False),
        ],
        source=ToolSource.BUILTIN,
    ),
    "send_msg": ToolDefinition(
        name="send_msg",
        description="向目标群组/会话发送消息（用于排查结论上报）。",
        params=[
            ParamDef(name="target", type=ParamType.STRING, required=True),
            ParamDef(name="message", type=ParamType.STRING, required=True),
        ],
        source=ToolSource.BUILTIN,
    ),
}


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    docs_query: ToolHandler | None = None,
    cls_query: ToolHandler | None = None,
    prometheus_query: ToolHandler | None = None,
    send_msg: ToolHandler | None = None,
) -> None:
    """向 ``registry`` 注册四个内置工具 (Req 16.4)。

    每个后端均为可注入回调，默认回退到无副作用占位句柄；这样当前即可测试，
    后续（任务 16）可注入真实实现。

    Args:
        registry: 目标工具集。
        docs_query: query_internal_docs 后端。
        cls_query: query_cls_log 后端。
        prometheus_query: query_prometheus_alarm 后端。
        send_msg: send_msg 后端。

    Raises:
        ToolNamingConflictError: 若 ``registry`` 中已存在同名工具 (Req 17.3)。
    """
    backends: dict[str, ToolHandler | None] = {
        "query_internal_docs": docs_query,
        "query_cls_log": cls_query,
        "query_prometheus_alarm": prometheus_query,
        "send_msg": send_msg,
    }
    for name, definition in _BUILTIN_DEFINITIONS.items():
        registry.register(definition, _wrap_backend(name, backends[name]))


def build_default_registry(
    *,
    docs_query: ToolHandler | None = None,
    cls_query: ToolHandler | None = None,
    prometheus_query: ToolHandler | None = None,
    send_msg: ToolHandler | None = None,
) -> ToolRegistry:
    """创建并返回一个已注册四个内置工具的 :class:`ToolRegistry` (Req 16.4)。"""
    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        docs_query=docs_query,
        cls_query=cls_query,
        prometheus_query=prometheus_query,
        send_msg=send_msg,
    )
    return registry

"""MCP_Client（MCP 客户端）实现 (Req 17.1–17.6)。

基于模型上下文协议（Model Context Protocol）接入外部工具，将 MCP 服务端暴露的
工具清单注册到 :class:`~ghost_agent.core.tool_registry.ToolRegistry`，并在 Agent
调用 MCP 工具时通过协议转发请求、回传响应。

职责（语言无关契约见 design.md「MCP_Client」一节）：

* :meth:`MCPClient.connect` —— 建立会话、获取工具清单，将每个工具的名称/参数
  定义/功能描述注册到 Tool_Registry (Req 17.1)。返回本次实际注册的工具定义列表。
* :meth:`MCPClient.invoke`  —— 通过 MCP 会话转发工具调用并回传响应 (Req 17.2)。
* :meth:`MCPClient.list_registered_mcp_tools` —— 返回本客户端已注册的 MCP 工具名。

关键约束：

- **命名冲突 (Req 17.3)**：发现的 MCP 工具名若与 Tool_Registry 已注册工具冲突，
  则跳过该工具（``ToolRegistry.register`` 抛 :class:`ToolNamingConflictError`），
  记录冲突信息，**保留已注册的同名工具不变**，并继续处理其余工具；工具集内
  该名称对应的工具总数不变（不覆盖、不重复）。
- **连接失败 (Req 17.4)**：会话建立或工具清单获取失败时抛
  :class:`McpConnectFailedError`，**不注册任何**该服务端工具，Tool_Registry 中
  既有内置工具不受影响。
- **工具错误响应 (Req 17.5)**：MCP 工具返回错误响应，或调用抛出非超时异常时，
  ``invoke`` 抛 :class:`McpToolError`（保留原始异常为 ``__cause__``）。
- **调用超时 (Req 17.6)**：调用耗时达到配置的超时时间仍未返回时抛
  :class:`McpToolTimeoutError`；此为单次调用级错误，不影响其余工具可用性。

设计要点（与 ``milvus_client`` / ``chat_model`` 一致的惰性 seam 模式）：

- **会话注入 seam**：构造函数接收可选的 ``session_factory`` 回调，用于构建/返回
  一个 MCP 会话对象（暴露 ``list_tools()`` 与 ``call_tool(name, arguments)``
  两个方法即可）。默认工厂惰性导入 ``mcp`` SDK；测试注入假会话，完全离线、无
  网络、无需真实 MCP 服务端。
- **超时实现**：:func:`_call_with_timeout` 借助 ``ThreadPoolExecutor.submit`` +
  ``future.result(timeout=...)``，使慢调用抛 ``TimeoutError``，再由 ``invoke``
  转换为 :class:`McpToolTimeoutError`。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Protocol, runtime_checkable

from ghost_agent.models.errors import (
    McpConnectFailedError,
    McpToolError,
    McpToolTimeoutError,
    ToolNamingConflictError,
)
from ghost_agent.models.tool import (
    ParamDef,
    ParamType,
    ToolDefinition,
    ToolSource,
)

from ghost_agent.core.tool_registry import ToolRegistry

__all__ = [
    "DEFAULT_MCP_TIMEOUT_SECONDS",
    "McpSession",
    "MCPClient",
]

logger = logging.getLogger(__name__)

#: 模块默认的 MCP 工具调用超时（秒）。当 ``config`` 未提供专用 MCP 超时项时，
#: 以此为默认值；可通过构造参数 ``timeout`` 覆盖 (Req 17.6)。
DEFAULT_MCP_TIMEOUT_SECONDS: float = 30.0


# --------------------------------------------------------------------------- #
# 会话协议（最小使用面）                                                         #
# --------------------------------------------------------------------------- #
@runtime_checkable
class McpSession(Protocol):
    """MCP 会话最小接口（本客户端仅依赖这两个方法）。

    真实实现由 ``mcp`` SDK 会话提供；测试以同名鸭子类型的假对象注入。
    """

    def list_tools(self) -> Any:
        """返回工具清单：可为工具描述符列表，或带 ``.tools`` 属性的结果对象。"""
        ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用名为 ``name`` 的工具并返回其响应。"""
        ...


#: 会话工厂类型：无参调用返回一个 :class:`McpSession`。
SessionFactory = Callable[[], McpSession]


# --------------------------------------------------------------------------- #
# JSON Schema 类型 -> ParamType 映射                                            #
# --------------------------------------------------------------------------- #
_JSON_TYPE_TO_PARAM: dict[str, ParamType] = {
    "string": ParamType.STRING,
    "number": ParamType.NUMBER,
    "integer": ParamType.NUMBER,
    "boolean": ParamType.BOOLEAN,
    "object": ParamType.OBJECT,
    "array": ParamType.ARRAY,
}


def _json_type_to_param_type(json_type: Any) -> ParamType:
    """将 JSON Schema 的 ``type`` 字段映射为 :class:`ParamType`，未知类型回退 STRING。"""
    if isinstance(json_type, str):
        return _JSON_TYPE_TO_PARAM.get(json_type.lower(), ParamType.STRING)
    return ParamType.STRING


# --------------------------------------------------------------------------- #
# 内部工具：超时识别                                                            #
# --------------------------------------------------------------------------- #
def _is_timeout(exc: BaseException) -> bool:
    """判断异常是否为"超时"语义。

    - 标准库 :class:`TimeoutError`（``concurrent.futures.TimeoutError`` 在
      Python 3.11+ 即为内建 ``TimeoutError`` 的别名）；
    - 或异常类型名中（不区分大小写）包含 ``timeout``。
    """
    if isinstance(exc, TimeoutError):
        return True
    return "timeout" in type(exc).__name__.lower()


def _call_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """在独立线程中执行 ``fn``，超过 ``timeout`` 秒未完成则抛 ``TimeoutError``。

    使用 ``ThreadPoolExecutor`` 隔离同步阻塞调用；超时后取消 future 并由调用方
    转换为 :class:`McpToolTimeoutError`。``fn`` 内部抛出的异常会原样向上传播。
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            future.cancel()
            # 统一抛内建 TimeoutError，便于 _is_timeout 识别。
            raise TimeoutError(
                f"MCP 工具调用超过 {timeout} 秒仍未返回"
            ) from exc


# --------------------------------------------------------------------------- #
# MCPClient                                                                     #
# --------------------------------------------------------------------------- #
class MCPClient:
    """MCP 客户端：连接服务端、注册工具、转发调用 (Req 17)。"""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        timeout: float | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        """构造 MCP 客户端。

        Args:
            registry: 接收 MCP 工具注册的 Tool_Registry。
            timeout: 单次工具调用超时（秒）；为 ``None`` 时取模块默认
                :data:`DEFAULT_MCP_TIMEOUT_SECONDS`（Req 17.6）。
            session_factory: 会话工厂回调，无参返回一个 :class:`McpSession`。
                为 ``None`` 时使用默认工厂（惰性导入 ``mcp`` SDK）。测试通常
                注入返回假会话的工厂以离线运行。
        """
        self._registry = registry
        self._timeout: float = (
            timeout if timeout is not None else DEFAULT_MCP_TIMEOUT_SECONDS
        )
        self._session_factory = session_factory
        # 已连接会话缓存；invoke 复用此会话转发调用。
        self._session: McpSession | None = None
        # 本客户端注册成功的 MCP 工具名（按注册顺序）。
        self._registered_mcp_names: list[str] = []
        # 记录命名冲突的工具名（Req 17.3）。
        self._conflicts: list[str] = []

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def timeout(self) -> float:
        """配置的单次工具调用超时（秒，Req 17.6）。"""
        return self._timeout

    @property
    def conflicts(self) -> list[str]:
        """连接过程中检测到的命名冲突工具名列表（Req 17.3）。"""
        return list(self._conflicts)

    # ------------------------------------------------------------------ #
    # 测试 seam：构建底层会话                                              #
    # ------------------------------------------------------------------ #
    def _build_session(self) -> McpSession:
        """构建 MCP 会话：优先使用注入的工厂，否则惰性构建真实会话。"""
        if self._session_factory is not None:
            return self._session_factory()
        return self._build_real_session()

    def _build_real_session(self) -> McpSession:
        """惰性导入 ``mcp`` SDK 并构建真实会话。

        真实 MCP 会话依赖具体传输（stdio/SSE）与服务端连接配置，属于集成层关注点
        （在应用启动装配阶段接线）。此处仅校验 SDK 可用，并提示需通过
        ``session_factory`` 注入连接好的会话；任何失败均封装为
        :class:`McpConnectFailedError`（Req 17.4）。
        """
        try:
            import mcp  # noqa: F401  # 惰性导入：确认 SDK 可用
        except Exception as exc:  # noqa: BLE001 - SDK 不可用统一封装
            raise McpConnectFailedError(
                "mcp SDK 不可用，无法建立默认 MCP 会话"
            ) from exc
        raise McpConnectFailedError(
            "未提供 session_factory：真实 MCP 会话需注入已连接服务端的会话工厂"
        )

    # ------------------------------------------------------------------ #
    # 连接与注册 (Req 17.1, 17.3, 17.4)                                   #
    # ------------------------------------------------------------------ #
    def connect(self) -> list[ToolDefinition]:
        """连接 MCP 服务端、获取工具清单并注册到 Tool_Registry。

        Returns:
            本次 ``connect`` 实际注册成功的 :class:`ToolDefinition` 列表（不含
            因命名冲突被跳过的工具，Req 17.3）。

        Raises:
            McpConnectFailedError: 会话建立或工具清单获取失败 (Req 17.4)；此时
                不注册任何该服务端工具，Tool_Registry 中既有工具不受影响。
        """
        # —— 1) 建立会话 ——
        try:
            session = self._build_session()
        except McpConnectFailedError:
            raise
        except Exception as exc:  # noqa: BLE001 - 连接异常统一封装 (Req 17.4)
            raise McpConnectFailedError("连接 MCP 服务端失败") from exc

        # —— 2) 获取工具清单 ——
        try:
            raw = session.list_tools()
        except Exception as exc:  # noqa: BLE001 - 清单获取失败视为连接失败 (Req 17.4)
            raise McpConnectFailedError(
                "获取 MCP 服务端工具清单失败"
            ) from exc

        descriptors = self._extract_descriptors(raw)

        # 清单获取成功后才缓存会话，供 invoke 复用。
        self._session = session

        # —— 3) 逐个注册工具 ——
        registered: list[ToolDefinition] = []
        for desc in descriptors:
            try:
                definition = self._to_tool_definition(desc)
            except Exception as exc:  # noqa: BLE001 - 跳过无法解析的描述符
                logger.warning("跳过无法解析的 MCP 工具描述符: %s", exc)
                continue

            try:
                self._registry.register(
                    definition, self._make_handler(definition.name)
                )
            except ToolNamingConflictError:
                # Req 17.3：拒绝注册、记录冲突、保留既有同名工具不变、继续其余。
                logger.warning(
                    "MCP 工具名 %r 与 Tool_Registry 已注册工具冲突，"
                    "跳过注册并保留既有工具",
                    definition.name,
                )
                self._conflicts.append(definition.name)
                continue

            self._registered_mcp_names.append(definition.name)
            registered.append(definition)

        return registered

    # ------------------------------------------------------------------ #
    # 调用转发 (Req 17.2, 17.5, 17.6)                                     #
    # ------------------------------------------------------------------ #
    def invoke(self, name: str, params: dict[str, Any]) -> Any:
        """通过 MCP 会话转发工具调用并回传响应 (Req 17.2)。

        Args:
            name: MCP 工具名。
            params: 调用参数。

        Returns:
            MCP 工具的响应。

        Raises:
            McpToolError: 会话未建立、调用抛非超时异常，或 MCP 工具返回错误响应
                (Req 17.5)。
            McpToolTimeoutError: 调用耗时达到配置超时仍未返回 (Req 17.6)；该错误
                仅影响本次调用，不影响其余工具可用性。
        """
        session = self._session
        if session is None:
            raise McpToolError(
                "MCP 会话未建立，请先调用 connect()",
                details={"tool": name},
            )

        try:
            result = _call_with_timeout(
                lambda: session.call_tool(name, params), self._timeout
            )
        except Exception as exc:  # noqa: BLE001 - 区分超时与一般错误后统一封装
            if _is_timeout(exc):
                raise McpToolTimeoutError(
                    f"MCP 工具 {name!r} 调用超时（>{self._timeout}s）",
                    details={"tool": name, "timeout_seconds": self._timeout},
                ) from exc
            raise McpToolError(
                f"MCP 工具 {name!r} 调用失败",
                details={"tool": name},
            ) from exc

        # Req 17.5：MCP 工具在服务端返回错误响应。
        if self._is_error_response(result):
            raise McpToolError(
                f"MCP 工具 {name!r} 返回错误响应",
                details={"tool": name},
            )
        return result

    # ------------------------------------------------------------------ #
    # 查询                                                                #
    # ------------------------------------------------------------------ #
    def list_registered_mcp_tools(self) -> list[str]:
        """返回本客户端已注册成功的 MCP 工具名（按注册顺序）。"""
        return list(self._registered_mcp_names)

    # ------------------------------------------------------------------ #
    # 内部工具：句柄、描述符解析、错误响应识别                              #
    # ------------------------------------------------------------------ #
    def _make_handler(self, tool_name: str) -> Callable[[dict[str, Any]], Any]:
        """为某 MCP 工具构造注册到 Tool_Registry 的执行句柄。

        句柄回调进入 :meth:`invoke`，从而经由已缓存的 MCP 会话转发调用。
        """

        def _handler(params: dict[str, Any]) -> Any:
            return self.invoke(tool_name, params)

        return _handler

    @staticmethod
    def _extract_descriptors(raw: Any) -> list[Any]:
        """从 ``list_tools()`` 返回值中提取工具描述符列表。

        兼容两种形态：带 ``.tools`` 属性的结果对象（如 SDK 的 ``ListToolsResult``）
        与直接的列表/元组。
        """
        tools = getattr(raw, "tools", None)
        if tools is not None:
            return list(tools)
        if isinstance(raw, (list, tuple)):
            return list(raw)
        return []

    @staticmethod
    def _desc_get(desc: Any, key: str) -> Any:
        """从描述符（dict 或对象）中取字段，缺失返回 ``None``。"""
        if isinstance(desc, dict):
            return desc.get(key)
        return getattr(desc, key, None)

    def _to_tool_definition(self, desc: Any) -> ToolDefinition:
        """将单个 MCP 工具描述符转换为 :class:`ToolDefinition`（source=MCP）。

        防御性地处理缺失/异常 schema：无 ``properties`` 时按无参数处理。
        """
        name = self._desc_get(desc, "name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"MCP 工具描述符缺少有效 name: {desc!r}")

        description = self._desc_get(desc, "description")
        if not isinstance(description, str) or not description:
            description = f"MCP 工具 {name}"

        schema = (
            self._desc_get(desc, "input_schema")
            or self._desc_get(desc, "inputSchema")
            or self._desc_get(desc, "parameters")
        )
        params = self._schema_to_params(schema)

        return ToolDefinition(
            name=name,
            description=description,
            params=params,
            source=ToolSource.MCP,
        )

    @staticmethod
    def _schema_to_params(schema: Any) -> list[ParamDef]:
        """将 JSON-Schema 风格的输入 schema 转换为 :class:`ParamDef` 列表。

        - ``properties`` 缺失或非 dict -> 返回空参数列表。
        - ``required`` 列表中的参数名标记 ``required=True``。
        - 参数类型按 :func:`_json_type_to_param_type` 映射，未知类型回退 STRING。
        - 跳过空参数名（防御性）。
        """
        if not isinstance(schema, dict):
            return []
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return []

        raw_required = schema.get("required")
        required_set: set[str] = (
            set(raw_required)
            if isinstance(raw_required, (list, tuple, set))
            else set()
        )

        params: list[ParamDef] = []
        for pname, pschema in properties.items():
            if not isinstance(pname, str) or not pname:
                continue
            json_type = pschema.get("type") if isinstance(pschema, dict) else None
            params.append(
                ParamDef(
                    name=pname,
                    type=_json_type_to_param_type(json_type),
                    required=pname in required_set,
                )
            )
        return params

    @staticmethod
    def _is_error_response(result: Any) -> bool:
        """判断 MCP 工具响应是否表示错误 (Req 17.5)。

        兼容多种形态：对象的 ``isError`` / ``is_error`` 属性，或 dict 的同名键。
        """
        for attr in ("isError", "is_error"):
            if isinstance(result, dict):
                if bool(result.get(attr)):
                    return True
            else:
                value = getattr(result, attr, None)
                if bool(value):
                    return True
        return False

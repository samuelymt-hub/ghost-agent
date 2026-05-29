"""统一错误模型与领域异常类。

本模块定义系统所有横切错误类型，供 API 接口层、Agent 编排层与各核心组件
统一使用：

* :class:`ErrorCode`     —— 机器可读错误码常量（字符串枚举）。
* :class:`ErrorResponse` —— 对外返回的结构化错误响应（Pydantic v2 模型）。
* :class:`GhostAgentError` —— 所有领域异常的基类，可一键转换为
  :class:`ErrorResponse`。
* 各领域异常子类（``EmptyMessageError``、``FileTooLargeError`` 等）—— 每个错
  误码一一对应一个异常类，便于调用方按类型捕获。

模块刻意保持极轻量、零外部依赖（除 Pydantic），不引用 ``ghost_agent.config``，
以便被任何更下层的模块复用。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    # 枚举与基础模型
    "ErrorCode",
    "ErrorResponse",
    "GhostAgentError",
    # API 接口层异常
    "EmptyMessageError",
    "MessageTooLongError",
    "ChatTimeoutError",
    "GenerationError",
    "GenerationTimeoutError",
    "SSEIdleTimeoutError",
    "EmptyFileError",
    "UnsupportedFileTypeError",
    "FileTooLargeError",
    "IngestFailedError",
    "MissingTroubleshootFieldsError",
    "WebhookSignatureInvalidError",
    "ServiceUnavailableError",
    "DuplicateTroubleshootingError",
    # 组件层异常
    "ParseFailedError",
    "ParseTimeoutError",
    "EmptyContentError",
    "SplitFailedError",
    "QueryEmptyError",
    "QueryEmbeddingFailedError",
    "DimensionMismatchError",
    "VectorDatabaseUnavailableError",
    "TemplateNotFoundError",
    "ToolValidationError",
    "ToolNotFoundError",
    "ToolNamingConflictError",
    "McpToolError",
    "McpToolTimeoutError",
    "McpConnectFailedError",
    "MemorySummarizeFailedError",
    "SourceFileNotFoundError",
    "SyncFailedError",
    "SendMsgFailedError",
    "TargetGroupUnavailableError",
    "ReplanLimitReachedError",
    "IterationLimitReachedError",
    "UnsupportedTechStackError",
]


class ErrorCode(str, Enum):
    """统一错误码常量。

    每个成员的值即对应错误码字符串本身（如
    ``ErrorCode.EMPTY_MESSAGE.value == "EMPTY_MESSAGE"``），便于 JSON 序列化
    与跨语言契约对齐。
    """

    # ---------------------------------------------------------------- API 层
    EMPTY_MESSAGE = "EMPTY_MESSAGE"                    # 1.3, 2.6
    MESSAGE_TOO_LONG = "MESSAGE_TOO_LONG"              # 1.6
    CHAT_TIMEOUT = "CHAT_TIMEOUT"                      # 1.5
    GENERATION_ERROR = "GENERATION_ERROR"              # 1.7, 9.5, 10.7
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"          # 9.6, 10.7
    SSE_IDLE_TIMEOUT = "SSE_IDLE_TIMEOUT"              # 2.8
    EMPTY_FILE = "EMPTY_FILE"                          # 3.3
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"    # 3.4
    FILE_TOO_LARGE = "FILE_TOO_LARGE"                  # 3.5
    INGEST_FAILED = "INGEST_FAILED"                    # 3.7
    MISSING_TROUBLESHOOT_FIELDS = "MISSING_TROUBLESHOOT_FIELDS"  # 4.3, 15.4
    WEBHOOK_SIGNATURE_INVALID = "WEBHOOK_SIGNATURE_INVALID"      # 4.5
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"        # 4.6
    DUPLICATE_TROUBLESHOOTING = "DUPLICATE_TROUBLESHOOTING"      # 15.5

    # ----------------------------------------------------------- 组件 / 内核
    PARSE_FAILED = "PARSE_FAILED"                      # 5.3
    PARSE_TIMEOUT = "PARSE_TIMEOUT"                    # 5.5
    EMPTY_CONTENT = "EMPTY_CONTENT"                    # 5.6, 6.6
    SPLIT_FAILED = "SPLIT_FAILED"                      # 6.7
    QUERY_EMPTY = "QUERY_EMPTY"                        # 8.7
    QUERY_EMBEDDING_FAILED = "QUERY_EMBEDDING_FAILED"  # 8.8
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"          # 21.4
    VECTOR_DB_UNAVAILABLE = "VECTOR_DB_UNAVAILABLE"    # 21.5
    TEMPLATE_NOT_FOUND = "TEMPLATE_NOT_FOUND"          # 20.5
    TOOL_VALIDATION_ERROR = "TOOL_VALIDATION_ERROR"    # 16.3
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"                  # 16.6
    TOOL_NAMING_CONFLICT = "TOOL_NAMING_CONFLICT"      # 17.3
    MCP_TOOL_ERROR = "MCP_TOOL_ERROR"                  # 17.5
    MCP_TOOL_TIMEOUT = "MCP_TOOL_TIMEOUT"              # 17.6
    MCP_CONNECT_FAILED = "MCP_CONNECT_FAILED"          # 17.4
    MEMORY_SUMMARIZE_FAILED = "MEMORY_SUMMARIZE_FAILED"  # 18.4
    SOURCE_FILE_NOT_FOUND = "SOURCE_FILE_NOT_FOUND"    # 22.6
    SYNC_FAILED = "SYNC_FAILED"                        # 22.5
    SEND_MSG_FAILED = "SEND_MSG_FAILED"                # 14.4
    TARGET_GROUP_UNAVAILABLE = "TARGET_GROUP_UNAVAILABLE"  # 14.6
    REPLAN_LIMIT_REACHED = "REPLAN_LIMIT_REACHED"      # 13.5
    ITERATION_LIMIT_REACHED = "ITERATION_LIMIT_REACHED"  # 10.5
    UNSUPPORTED_TECH_STACK = "UNSUPPORTED_TECH_STACK"  # 23.7


class ErrorResponse(BaseModel):
    """对外暴露的统一错误响应模型。

    与 design.md 中"统一错误模型"完全对齐：

    * ``error_code``：机器可读错误码（``ErrorCode`` 枚举或裸字符串均可，
      序列化时一律为字符串）。
    * ``message``：人类可读说明。
    * ``details``：可选附加信息字典（如受支持类型列表、最大长度、已有任
      务标识等）。
    """

    model_config = ConfigDict(use_enum_values=True)

    error_code: str
    message: str
    details: dict[str, Any] | None = None

    @field_validator("error_code", mode="before")
    @classmethod
    def _coerce_error_code(cls, value: Any) -> str:
        """允许调用方传入 ``ErrorCode`` 枚举或裸字符串。"""
        if isinstance(value, ErrorCode):
            return value.value
        if isinstance(value, str):
            return value
        raise TypeError(
            "error_code 必须是 ErrorCode 枚举或字符串，"
            f"实际收到: {type(value).__name__}"
        )


class GhostAgentError(Exception):
    """所有领域异常的基类。

    子类需通过类属性 ``error_code`` 绑定一个 :class:`ErrorCode` 成员；上层
    捕获后可调用 :meth:`to_response` 生成结构化 :class:`ErrorResponse`。
    """

    #: 子类必须覆盖此类属性，将自身映射到一个具体的 ``ErrorCode``。
    error_code: ClassVar[ErrorCode]

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_response(self) -> ErrorResponse:
        """将异常转换为对外可序列化的 :class:`ErrorResponse`。"""
        return ErrorResponse(
            error_code=self.error_code,
            message=self.message,
            details=self.details,
        )

    def __str__(self) -> str:  # pragma: no cover - 透明委托给 message
        return self.message


# ---------------------------------------------------------------------------
# API 接口层异常 —— Req 1, 2, 3, 4, 14, 15
# ---------------------------------------------------------------------------


class EmptyMessageError(GhostAgentError):
    """用户消息缺失或去除首尾空白后长度为 0 (Req 1.3, 2.6)。"""

    error_code = ErrorCode.EMPTY_MESSAGE


class MessageTooLongError(GhostAgentError):
    """用户消息长度超过上限 (Req 1.6)。"""

    error_code = ErrorCode.MESSAGE_TOO_LONG


class ChatTimeoutError(GhostAgentError):
    """``/chat`` 接口处理超时 (Req 1.5)。"""

    error_code = ErrorCode.CHAT_TIMEOUT


class GenerationError(GhostAgentError):
    """Conversation_Agent / Chat_Model 生成阶段错误 (Req 1.7, 9.5, 10.7)。"""

    error_code = ErrorCode.GENERATION_ERROR


class GenerationTimeoutError(GhostAgentError):
    """模型生成超时 (Req 9.6, 10.7)。"""

    error_code = ErrorCode.GENERATION_TIMEOUT


class SSEIdleTimeoutError(GhostAgentError):
    """SSE 推送空闲超时 (Req 2.8)。"""

    error_code = ErrorCode.SSE_IDLE_TIMEOUT


class EmptyFileError(GhostAgentError):
    """上传文件缺失或为空 (Req 3.3)。"""

    error_code = ErrorCode.EMPTY_FILE


class UnsupportedFileTypeError(GhostAgentError):
    """上传文档类型不在受支持类型集合内 (Req 3.4)。"""

    error_code = ErrorCode.UNSUPPORTED_FILE_TYPE


class FileTooLargeError(GhostAgentError):
    """上传文件大小超过单文件上限 (Req 3.5)。"""

    error_code = ErrorCode.FILE_TOO_LARGE


class IngestFailedError(GhostAgentError):
    """入库管线整体失败 (Req 3.7)。"""

    error_code = ErrorCode.INGEST_FAILED


class MissingTroubleshootFieldsError(GhostAgentError):
    """``/ai_ops`` 触发请求缺少告警信息或排查目标 (Req 4.3, 15.4)。"""

    error_code = ErrorCode.MISSING_TROUBLESHOOT_FIELDS


class WebhookSignatureInvalidError(GhostAgentError):
    """webhook 来源签名校验失败 (Req 4.5)。"""

    error_code = ErrorCode.WEBHOOK_SIGNATURE_INVALID


class ServiceUnavailableError(GhostAgentError):
    """Ops_Agent 不可用或并发达上限 (Req 4.6)。"""

    error_code = ErrorCode.SERVICE_UNAVAILABLE


class DuplicateTroubleshootingError(GhostAgentError):
    """同一排查目标已有进行中流程 (Req 15.5)。"""

    error_code = ErrorCode.DUPLICATE_TROUBLESHOOTING


# ---------------------------------------------------------------------------
# 组件层异常 —— Req 5–13, 14, 16–22
# ---------------------------------------------------------------------------


class ParseFailedError(GhostAgentError):
    """Loader 文档解析失败 (Req 5.3)。"""

    error_code = ErrorCode.PARSE_FAILED


class ParseTimeoutError(GhostAgentError):
    """Loader 文档解析超时 (Req 5.5)。"""

    error_code = ErrorCode.PARSE_TIMEOUT


class EmptyContentError(GhostAgentError):
    """解析后或待分片内容为空 (Req 5.6, 6.6)。"""

    error_code = ErrorCode.EMPTY_CONTENT


class SplitFailedError(GhostAgentError):
    """Transformer 分片策略不可应用或失败 (Req 6.7)。"""

    error_code = ErrorCode.SPLIT_FAILED


class QueryEmptyError(GhostAgentError):
    """Retriever 查询为空或去空白后为空 (Req 8.7)。"""

    error_code = ErrorCode.QUERY_EMPTY


class QueryEmbeddingFailedError(GhostAgentError):
    """Retriever 查询向量生成失败 (Req 8.8)。"""

    error_code = ErrorCode.QUERY_EMBEDDING_FAILED


class DimensionMismatchError(GhostAgentError):
    """向量维度与 Embedding_Model 输出维度不一致 (Req 21.4)。"""

    error_code = ErrorCode.DIMENSION_MISMATCH


class VectorDatabaseUnavailableError(GhostAgentError):
    """向量库连接超时或不可用 (Req 21.5)。"""

    error_code = ErrorCode.VECTOR_DB_UNAVAILABLE


class TemplateNotFoundError(GhostAgentError):
    """提示词模板按名称查找不到 (Req 20.5)。"""

    error_code = ErrorCode.TEMPLATE_NOT_FOUND


class ToolValidationError(GhostAgentError):
    """Tool_Registry 工具参数校验失败 (Req 16.3)。"""

    error_code = ErrorCode.TOOL_VALIDATION_ERROR


class ToolNotFoundError(GhostAgentError):
    """Tool_Registry 调用了不存在的工具名 (Req 16.6)。"""

    error_code = ErrorCode.TOOL_NOT_FOUND


class ToolNamingConflictError(GhostAgentError):
    """MCP 工具与已注册工具命名冲突 (Req 17.3)。"""

    error_code = ErrorCode.TOOL_NAMING_CONFLICT


class McpToolError(GhostAgentError):
    """MCP 工具返回错误响应 (Req 17.5)。"""

    error_code = ErrorCode.MCP_TOOL_ERROR


class McpToolTimeoutError(GhostAgentError):
    """MCP 工具调用超时 (Req 17.6)。"""

    error_code = ErrorCode.MCP_TOOL_TIMEOUT


class McpConnectFailedError(GhostAgentError):
    """MCP 服务端连接失败 (Req 17.4)。"""

    error_code = ErrorCode.MCP_CONNECT_FAILED


class MemorySummarizeFailedError(GhostAgentError):
    """Memory_Module 总结/写入长期记忆失败 (Req 18.4)。"""

    error_code = ErrorCode.MEMORY_SUMMARIZE_FAILED


class SourceFileNotFoundError(GhostAgentError):
    """知识库待移除来源文件不存在 (Req 22.6)。"""

    error_code = ErrorCode.SOURCE_FILE_NOT_FOUND


class SyncFailedError(GhostAgentError):
    """知识库同步任一阶段失败 (Req 22.5)。"""

    error_code = ErrorCode.SYNC_FAILED


class SendMsgFailedError(GhostAgentError):
    """Ops_Agent send_msg 上报失败重试达上限 (Req 14.4)。"""

    error_code = ErrorCode.SEND_MSG_FAILED


class TargetGroupUnavailableError(GhostAgentError):
    """目标群组未配置或不存在 (Req 14.6)。"""

    error_code = ErrorCode.TARGET_GROUP_UNAVAILABLE


class ReplanLimitReachedError(GhostAgentError):
    """Replanner_Agent 重规划次数达上限 (Req 13.5)。"""

    error_code = ErrorCode.REPLAN_LIMIT_REACHED


class IterationLimitReachedError(GhostAgentError):
    """Conversation_Agent ReAct 迭代次数达上限 (Req 10.5)。"""

    error_code = ErrorCode.ITERATION_LIMIT_REACHED


class UnsupportedTechStackError(GhostAgentError):
    """启动时配置的技术栈不在受支持集合内 (Req 23.7)。"""

    error_code = ErrorCode.UNSUPPORTED_TECH_STACK

"""集中式配置管理模块。

集中管理所有运行时可配置参数（超时、Top-K、重试次数、分片长度等），
并提供启动期技术栈校验（仅允许 python/go/java 三者之一）。

设计要点：
- 基于 pydantic-settings v2，支持从环境变量与 ``.env`` 加载。
- 使用 ``Field(..., ge=..., le=...)`` 进行参数范围校验，越界即拒绝启动。
- ``validate_tech_stack`` 可在 app 启动装配阶段独立调用，
  对应 Requirements 23.6 / 23.7（单次部署仅启用一种技术栈，
  不在 {python, go, java} 集合内时拒绝以该技术栈启动）。
- ``get_settings`` 通过 ``functools.lru_cache`` 暴露进程级单例，
  避免重复解析环境变量带来的开销。

本模块刻意不依赖 ``ghost_agent`` 包内任何其他子模块，避免循环导入。
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import FrozenSet

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# --------------------------------------------------------------------------- #
# 技术栈枚举与独立校验函数                                                       #
# --------------------------------------------------------------------------- #
class TechStack(str, Enum):
    """系统支持的技术栈实现选项 (Requirements 23.1/23.2/23.3)。"""

    PYTHON = "python"
    GO = "go"
    JAVA = "java"


# 受支持技术栈的取值集合，用于错误信息与校验
_SUPPORTED_TECH_STACKS: FrozenSet[str] = frozenset({ts.value for ts in TechStack})


def validate_tech_stack(value: str) -> TechStack:
    """校验技术栈取值并返回对应枚举。

    Requirements 23.7: 部署配置指定的技术栈不属于 {python, go, java}
    时，必须拒绝以该技术栈启动并返回指明不支持该技术栈的错误信息。

    Args:
        value: 用户配置（不区分大小写）。

    Returns:
        对应的 ``TechStack`` 枚举成员。

    Raises:
        ValueError: 当取值不在受支持集合内时抛出。
    """
    if value is None:
        raise ValueError("技术栈不能为空，仅支持 python/go/java")

    normalized = str(value).strip().lower()
    if normalized not in _SUPPORTED_TECH_STACKS:
        raise ValueError(
            f"不支持该技术栈: {value}，仅支持 python/go/java"
        )
    return TechStack(normalized)


# --------------------------------------------------------------------------- #
# Settings 主体                                                                 #
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """运行时配置集合。

    所有字段对应需求文档中提及的"可配置参数"，并尽量给出与 design.md
    "超时与重试参数（可配置）" 表一致的默认值与取值范围。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----------------------- 技术栈与部署 ---------------------------------- #
    tech_stack: TechStack = Field(
        default=TechStack.PYTHON,
        description="本次部署启用的技术栈，仅允许 python/go/java 三者之一",
    )

    # ----------------------- API 接口层超时 -------------------------------- #
    chat_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="/chat 接口处理超时（秒），Req 1.5",
    )
    sse_idle_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="/chat_stream SSE 空闲超时（秒），Req 2.8",
    )

    # ----------------------- Embedding / 索引 ------------------------------ #
    embedding_max_retries: int = Field(
        default=3,
        ge=0,
        le=5,
        description="嵌入失败的最大重试次数（0–5，默认 3），Req 7.3",
    )
    embedding_dim: int = Field(
        default=2560,
        gt=0,
        description="Embedding 输出维度（Doubao-embedding-text-240715），Req 21.2/21.4",
    )
    embedding_max_input_length: int = Field(
        default=4096,
        gt=0,
        description="Embedding 最大输入长度（字符/Token 上限），Req 6.4",
    )

    # ----------------------- ReAct 循环 ------------------------------------ #
    react_max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="ReAct 最大迭代次数（1–50，默认 10），Req 10.5",
    )
    tool_call_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="工具调用超时（1–300 秒，默认 30），Req 10.6",
    )
    model_call_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=120.0,
        description="模型调用超时（1–120 秒，默认 60），Req 10.7",
    )

    # ----------------------- 运维 Agent 重规划 ----------------------------- #
    max_replan_count: int = Field(
        default=10,
        ge=1,
        le=50,
        description="最大重规划次数（1–50，默认 10），Req 13.5",
    )
    max_plan_steps: int = Field(
        default=20,
        ge=1,
        le=100,
        description="单个执行计划的最大步骤数上限，Req 11.2",
    )

    # ----------------------- 检索召回 -------------------------------------- #
    history_message_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="历史消息 Top-K（1–50），Req 10.1 / 19.4",
    )
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="文档检索 Top-K（1–100，默认 5），Req 8.2",
    )
    min_similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="检索召回的最小相似度阈值（[0.0, 1.0]），Req 8.2 / 8.6",
    )

    # ----------------------- 记忆模块 -------------------------------------- #
    short_term_memory_limit: int = Field(
        default=20,
        ge=1,
        description="短期记忆保留条数上限（≥1），Req 18.2 / 18.3",
    )

    # ----------------------- 分片策略 -------------------------------------- #
    chunk_min_length: int = Field(
        default=100,
        ge=1,
        description="单个 Chunk 的最小长度（字符），Req 6.1",
    )
    chunk_max_length: int = Field(
        default=1000,
        ge=1,
        description="单个 Chunk 的最大长度（字符），Req 6.1 / 6.4",
    )

    # ----------------------- 文件上传 -------------------------------------- #
    supported_document_types: FrozenSet[str] = Field(
        default=frozenset({"txt", "md", "pdf", "docx", "html"}),
        description="受支持的文档类型集合，Req 3.4 / 5.1",
    )
    max_file_size_bytes: int = Field(
        default=50 * 1024 * 1024,  # 50 MiB
        gt=0,
        description="单文件大小上限（字节），Req 3.5",
    )

    # ----------------------- 字段级 validators ----------------------------- #
    @field_validator("tech_stack", mode="before")
    @classmethod
    def _coerce_tech_stack(cls, value: object) -> TechStack:
        """允许从环境变量传入字符串（不区分大小写）；非法值给出中文错误。"""
        if isinstance(value, TechStack):
            return value
        if value is None:
            return TechStack.PYTHON
        return validate_tech_stack(str(value))

    @field_validator("supported_document_types", mode="before")
    @classmethod
    def _normalize_document_types(cls, value: object) -> FrozenSet[str]:
        """支持以逗号分隔字符串（环境变量友好）或 set/list/tuple 输入。

        值会统一为小写并去除首尾空白与前导 ``.``，便于与文件扩展名比较。
        """
        if value is None:
            return frozenset()
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (set, frozenset, list, tuple)):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            raise ValueError("supported_document_types 必须为字符串或集合类型")
        normalized = {item.lower().lstrip(".") for item in items if item}
        return frozenset(normalized)

    # ----------------------- 跨字段校验 ------------------------------------ #
    @model_validator(mode="after")
    def _validate_cross_field(self) -> "Settings":
        """跨字段一致性校验。

        - chunk_min_length 必须严格小于 chunk_max_length；
        - chunk_max_length 不得超过 embedding_max_input_length，否则
          单分片可能超过 Embedding_Model 的输入上限（Req 6.4）。
        """
        if self.chunk_min_length >= self.chunk_max_length:
            raise ValueError(
                f"分片配置非法：chunk_min_length ({self.chunk_min_length}) "
                f"必须严格小于 chunk_max_length ({self.chunk_max_length})"
            )
        if self.chunk_max_length > self.embedding_max_input_length:
            raise ValueError(
                f"分片配置非法：chunk_max_length ({self.chunk_max_length}) "
                f"不得超过 embedding_max_input_length "
                f"({self.embedding_max_input_length})"
            )
        return self


# --------------------------------------------------------------------------- #
# 单例工厂                                                                      #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程级共享的 ``Settings`` 单例。

    通过 ``lru_cache`` 缓存首次解析结果，后续调用直接复用。
    在测试中如需刷新，可调用 ``get_settings.cache_clear()``。
    """
    return Settings()


__all__ = [
    "TechStack",
    "Settings",
    "validate_tech_stack",
    "get_settings",
]

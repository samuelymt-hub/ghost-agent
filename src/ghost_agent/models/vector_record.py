"""向量记录 VectorRecord 数据模型 (Req 21.3, 19.1)。

VectorRecord 是写入 :class:`Vector_Database` (Milvus) 的最小持久化单元，统一
承载文档分片向量（``DOC_CHUNK``）与对话消息向量（``MESSAGE``）。

注意：向量维度与 Embedding_Model 输出维度的一致性校验（Req 21.4）由
:mod:`ghost_agent.vector_db.vector_store` 在写入时完成；这里仅约束 ``vector``
非空，避免重复职责。
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _new_id() -> str:
    """生成 VectorRecord 的唯一主键。"""
    return str(uuid.uuid4())


class VectorType(str, Enum):
    """向量类型枚举 (Req 21.3)。

    * ``DOC_CHUNK`` —— 文档分片向量。
    * ``MESSAGE``   —— 对话消息向量。
    """

    DOC_CHUNK = "DOC_CHUNK"
    MESSAGE = "MESSAGE"


class VectorRecord(BaseModel):
    """向量库统一记录结构 (Req 21.3)。

    持久化到 Milvus 的每条记录都必须同时携带原始文本、来源标识与向量类型，
    供检索结果回填以及按来源/类型过滤使用。
    """

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    id: str = Field(
        default_factory=_new_id,
        description="向量记录主键。",
    )
    vector: list[float] = Field(
        ...,
        min_length=1,
        description="向量值；维度等于 Embedding_Model 输出维度 (Req 21.4，由 vector_store 校验)。",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="原始文本 (Req 21.3)。",
    )
    source_id: str = Field(
        ...,
        min_length=1,
        description="来源标识：source_file_id (DOC_CHUNK) 或 session_id (MESSAGE)。",
    )
    vector_type: VectorType = Field(
        ...,
        description="向量类型：DOC_CHUNK 或 MESSAGE (Req 21.3)。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="附加元数据（如 seq/offset、role/timestamp 等）。",
    )


__all__ = ["VectorRecord", "VectorType"]

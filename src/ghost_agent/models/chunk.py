"""文档分片 Chunk 数据模型 (Req 6.3, 6.4, 22.1)。

本模块仅定义文档分片的 DTO，不承载分片业务逻辑——分片策略与长度约束的运行
时校验由 :mod:`ghost_agent.core.transformer` 在执行期完成。这里只覆盖结构性
约束（字段必填、序号与位置非负、起止位置顺序合理）。
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _new_id() -> str:
    """生成全局唯一的 chunk 标识。"""
    return str(uuid.uuid4())


class Chunk(BaseModel):
    """文档分片：源文档切分后的最小可索引单元。

    对应 design.md "Data Models / Chunk" 章节，覆盖 Req 6.3（必含来源文件标识、
    顺序序号与起止位置）、6.4（超长 Chunk 二次切分时记录父分片标识）、22.1
    （为每个 Chunk 附来源文件标识）。
    """

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    chunk_id: str = Field(
        default_factory=_new_id,
        description="全局唯一的分片标识。",
    )
    source_file_id: str = Field(
        ...,
        min_length=1,
        description="来源文件标识 (Req 6.3, 22.1)。",
    )
    seq: int = Field(
        ...,
        ge=0,
        description="该 Chunk 在源文档中的顺序序号 (Req 6.3)。",
    )
    start_offset: int = Field(
        ...,
        ge=0,
        description="该 Chunk 在源文档中的起始位置 (Req 6.3)。",
    )
    end_offset: int = Field(
        ...,
        ge=0,
        description="该 Chunk 在源文档中的结束位置 (Req 6.3)。",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="分片文本，不允许为空字符串。",
    )
    parent_chunk_id: str | None = Field(
        default=None,
        description="若由超长 Chunk 二次切分产生，则记录父分片标识 (Req 6.4)。",
    )

    @model_validator(mode="after")
    def _check_offsets(self) -> "Chunk":
        if self.start_offset > self.end_offset:
            raise ValueError(
                "start_offset 不得大于 end_offset："
                f"start={self.start_offset}, end={self.end_offset}"
            )
        return self


__all__ = ["Chunk"]

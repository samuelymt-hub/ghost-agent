"""入库任务 IngestTask 数据模型 (Req 3.2, 3.6, 3.7)。

IngestTask 表示一次 ``/upload_file`` 文档入库的端到端任务，状态在 PENDING →
RUNNING → (COMPLETED / FAILED) 间流转。COMPLETED 时必须提供 ``chunk_count``
(Req 3.6)；FAILED 时必须提供 ``failure_reason`` (Req 3.7)。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _new_id() -> str:
    """生成入库任务的唯一标识。"""
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。"""
    return datetime.now(timezone.utc)


class IngestTaskStatus(str, Enum):
    """入库任务状态 (Req 3.2, 3.6, 3.7)。"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IngestTask(BaseModel):
    """入库任务记录 (Req 3.2, 3.6, 3.7)。"""

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    task_id: str = Field(
        default_factory=_new_id,
        description="唯一入库任务标识 (Req 3.2)。",
    )
    file_name: str = Field(
        ...,
        min_length=1,
        description="原始文件名。",
    )
    file_format: str = Field(
        ...,
        min_length=1,
        description="文件格式（如 ``txt``、``md``、``pdf``）。",
    )
    status: IngestTaskStatus = Field(
        default=IngestTaskStatus.PENDING,
        description="任务状态。",
    )
    chunk_count: int | None = Field(
        default=None,
        ge=0,
        description="COMPLETED 时提供的成功写入分片数 (Req 3.6)。",
    )
    failure_reason: str | None = Field(
        default=None,
        description="FAILED 时提供的失败原因 (Req 3.7)。",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="任务创建时间 (UTC)。",
    )

    @model_validator(mode="after")
    def _check_terminal_fields(self) -> "IngestTask":
        if self.status is IngestTaskStatus.COMPLETED and self.chunk_count is None:
            raise ValueError(
                "IngestTask.status=COMPLETED 时必须提供 chunk_count (Req 3.6)"
            )
        if self.status is IngestTaskStatus.FAILED and (
            self.failure_reason is None or not self.failure_reason.strip()
        ):
            raise ValueError(
                "IngestTask.status=FAILED 时必须提供非空 failure_reason (Req 3.7)"
            )
        return self


__all__ = ["IngestTaskStatus", "IngestTask"]

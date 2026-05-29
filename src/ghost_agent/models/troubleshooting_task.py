"""排查任务 TroubleshootingTask 与相关数据模型 (Req 4.2, 14.x, 15.x, 13.4)。

包含:

* :class:`TriggerType`                —— 触发方式枚举 (Req 15.1–15.3)。
* :class:`TroubleshootingTaskStatus`  —— 排查任务状态枚举 (Req 4.2, 14.5)。
* :class:`ReportStatus`               —— 上报状态枚举 (Req 14.4–14.6)。
* :class:`AlarmInfo`                  —— 告警信息载荷 (Req 4.3, 15.4)。
* :class:`AnalysisSummary`            —— 三段式分析总结 (Req 14.2)；空内容
  以模块级常量 :data:`NO_CONTENT` 占位。
* :class:`TroubleshootingTask`        —— 排查任务主体。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

#: 分析总结中"无可填充内容"的明确占位符 (Req 14.2)。
NO_CONTENT: Final[str] = "NO_CONTENT"


def _new_id() -> str:
    """生成排查任务的唯一标识。"""
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。"""
    return datetime.now(timezone.utc)


class TriggerType(str, Enum):
    """排查任务触发方式 (Req 15.1, 15.2, 15.3)。"""

    MANUAL = "MANUAL"
    SCHEDULED = "SCHEDULED"
    WEBHOOK = "WEBHOOK"


class TroubleshootingTaskStatus(str, Enum):
    """排查任务状态 (Req 4.2, 14.5)。"""

    ACCEPTED = "ACCEPTED"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    REPLANNING = "REPLANNING"
    DONE = "DONE"
    FAILED = "FAILED"
    REPORTED = "REPORTED"


class ReportStatus(str, Enum):
    """上报状态 (Req 14.4, 14.5, 14.6)。"""

    NOT_REPORTED = "NOT_REPORTED"
    REPORTED = "REPORTED"
    REPORT_FAILED = "REPORT_FAILED"
    SKIPPED = "SKIPPED"


class AlarmInfo(BaseModel):
    """告警信息载荷 (Req 4.3, 15.4)。

    字段保持最小化以适配多种告警源：``message`` 为人类可读必填描述，
    ``raw`` 用于承载原始结构化负载，便于后续诊断追溯。
    """

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    source: str | None = Field(
        default=None,
        description="告警来源系统（如 prometheus、cls）。",
    )
    level: str | None = Field(
        default=None,
        description="告警级别（如 INFO/WARN/CRITICAL）。",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="告警人类可读描述。",
    )
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="告警原始结构化负载（保留扩展空间）。",
    )


class AnalysisSummary(BaseModel):
    """运维 Agent 输出的三段式分析总结 (Req 14.2)。

    总结固定包含根因分析、处理建议与已执行操作记录三个部分；任一部分若无可填
    充内容则使用 :data:`NO_CONTENT` 占位以满足 Req 14.2 中"对无内容部分以
    明确无内容说明标注"的要求。
    """

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    root_cause: str = Field(
        default=NO_CONTENT,
        min_length=1,
        description="根因分析；无内容时为 NO_CONTENT。",
    )
    suggestions: str = Field(
        default=NO_CONTENT,
        min_length=1,
        description="处理建议；无内容时为 NO_CONTENT。",
    )
    executed_actions: str = Field(
        default=NO_CONTENT,
        min_length=1,
        description="已执行操作记录；无内容时为 NO_CONTENT。",
    )


class TroubleshootingTask(BaseModel):
    """排查任务主体 (Req 4.2, 13.4, 14, 15)。

    ``replan_count`` 仅在结构上要求 ``>= 0``；与配置上限的对比由
    :mod:`ghost_agent.agents.replanner` 在运行时进行 (Req 13.5)。
    """

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    task_id: str = Field(
        default_factory=_new_id,
        description="唯一排查任务标识 (Req 4.2)。",
    )
    trigger_type: TriggerType = Field(
        ...,
        description="触发方式 (Req 15.1–15.3)。",
    )
    target: str = Field(
        ...,
        min_length=1,
        description="排查目标，并发去重键 (Req 15.5)。",
    )
    alarm: AlarmInfo = Field(
        ...,
        description="告警信息 (Req 4.3, 15.4)。",
    )
    status: TroubleshootingTaskStatus = Field(
        default=TroubleshootingTaskStatus.ACCEPTED,
        description="任务状态。",
    )
    replan_count: int = Field(
        default=0,
        ge=0,
        description="重规划次数 (Req 13.4, 13.5)。",
    )
    summary: AnalysisSummary | None = Field(
        default=None,
        description="分析结果总结 (Req 14.2)。",
    )
    report_status: ReportStatus | None = Field(
        default=None,
        description="上报状态 (Req 14.4–14.6)。",
    )
    reported_at: datetime | None = Field(
        default=None,
        description="上报成功时间 (Req 14.5)。",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="任务创建时间 (UTC)。",
    )


__all__ = [
    "NO_CONTENT",
    "TriggerType",
    "TroubleshootingTaskStatus",
    "ReportStatus",
    "AlarmInfo",
    "AnalysisSummary",
    "TroubleshootingTask",
]

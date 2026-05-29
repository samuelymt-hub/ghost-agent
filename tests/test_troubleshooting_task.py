"""TroubleshootingTask 与相关模型单元测试 (Req 4.2, 14.x, 15.x)。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ghost_agent.models import (
    NO_CONTENT,
    AlarmInfo,
    AnalysisSummary,
    ReportStatus,
    TriggerType,
    TroubleshootingTask,
    TroubleshootingTaskStatus,
)


# ---------------------------------------------------------------------------
# AlarmInfo
# ---------------------------------------------------------------------------


def test_alarm_info_minimal_payload() -> None:
    a = AlarmInfo(message="High CPU on host-1")
    assert a.message == "High CPU on host-1"
    assert a.source is None
    assert a.level is None
    assert a.raw == {}


def test_alarm_info_full_payload() -> None:
    a = AlarmInfo(
        source="prometheus",
        level="CRITICAL",
        message="High CPU",
        raw={"value": 99.9},
    )
    assert a.source == "prometheus"
    assert a.raw == {"value": 99.9}


def test_alarm_info_rejects_empty_message() -> None:
    with pytest.raises(ValidationError):
        AlarmInfo(message="")


# ---------------------------------------------------------------------------
# AnalysisSummary
# ---------------------------------------------------------------------------


def test_analysis_summary_default_uses_no_content_marker() -> None:
    s = AnalysisSummary()
    assert s.root_cause == NO_CONTENT
    assert s.suggestions == NO_CONTENT
    assert s.executed_actions == NO_CONTENT


def test_analysis_summary_partial_override() -> None:
    s = AnalysisSummary(root_cause="磁盘 IO 高")
    assert s.root_cause == "磁盘 IO 高"
    assert s.suggestions == NO_CONTENT
    assert s.executed_actions == NO_CONTENT


def test_analysis_summary_rejects_empty_string() -> None:
    """空字符串不是合法占位；应使用 NO_CONTENT 显式标注无内容。"""
    with pytest.raises(ValidationError):
        AnalysisSummary(root_cause="")


# ---------------------------------------------------------------------------
# TroubleshootingTask
# ---------------------------------------------------------------------------


def _alarm() -> AlarmInfo:
    return AlarmInfo(message="High CPU")


def _payload(**overrides: object) -> dict:
    base = {
        "trigger_type": TriggerType.MANUAL,
        "target": "host-1",
        "alarm": _alarm(),
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def test_task_default_status_and_replan_count() -> None:
    t = TroubleshootingTask(**_payload())
    assert t.status is TroubleshootingTaskStatus.ACCEPTED
    assert t.replan_count == 0
    assert t.summary is None
    assert t.report_status is None
    assert t.reported_at is None
    assert t.created_at.tzinfo is not None
    assert isinstance(t.task_id, str) and t.task_id


def test_task_missing_target_rejected() -> None:
    with pytest.raises(ValidationError):
        TroubleshootingTask(
            trigger_type=TriggerType.MANUAL, alarm=_alarm()
        )  # type: ignore[call-arg]


def test_task_empty_target_rejected() -> None:
    with pytest.raises(ValidationError):
        TroubleshootingTask(**_payload(target=""))


def test_task_invalid_trigger_type_rejected() -> None:
    with pytest.raises(ValidationError):
        TroubleshootingTask(**_payload(trigger_type="UNKNOWN"))


def test_task_invalid_status_rejected() -> None:
    with pytest.raises(ValidationError):
        TroubleshootingTask(**_payload(status="UNKNOWN"))


def test_task_negative_replan_count_rejected() -> None:
    with pytest.raises(ValidationError):
        TroubleshootingTask(**_payload(replan_count=-1))


def test_task_with_summary_and_report_status() -> None:
    summary = AnalysisSummary(
        root_cause="磁盘 IO 高",
        suggestions="扩容 IOPS",
        executed_actions="已通知值班",
    )
    t = TroubleshootingTask(
        **_payload(
            status=TroubleshootingTaskStatus.REPORTED,
            summary=summary,
            report_status=ReportStatus.REPORTED,
        )
    )
    assert t.summary == summary
    assert t.report_status is ReportStatus.REPORTED


def test_task_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        TroubleshootingTask(**_payload(unknown="x"))


def test_task_json_round_trip_serializes_enums_as_strings() -> None:
    original = TroubleshootingTask(
        **_payload(
            trigger_type=TriggerType.WEBHOOK,
            status=TroubleshootingTaskStatus.EXECUTING,
            summary=AnalysisSummary(root_cause="test"),
            report_status=ReportStatus.NOT_REPORTED,
        )
    )
    raw = original.model_dump_json()
    payload = json.loads(raw)
    assert payload["trigger_type"] == "WEBHOOK"
    assert payload["status"] == "EXECUTING"
    assert payload["report_status"] == "NOT_REPORTED"

    restored = TroubleshootingTask.model_validate_json(raw)
    assert restored == original

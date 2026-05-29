"""IngestTask 数据模型单元测试 (Req 3.2, 3.6, 3.7)。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ghost_agent.models import IngestTask, IngestTaskStatus


def _payload(**overrides: object) -> dict:
    base = {
        "file_name": "manual.md",
        "file_format": "md",
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def test_default_status_is_pending() -> None:
    t = IngestTask(**_payload())
    assert t.status is IngestTaskStatus.PENDING
    assert t.chunk_count is None
    assert t.failure_reason is None
    assert isinstance(t.task_id, str) and t.task_id


def test_running_status_does_not_require_chunk_count_or_reason() -> None:
    t = IngestTask(**_payload(status=IngestTaskStatus.RUNNING))
    assert t.status is IngestTaskStatus.RUNNING


def test_completed_requires_chunk_count() -> None:
    with pytest.raises(ValidationError) as exc_info:
        IngestTask(**_payload(status=IngestTaskStatus.COMPLETED))
    assert "chunk_count" in str(exc_info.value)


def test_completed_with_chunk_count_ok() -> None:
    t = IngestTask(
        **_payload(status=IngestTaskStatus.COMPLETED, chunk_count=42)
    )
    assert t.chunk_count == 42


def test_completed_with_zero_chunk_count_ok() -> None:
    """空文档但成功完成的边界场景，chunk_count=0 视为合法。"""
    t = IngestTask(
        **_payload(status=IngestTaskStatus.COMPLETED, chunk_count=0)
    )
    assert t.chunk_count == 0


def test_failed_requires_failure_reason() -> None:
    with pytest.raises(ValidationError) as exc_info:
        IngestTask(**_payload(status=IngestTaskStatus.FAILED))
    assert "failure_reason" in str(exc_info.value)


def test_failed_rejects_blank_failure_reason() -> None:
    with pytest.raises(ValidationError):
        IngestTask(
            **_payload(status=IngestTaskStatus.FAILED, failure_reason="   ")
        )


def test_failed_with_failure_reason_ok() -> None:
    t = IngestTask(
        **_payload(
            status=IngestTaskStatus.FAILED,
            failure_reason="parse timeout",
        )
    )
    assert t.failure_reason == "parse timeout"


def test_negative_chunk_count_rejected() -> None:
    with pytest.raises(ValidationError):
        IngestTask(
            **_payload(status=IngestTaskStatus.COMPLETED, chunk_count=-1)
        )


def test_invalid_status_rejected() -> None:
    with pytest.raises(ValidationError):
        IngestTask(**_payload(status="UNKNOWN"))


def test_missing_file_name_rejected() -> None:
    with pytest.raises(ValidationError):
        IngestTask(file_format="md")  # type: ignore[call-arg]


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        IngestTask(**_payload(unknown="x"))


def test_json_round_trip() -> None:
    original = IngestTask(
        **_payload(status=IngestTaskStatus.COMPLETED, chunk_count=7)
    )
    raw = original.model_dump_json()
    assert '"status":"COMPLETED"' in raw

    restored = IngestTask.model_validate_json(raw)
    assert restored == original

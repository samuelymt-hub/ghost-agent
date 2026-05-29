"""Session / Message / ShortTermMemory / LongTermSummary 单元测试 (Req 18.x)。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ghost_agent.models import (
    LongTermSummary,
    Message,
    Role,
    Session,
    ShortTermMemory,
)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_session_defaults_generate_id_and_timestamp() -> None:
    s = Session()
    assert isinstance(s.session_id, str) and s.session_id
    assert s.created_at.tzinfo is not None  # 带时区


def test_session_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Session(unknown="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


def _msg(
    session_id: str = "s-1",
    role: Role = Role.USER,
    content: str = "hi",
    *,
    created_at: datetime | None = None,
) -> Message:
    return Message(
        session_id=session_id,
        role=role,
        content=content,
        created_at=created_at or datetime.now(timezone.utc),
    )


def test_message_happy_path() -> None:
    m = _msg()
    assert m.session_id == "s-1"
    assert m.role is Role.USER


def test_message_missing_required_field_rejected() -> None:
    with pytest.raises(ValidationError):
        Message(role=Role.USER, content="hi")  # type: ignore[call-arg]


def test_message_empty_content_rejected() -> None:
    with pytest.raises(ValidationError):
        Message(session_id="s-1", role=Role.USER, content="")


def test_message_invalid_role_rejected() -> None:
    with pytest.raises(ValidationError):
        Message(session_id="s-1", role="SYSTEM", content="hi")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------


def test_short_term_memory_happy_path_empty() -> None:
    stm = ShortTermMemory(session_id="s-1")
    assert stm.messages == []


def test_short_term_memory_happy_path_with_messages_in_order() -> None:
    base = datetime.now(timezone.utc)
    msgs = [
        _msg("s-1", Role.USER, "q1", created_at=base),
        _msg("s-1", Role.ASSISTANT, "a1", created_at=base + timedelta(seconds=1)),
        _msg("s-1", Role.USER, "q2", created_at=base + timedelta(seconds=2)),
    ]
    stm = ShortTermMemory(session_id="s-1", messages=msgs)
    assert len(stm.messages) == 3


def test_short_term_memory_rejects_message_from_other_session() -> None:
    base = datetime.now(timezone.utc)
    msgs = [
        _msg("s-1", Role.USER, "q1", created_at=base),
        _msg("s-2", Role.ASSISTANT, "a1", created_at=base + timedelta(seconds=1)),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ShortTermMemory(session_id="s-1", messages=msgs)
    assert "session_id" in str(exc_info.value)


def test_short_term_memory_rejects_out_of_order_messages() -> None:
    base = datetime.now(timezone.utc)
    msgs = [
        _msg("s-1", Role.USER, "q1", created_at=base + timedelta(seconds=2)),
        _msg("s-1", Role.ASSISTANT, "a1", created_at=base),  # 时间倒退
    ]
    with pytest.raises(ValidationError) as exc_info:
        ShortTermMemory(session_id="s-1", messages=msgs)
    assert "created_at" in str(exc_info.value)


def test_short_term_memory_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ShortTermMemory(session_id="s-1", messages=[], unknown="x")  # type: ignore[call-arg]


def test_short_term_memory_json_round_trip() -> None:
    base = datetime.now(timezone.utc)
    msgs = [
        _msg("s-1", Role.USER, "q1", created_at=base),
        _msg("s-1", Role.ASSISTANT, "a1", created_at=base + timedelta(seconds=1)),
    ]
    stm = ShortTermMemory(session_id="s-1", messages=msgs)
    raw = stm.model_dump_json()

    payload = json.loads(raw)
    # role 枚举应序列化为字符串
    assert payload["messages"][0]["role"] == "USER"
    assert payload["messages"][1]["role"] == "ASSISTANT"

    restored = ShortTermMemory.model_validate_json(raw)
    assert restored == stm


# ---------------------------------------------------------------------------
# LongTermSummary
# ---------------------------------------------------------------------------


def test_long_term_summary_happy_path() -> None:
    s = LongTermSummary(
        session_id="s-1",
        summary_text="3 轮对话总结",
        covered_message_ids=["m1", "m2", "m3"],
    )
    assert s.session_id == "s-1"
    assert s.covered_message_ids == ["m1", "m2", "m3"]
    assert s.created_at.tzinfo is not None


def test_long_term_summary_rejects_empty_summary() -> None:
    with pytest.raises(ValidationError):
        LongTermSummary(session_id="s-1", summary_text="")


def test_long_term_summary_missing_session_id_rejected() -> None:
    with pytest.raises(ValidationError):
        LongTermSummary(summary_text="x")  # type: ignore[call-arg]

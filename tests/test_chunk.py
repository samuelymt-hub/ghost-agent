"""Chunk 数据模型单元测试 (Req 6.3, 6.4)。

覆盖：
* happy path 与默认值（chunk_id 自动生成、parent_chunk_id 默认 None）。
* 必填字段缺失被拒绝。
* 非负与字符串非空约束。
* 跨字段约束 ``start_offset <= end_offset``。
* JSON 序列化往返。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ghost_agent.models import Chunk


def _valid_payload(**overrides: object) -> dict:
    base = {
        "source_file_id": "doc-1",
        "seq": 0,
        "start_offset": 0,
        "end_offset": 10,
        "text": "hello world",
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def test_happy_path_with_defaults() -> None:
    c = Chunk(**_valid_payload())
    assert c.source_file_id == "doc-1"
    assert c.seq == 0
    assert c.start_offset == 0
    assert c.end_offset == 10
    assert c.text == "hello world"
    assert c.parent_chunk_id is None
    assert isinstance(c.chunk_id, str) and c.chunk_id  # 自动生成


def test_happy_path_with_parent_chunk_id() -> None:
    c = Chunk(**_valid_payload(parent_chunk_id="parent-uuid"))
    assert c.parent_chunk_id == "parent-uuid"


def test_start_equal_end_is_allowed() -> None:
    """边界：start_offset == end_offset 视为合法（如长度为 1 的边界 chunk）。"""
    c = Chunk(**_valid_payload(start_offset=5, end_offset=5))
    assert c.start_offset == c.end_offset == 5


@pytest.mark.parametrize(
    "missing_field",
    ["source_file_id", "seq", "start_offset", "end_offset", "text"],
)
def test_missing_required_field_rejected(missing_field: str) -> None:
    payload = _valid_payload()
    payload.pop(missing_field)
    with pytest.raises(ValidationError):
        Chunk(**payload)


def test_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        Chunk(**_valid_payload(text=""))


def test_empty_source_file_id_rejected() -> None:
    with pytest.raises(ValidationError):
        Chunk(**_valid_payload(source_file_id=""))


@pytest.mark.parametrize("field", ["seq", "start_offset", "end_offset"])
def test_negative_numeric_fields_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        Chunk(**_valid_payload(**{field: -1}))


def test_start_offset_greater_than_end_offset_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Chunk(**_valid_payload(start_offset=10, end_offset=5))
    assert "start_offset" in str(exc_info.value)


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Chunk(**_valid_payload(unknown_field="x"))


def test_json_round_trip() -> None:
    original = Chunk(**_valid_payload(parent_chunk_id="p1"))
    raw = original.model_dump_json()
    restored = Chunk.model_validate_json(raw)
    assert restored == original

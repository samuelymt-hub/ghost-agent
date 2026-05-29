"""VectorRecord 数据模型单元测试 (Req 21.3)。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ghost_agent.models import VectorRecord, VectorType


def _valid_payload(**overrides: object) -> dict:
    base = {
        "vector": [0.1, 0.2, 0.3],
        "text": "hello",
        "source_id": "doc-1",
        "vector_type": VectorType.DOC_CHUNK,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def test_happy_path_doc_chunk() -> None:
    r = VectorRecord(**_valid_payload())
    assert r.vector == [0.1, 0.2, 0.3]
    assert r.text == "hello"
    assert r.source_id == "doc-1"
    assert r.vector_type is VectorType.DOC_CHUNK
    assert r.metadata == {}
    assert isinstance(r.id, str) and r.id


def test_happy_path_message_with_metadata() -> None:
    r = VectorRecord(
        **_valid_payload(
            vector_type=VectorType.MESSAGE,
            source_id="session-abc",
            metadata={"role": "USER", "ts": 1700000000},
        )
    )
    assert r.vector_type is VectorType.MESSAGE
    assert r.metadata == {"role": "USER", "ts": 1700000000}


@pytest.mark.parametrize(
    "missing_field",
    ["vector", "text", "source_id", "vector_type"],
)
def test_missing_required_field_rejected(missing_field: str) -> None:
    payload = _valid_payload()
    payload.pop(missing_field)
    with pytest.raises(ValidationError):
        VectorRecord(**payload)


def test_empty_vector_rejected() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(**_valid_payload(vector=[]))


def test_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(**_valid_payload(text=""))


def test_empty_source_id_rejected() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(**_valid_payload(source_id=""))


def test_invalid_vector_type_rejected() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(**_valid_payload(vector_type="UNKNOWN"))


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(**_valid_payload(unexpected="x"))


def test_json_round_trip_preserves_enum_as_string() -> None:
    original = VectorRecord(**_valid_payload(metadata={"k": 1}))
    raw = original.model_dump_json()
    payload = json.loads(raw)
    # 枚举应序列化为字符串而非 "VectorType.DOC_CHUNK"
    assert payload["vector_type"] == "DOC_CHUNK"

    restored = VectorRecord.model_validate_json(raw)
    assert restored == original
    assert restored.vector_type is VectorType.DOC_CHUNK

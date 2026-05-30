"""VectorStore 测试（任务 3.2 / 3.3 / 3.4）。

包含：
- 内存版 ``FakeMilvusClient``：复刻 production ``VectorStore`` 调用的底层 client 方法
  （``has_collection`` / ``create_schema`` / ``prepare_index_params`` / ``create_collection``
  / ``insert`` / ``search`` / ``delete`` / ``close``），并以真实 cosine 相似度实现 search、
  支持 ``source_id == '...'`` 与 ``vector_type == '...'`` 的等值过滤、delete 按过滤返回计数。
- 任务 3.3 属性测试 Property 26（Hypothesis, ``@settings(max_examples=100)``）。
- 任务 3.4 单元测试：连接超时返回不可用错误且未写入数据不丢失、delete_by_source 计数、
  search 阈值过滤 / top_k / 降序、source_scope / vector_type 过滤、维度不一致校验。
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ghost_agent.clients import MilvusClientWrapper
from ghost_agent.models.errors import (
    DimensionMismatchError,
    VectorDatabaseUnavailableError,
)
from ghost_agent.models.vector_record import VectorRecord, VectorType
from ghost_agent.vector_db import SearchHit, VectorStore

DIM = 4


# --------------------------------------------------------------------------- #
# 内存版假 Milvus 客户端                                                        #
# --------------------------------------------------------------------------- #
class _FakeSchema:
    """复刻 ``client.create_schema()`` 返回对象：仅记录字段。"""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.fields: list[dict[str, Any]] = []

    def add_field(self, **field: Any) -> None:
        self.fields.append(field)


class _FakeIndexParams:
    """复刻 ``client.prepare_index_params()`` 返回对象：仅记录索引。"""

    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **index: Any) -> None:
        self.indexes.append(index)


def _cosine(a: list[float], b: list[float]) -> float:
    """计算两个向量的 cosine 相似度（与 production 的 COSINE 度量对齐）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# 解析形如 ``field == 'value'`` 的等值子句（值中的 \\ 与 \' 已被 production 转义）。
_EQ_CLAUSE = re.compile(r"(\w+)\s*==\s*'((?:[^'\\]|\\.)*)'")


def _unescape(value: str) -> str:
    """还原 production ``_escape`` 的转义（\\\\ -> \\，\\' -> '）。"""
    return value.replace("\\'", "'").replace("\\\\", "\\")


class FakeMilvusClient:
    """内存版 Milvus 高层 client，复刻 VectorStore 使用的方法子集。

    行内记录以 dict 存储于 :attr:`rows`（按 collection 名分组），供测试断言持久化内容。
    """

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.closed = False

    # --- collection 生命周期 --------------------------------------------- #
    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_schema(self, **kwargs: Any) -> _FakeSchema:
        return _FakeSchema(**kwargs)

    def prepare_index_params(self) -> _FakeIndexParams:
        return _FakeIndexParams()

    def create_collection(
        self,
        *,
        collection_name: str,
        schema: _FakeSchema | None = None,
        index_params: _FakeIndexParams | None = None,
        **_: Any,
    ) -> None:
        self.collections[collection_name] = {
            "schema": schema,
            "index_params": index_params,
        }
        self.rows.setdefault(collection_name, [])

    # --- 写入 ------------------------------------------------------------- #
    def insert(self, *, collection_name: str, data: list[dict[str, Any]]) -> dict[str, int]:
        store = self.rows.setdefault(collection_name, [])
        for row in data:
            store.append(dict(row))
        return {"insert_count": len(data)}

    # --- 检索 ------------------------------------------------------------- #
    def search(
        self,
        *,
        collection_name: str,
        data: list[list[float]],
        limit: int,
        output_fields: list[str],
        filter: str = "",  # noqa: A002 - 对齐 SDK 入参名
        **_: Any,
    ) -> list[list[dict[str, Any]]]:
        query = data[0]
        candidates = self._apply_filter(self.rows.get(collection_name, []), filter)
        scored = [
            {
                "id": row["id"],
                "distance": _cosine(query, row["vector"]),
                "entity": {field: row.get(field) for field in output_fields},
            }
            for row in candidates
        ]
        scored.sort(key=lambda item: item["distance"], reverse=True)
        return [scored[:limit]]

    # --- 删除 ------------------------------------------------------------- #
    def delete(self, *, collection_name: str, filter: str = "") -> dict[str, int]:  # noqa: A002
        store = self.rows.get(collection_name, [])
        matched = self._apply_filter(store, filter)
        matched_ids = {row["id"] for row in matched}
        self.rows[collection_name] = [
            row for row in store if row["id"] not in matched_ids
        ]
        return {"delete_count": len(matched_ids)}

    def close(self) -> None:
        self.closed = True

    # --- 过滤求值 --------------------------------------------------------- #
    @staticmethod
    def _apply_filter(rows: list[dict[str, Any]], filter_expr: str) -> list[dict[str, Any]]:
        if not filter_expr:
            return list(rows)
        conditions = [
            (field, _unescape(value))
            for field, value in _EQ_CLAUSE.findall(filter_expr)
        ]
        result = []
        for row in rows:
            if all(str(row.get(field)) == value for field, value in conditions):
                result.append(row)
        return result


# --------------------------------------------------------------------------- #
# fixtures / 工厂                                                               #
# --------------------------------------------------------------------------- #
def _make_store(monkeypatch, *, dim: int = DIM) -> tuple[VectorStore, FakeMilvusClient]:
    """构造注入内存假客户端的 VectorStore，返回 (store, fake_client)。"""
    wrapper = MilvusClientWrapper(uri="http://fake:19530")
    fake = FakeMilvusClient()
    monkeypatch.setattr(wrapper, "_build_client", lambda: fake)
    store = VectorStore(milvus=wrapper, dim=dim, collection_name="test_vectors")
    return store, fake


def _record(
    *,
    vector: list[float],
    text: str = "hello",
    source_id: str = "src-1",
    vector_type: VectorType = VectorType.DOC_CHUNK,
    metadata: dict[str, Any] | None = None,
) -> VectorRecord:
    return VectorRecord(
        vector=vector,
        text=text,
        source_id=source_id,
        vector_type=vector_type,
        metadata=metadata or {},
    )


# =========================================================================== #
# 任务 3.3 — 属性测试 Property 26                                               #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 26: 对任意向量写入请求，当向量维度等于
# Embedding_Model 输出维度时写入成功且持久化记录同时包含原始文本、来源标识与向量类型；
# 当向量维度不等于该维度时拒绝写入并返回维度不一致错误。
# Validates: Requirements 21.3, 21.4


_finite_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_safe_text = st.text(min_size=1, max_size=50)
_metadata = st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=st.one_of(st.integers(), st.text(max_size=20), st.booleans()),
    max_size=4,
)
_vector_types = st.sampled_from(list(VectorType))


@settings(
    max_examples=100,
    deadline=None,  # 首个样本会惰性 import pymilvus.DataType，耗时不稳定，关闭 per-example deadline
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    matching_dim=st.booleans(),
    wrong_dim=st.integers(min_value=1, max_value=16).filter(lambda d: d != DIM),
    vector_values=st.lists(_finite_floats, min_size=1, max_size=16),
    text=_safe_text,
    source_id=_safe_text,
    vector_type=_vector_types,
    metadata=_metadata,
)
def test_property_26_write_dimension_check_and_metadata(
    monkeypatch,
    matching_dim: bool,
    wrong_dim: int,
    vector_values: list[float],
    text: str,
    source_id: str,
    vector_type: VectorType,
    metadata: dict[str, Any],
):
    """Property 26：维度匹配则写入成功并持久化文本/来源/类型；不匹配则拒绝并报错。"""
    store, fake = _make_store(monkeypatch)

    if matching_dim:
        # 维度等于配置 dim：写入成功，持久化记录必含原始文本、来源标识与向量类型。
        vector = [vector_values[i % len(vector_values)] for i in range(DIM)]
        record = _record(
            vector=vector,
            text=text,
            source_id=source_id,
            vector_type=vector_type,
            metadata=metadata,
        )
        store.write(record)

        persisted = fake.rows[store.collection_name]
        assert len(persisted) == 1
        row = persisted[0]
        assert row["text"] == text
        assert row["source_id"] == source_id
        assert row["vector_type"] == vector_type.value
        assert row["metadata"] == metadata
        assert len(row["vector"]) == DIM
    else:
        # 维度不等于配置 dim：拒绝写入并返回维度不一致错误，且不持久化任何记录。
        vector = [vector_values[i % len(vector_values)] for i in range(wrong_dim)]
        record = _record(
            vector=vector,
            text=text,
            source_id=source_id,
            vector_type=vector_type,
            metadata=metadata,
        )
        with pytest.raises(DimensionMismatchError) as exc_info:
            store.write(record)
        assert exc_info.value.details == {"expected_dim": DIM, "actual_dim": wrong_dim}
        # 未写入数据不丢失：底层不应残留任何记录。
        assert fake.rows.get(store.collection_name, []) == []


# =========================================================================== #
# 任务 3.4 — 单元测试                                                            #
# =========================================================================== #

# --- 写入与维度校验 --------------------------------------------------------- #
def test_write_persists_text_source_and_type(monkeypatch):
    store, fake = _make_store(monkeypatch)
    record = _record(
        vector=[0.1, 0.2, 0.3, 0.4],
        text="接入手册第一节",
        source_id="doc-42",
        vector_type=VectorType.DOC_CHUNK,
        metadata={"seq": 1},
    )
    store.write(record)

    rows = fake.rows[store.collection_name]
    assert len(rows) == 1
    assert rows[0]["text"] == "接入手册第一节"
    assert rows[0]["source_id"] == "doc-42"
    assert rows[0]["vector_type"] == "DOC_CHUNK"
    assert rows[0]["metadata"] == {"seq": 1}


def test_write_dimension_mismatch_rejected(monkeypatch):
    store, fake = _make_store(monkeypatch)
    record = _record(vector=[0.1, 0.2, 0.3])  # dim=3 != 4
    with pytest.raises(DimensionMismatchError) as exc_info:
        store.write(record)
    assert exc_info.value.details == {"expected_dim": 4, "actual_dim": 3}
    # 维度校验发生在任何 DB 调用之前：collection 都不应被创建。
    assert fake.collections == {}
    assert fake.rows == {}


def test_search_query_vector_dimension_mismatch_rejected(monkeypatch):
    store, _ = _make_store(monkeypatch)
    with pytest.raises(DimensionMismatchError) as exc_info:
        store.search([1.0, 2.0], top_k=5, min_score=0.0)  # dim=2 != 4
    assert exc_info.value.details == {"expected_dim": 4, "actual_dim": 2}


# --- 连接超时 / 不可用且未写入数据不丢失 ------------------------------------ #
def _failing_store() -> tuple[VectorStore, dict[str, int]]:
    """构造一个底层连接始终失败的 VectorStore。"""
    wrapper = MilvusClientWrapper(uri="http://unreachable:19530", timeout=2.0)
    calls = {"build": 0}

    def _boom():
        calls["build"] += 1
        raise TimeoutError("connection timed out")

    wrapper._build_client = _boom  # type: ignore[method-assign]
    store = VectorStore(milvus=wrapper, dim=DIM, collection_name="test_vectors")
    return store, calls


def test_write_connection_failure_raises_unavailable_and_preserves_data():
    store, _ = _failing_store()
    record = _record(vector=[0.1, 0.2, 0.3, 0.4])
    with pytest.raises(VectorDatabaseUnavailableError) as exc_info:
        store.write(record)
    # 维度校验已通过，失败来自连接阶段；原始异常被保留为 __cause__。
    assert isinstance(exc_info.value.__cause__, VectorDatabaseUnavailableError | TimeoutError)
    # 未写入数据不丢失：传入的 record 对象本身不被破坏，可再次尝试写入。
    assert record.vector == [0.1, 0.2, 0.3, 0.4]
    assert record.text == "hello"


def test_search_connection_failure_raises_unavailable():
    store, _ = _failing_store()
    with pytest.raises(VectorDatabaseUnavailableError):
        store.search([0.1, 0.2, 0.3, 0.4], top_k=5, min_score=0.0)


def test_delete_connection_failure_raises_unavailable():
    store, _ = _failing_store()
    with pytest.raises(VectorDatabaseUnavailableError):
        store.delete_by_source("doc-1")


def test_write_failure_does_not_persist_partial_data(monkeypatch):
    """insert 抛错时应封装为不可用错误，且内存中不残留该记录。"""
    store, fake = _make_store(monkeypatch)

    def _boom_insert(*, collection_name: str, data: list[dict[str, Any]]):
        raise RuntimeError("insert failed mid-write")

    monkeypatch.setattr(fake, "insert", _boom_insert)
    record = _record(vector=[0.1, 0.2, 0.3, 0.4])
    with pytest.raises(VectorDatabaseUnavailableError):
        store.write(record)
    assert fake.rows.get(store.collection_name, []) == []


# --- delete_by_source 计数 -------------------------------------------------- #
def test_delete_by_source_counts_and_only_removes_matching(monkeypatch):
    store, fake = _make_store(monkeypatch)
    store.write(_record(vector=[1.0, 0.0, 0.0, 0.0], source_id="A", text="a1"))
    store.write(_record(vector=[0.0, 1.0, 0.0, 0.0], source_id="A", text="a2"))
    store.write(_record(vector=[0.0, 0.0, 1.0, 0.0], source_id="B", text="b1"))

    deleted = store.delete_by_source("A")
    assert deleted == 2

    remaining = fake.rows[store.collection_name]
    assert len(remaining) == 1
    assert remaining[0]["source_id"] == "B"


def test_delete_by_source_absent_returns_zero(monkeypatch):
    store, _ = _make_store(monkeypatch)
    store.write(_record(vector=[1.0, 0.0, 0.0, 0.0], source_id="A"))
    assert store.delete_by_source("does-not-exist") == 0


# --- search 阈值 / top_k / 降序 / 过滤 -------------------------------------- #
def test_search_applies_min_score_threshold(monkeypatch):
    store, _ = _make_store(monkeypatch)
    # 与查询 [1,0,0,0] 完全对齐 -> cosine=1.0；正交 -> cosine=0.0。
    store.write(_record(vector=[1.0, 0.0, 0.0, 0.0], text="aligned", source_id="s"))
    store.write(_record(vector=[0.0, 1.0, 0.0, 0.0], text="orthogonal", source_id="s"))

    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=5, min_score=0.5)
    assert [h.text for h in hits] == ["aligned"]
    assert all(h.score >= 0.5 for h in hits)


def test_search_respects_top_k_and_descending_order(monkeypatch):
    store, _ = _make_store(monkeypatch)
    # 三个与查询夹角递增的向量：分数 a > b > c。
    store.write(_record(vector=[1.0, 0.0, 0.0, 0.0], text="a", source_id="s"))
    store.write(_record(vector=[1.0, 0.5, 0.0, 0.0], text="b", source_id="s"))
    store.write(_record(vector=[1.0, 2.0, 0.0, 0.0], text="c", source_id="s"))

    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=2, min_score=0.0)
    assert len(hits) == 2  # 受 top_k 限制
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)  # 降序
    assert [h.text for h in hits] == ["a", "b"]


def test_search_source_scope_filter(monkeypatch):
    store, _ = _make_store(monkeypatch)
    store.write(_record(vector=[1.0, 0.0, 0.0, 0.0], text="a", source_id="S1"))
    store.write(_record(vector=[1.0, 0.0, 0.0, 0.0], text="b", source_id="S2"))

    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=5, min_score=0.0, source_scope="S2")
    assert [h.source_id for h in hits] == ["S2"]
    assert [h.text for h in hits] == ["b"]


def test_search_vector_type_filter(monkeypatch):
    store, _ = _make_store(monkeypatch)
    store.write(
        _record(vector=[1.0, 0.0, 0.0, 0.0], text="doc", vector_type=VectorType.DOC_CHUNK, source_id="s")
    )
    store.write(
        _record(vector=[1.0, 0.0, 0.0, 0.0], text="msg", vector_type=VectorType.MESSAGE, source_id="s")
    )

    hits = store.search(
        [1.0, 0.0, 0.0, 0.0],
        top_k=5,
        min_score=0.0,
        vector_type=VectorType.MESSAGE,
    )
    assert [h.text for h in hits] == ["msg"]
    assert all(h.vector_type == VectorType.MESSAGE for h in hits)


def test_search_combined_source_and_type_filter(monkeypatch):
    store, _ = _make_store(monkeypatch)
    store.write(
        _record(vector=[1.0, 0.0, 0.0, 0.0], text="hit", vector_type=VectorType.MESSAGE, source_id="sess-1")
    )
    store.write(
        _record(vector=[1.0, 0.0, 0.0, 0.0], text="other-session", vector_type=VectorType.MESSAGE, source_id="sess-2")
    )
    store.write(
        _record(vector=[1.0, 0.0, 0.0, 0.0], text="other-type", vector_type=VectorType.DOC_CHUNK, source_id="sess-1")
    )

    hits = store.search(
        [1.0, 0.0, 0.0, 0.0],
        top_k=5,
        min_score=0.0,
        source_scope="sess-1",
        vector_type=VectorType.MESSAGE,
    )
    assert [h.text for h in hits] == ["hit"]


def test_search_empty_collection_returns_empty(monkeypatch):
    store, _ = _make_store(monkeypatch)
    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=5, min_score=0.0)
    assert hits == []


def test_search_hit_is_searchhit_instance(monkeypatch):
    store, _ = _make_store(monkeypatch)
    store.write(_record(vector=[1.0, 0.0, 0.0, 0.0], text="x", source_id="s"))
    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=1, min_score=0.0)
    assert isinstance(hits[0], SearchHit)


# --- ensure_collection 惰性建表 -------------------------------------------- #
def test_ensure_collection_created_once(monkeypatch):
    store, fake = _make_store(monkeypatch)
    create_calls = {"count": 0}
    original_create = fake.create_collection

    def _counting_create(**kwargs: Any):
        create_calls["count"] += 1
        return original_create(**kwargs)

    monkeypatch.setattr(fake, "create_collection", _counting_create)

    store.write(_record(vector=[1.0, 0.0, 0.0, 0.0]))
    store.write(_record(vector=[0.0, 1.0, 0.0, 0.0]))
    store.delete_by_source("src-1")
    store.search([1.0, 0.0, 0.0, 0.0], top_k=1, min_score=0.0)

    assert create_calls["count"] == 1  # 仅建表一次
    assert store.collection_name in fake.collections

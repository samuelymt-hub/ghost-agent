"""Indexer 测试（任务 5.1 / 5.2 / 5.3 / 5.4）。

包含：
- 内存替身 ``FakeEmbeddingClient``：返回固定维度的确定性向量，可按文本/调用次数/全局
  策略注入嵌入失败，并统计调用次数（用于验证重试上界 Property 5）。
- 内存替身 ``FakeVectorStore``：把写入记录追加到列表，可按文本注入写入失败，并复刻真实
  VectorStore 的维度校验（用于验证维度不一致计入失败而非崩溃）。
- 任务 5.2 属性测试 Property 4（索引计数守恒，Hypothesis, ``max_examples=100``）。
- 任务 5.3 属性测试 Property 5（嵌入重试次数上界，Hypothesis, ``max_examples=100``）。
- 任务 5.4 单元测试：消息向量写入失败仅记录不中断（Req 19.2）、单 Chunk 写入失败隔离
  （Req 7.4）、空集合不调用 Embedding（Req 7.6）、维度不一致计入失败、消息向量成功路径。
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.core.indexer import IndexFailure, Indexer, IndexResult
from ghost_agent.models.chunk import Chunk
from ghost_agent.models.errors import (
    DimensionMismatchError,
    QueryEmbeddingFailedError,
    VectorDatabaseUnavailableError,
)
from ghost_agent.models.vector_record import VectorRecord, VectorType

DIM = 4


# --------------------------------------------------------------------------- #
# 内存替身                                                                      #
# --------------------------------------------------------------------------- #
class FakeEmbeddingClient:
    """内存版 Embedding 客户端。

    返回长度为 ``dim`` 的确定性向量；支持多种失败注入策略以驱动属性/单元测试：

    * ``always_fail``    —— 任何调用都抛 :class:`QueryEmbeddingFailedError`。
    * ``fail_first_n``   —— 前 ``n`` 次调用失败、之后成功（全局计数，适合单 Chunk 重试场景）。
    * ``fail_texts``     —— 命中集合内文本的调用始终失败（适合多 Chunk 成功/失败分布）。
    * ``wrong_dim_for``  —— 命中集合内文本时返回维度 ``dim+1`` 的向量（触发下游维度校验失败）。

    :attr:`calls` 统计 ``embed`` 被调用的总次数（每次 ``embed([text])`` 计 1）。
    """

    def __init__(
        self,
        *,
        dim: int = DIM,
        always_fail: bool = False,
        fail_first_n: int = 0,
        fail_texts: set[str] | None = None,
        wrong_dim_for: set[str] | None = None,
    ) -> None:
        self._dim = dim
        self._always_fail = always_fail
        self._fail_first_n = fail_first_n
        self._fail_texts = set(fail_texts or [])
        self._wrong_dim_for = set(wrong_dim_for or [])
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self._always_fail:
            raise QueryEmbeddingFailedError("always fail")
        text = texts[0]
        if text in self._fail_texts:
            raise QueryEmbeddingFailedError("configured text failure")
        if self._fail_first_n > 0:
            self._fail_first_n -= 1
            raise QueryEmbeddingFailedError("transient failure")
        return [self._vector_for(t) for t in texts]

    def _vector_for(self, text: str) -> list[float]:
        length = self._dim + (1 if text in self._wrong_dim_for else 0)
        # 确定性、非零向量（具体值不重要，长度才是关键）。
        return [float((len(text) + i) % 7) + 0.5 for i in range(length)]


class SequencedEmbeddingClient:
    """按预设结果序列响应的 Embedding 替身：用于覆盖"任意嵌入失败序列"。

    第 ``i`` 次（0-based）``embed`` 调用：当 ``i < len(outcomes)`` 且 ``outcomes[i]``
    为 ``True`` 时成功返回向量，否则抛 :class:`QueryEmbeddingFailedError`。
    :attr:`calls` 统计总调用次数。
    """

    def __init__(self, outcomes: list[bool], *, dim: int = DIM) -> None:
        self._outcomes = list(outcomes)
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        idx = self.calls
        self.calls += 1
        if idx < len(self._outcomes) and self._outcomes[idx]:
            return [[0.5] * self._dim for _ in texts]
        raise QueryEmbeddingFailedError("sequenced failure")


class FakeVectorStore:
    """内存版 vector_store：记录写入并可按文本注入失败。

    复刻真实 :class:`VectorStore` 的两类失败：维度不一致抛
    :class:`DimensionMismatchError`（Req 21.4），写入故障抛
    :class:`VectorDatabaseUnavailableError`（Req 21.5）。
    """

    def __init__(
        self,
        *,
        dim: int = DIM,
        fail_texts: set[str] | None = None,
        check_dim: bool = True,
    ) -> None:
        self._dim = dim
        self._fail_texts = set(fail_texts or [])
        self._check_dim = check_dim
        self.written: list[VectorRecord] = []

    @property
    def dim(self) -> int:
        return self._dim

    def write(self, record: VectorRecord) -> None:
        if self._check_dim and len(record.vector) != self._dim:
            raise DimensionMismatchError(
                f"向量维度({len(record.vector)})与配置维度({self._dim})不一致",
                details={"expected_dim": self._dim, "actual_dim": len(record.vector)},
            )
        if record.text in self._fail_texts:
            raise VectorDatabaseUnavailableError("configured write failure")
        self.written.append(record)


# --------------------------------------------------------------------------- #
# 工厂                                                                          #
# --------------------------------------------------------------------------- #
def _make_chunk(*, text: str, seq: int, source_file_id: str = "src-1") -> Chunk:
    return Chunk(
        source_file_id=source_file_id,
        seq=seq,
        start_offset=seq,
        end_offset=seq + len(text),
        text=text,
    )


# =========================================================================== #
# 任务 5.2 — 属性测试 Property 4                                                #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 4: 对任意 Chunk 集合（含空集合）及任意
# 成功/失败分布，Indexer 返回的成功写入数量与失败数量之和恒等于接收到的 Chunk 总数。
# Validates: Requirements 7.5, 7.6, 22.4


@settings(max_examples=100, deadline=None)
@given(
    labels=st.lists(
        st.sampled_from(["ok", "embed_fail", "write_fail"]),
        min_size=0,
        max_size=30,
    )
)
def test_property_4_index_count_conservation(labels: list[str]):
    """Property 4：success_count + failure_count 恒等于接收到的 Chunk 总数（含空集合）。"""
    chunks: list[Chunk] = []
    embed_fail_texts: set[str] = set()
    write_fail_texts: set[str] = set()
    for i, label in enumerate(labels):
        text = f"chunk-{i}-content"  # 索引保证文本唯一
        chunks.append(_make_chunk(text=text, seq=i))
        if label == "embed_fail":
            embed_fail_texts.add(text)
        elif label == "write_fail":
            write_fail_texts.add(text)

    embed = FakeEmbeddingClient(dim=DIM, fail_texts=embed_fail_texts)
    store = FakeVectorStore(dim=DIM, fail_texts=write_fail_texts)
    indexer = Indexer(embedding_client=embed, vector_store=store, max_retries=0)

    result = indexer.index(chunks)

    # 计数守恒（Req 7.5）：成功 + 失败 == Chunk 总数。
    assert result.success_count + result.failure_count == len(labels)
    # 失败明细数量与失败计数一致。
    assert len(result.failures) == result.failure_count
    # 成功数恰为标记为 ok 的数量。
    expected_success = sum(1 for label in labels if label == "ok")
    assert result.success_count == expected_success
    # 成功写入的记录数与成功计数一致。
    assert len(store.written) == result.success_count
    # Req 7.6：空集合不调用 Embedding_Model。
    if not labels:
        assert embed.calls == 0
        assert result.success_count == 0
        assert result.failure_count == 0


# =========================================================================== #
# 任务 5.3 — 属性测试 Property 5                                                #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 5: 对任意嵌入失败序列，Indexer 对单个
# Chunk 的累计重试次数不超过配置的最大重试次数（取值范围 0–5，默认 3）。
# Validates: Requirements 7.3


@settings(max_examples=100, deadline=None)
@given(
    max_retries=st.integers(min_value=0, max_value=5),
    k=st.integers(min_value=0, max_value=7),
    outcomes=st.lists(st.booleans(), min_size=0, max_size=12),
)
def test_property_5_embedding_retry_upper_bound(
    max_retries: int, k: int, outcomes: list[bool]
):
    """Property 5：单 Chunk 的总嵌入调用次数不超过 max_retries + 1（即重试 <= max_retries）。"""
    # 情形 A：嵌入始终失败 —— 总调用次数恰为 max_retries + 1，Chunk 计为失败。
    always_fail = FakeEmbeddingClient(dim=DIM, always_fail=True)
    indexer_a = Indexer(
        embedding_client=always_fail,
        vector_store=FakeVectorStore(dim=DIM),
        max_retries=max_retries,
    )
    result_a = indexer_a.index([_make_chunk(text="retry-target", seq=0)])
    assert always_fail.calls == max_retries + 1  # 重试次数 == max_retries
    assert result_a.failure_count == 1
    assert result_a.success_count == 0

    # 情形 B：失败 k 次后成功。k <= max_retries 时成功且调用 k+1 次；否则失败且调用 max_retries+1 次。
    flaky = FakeEmbeddingClient(dim=DIM, fail_first_n=k)
    indexer_b = Indexer(
        embedding_client=flaky,
        vector_store=FakeVectorStore(dim=DIM),
        max_retries=max_retries,
    )
    result_b = indexer_b.index([_make_chunk(text="flaky-target", seq=0)])
    if k <= max_retries:
        assert result_b.success_count == 1
        assert flaky.calls == k + 1
    else:
        assert result_b.failure_count == 1
        assert flaky.calls == max_retries + 1
    # 无论如何，调用次数都不超过上界（重试 <= max_retries）。
    assert flaky.calls <= max_retries + 1

    # 情形 C：任意成功/失败序列。无论序列形态如何，单 Chunk 的累计嵌入调用次数
    # （= 1 次首次尝试 + 重试次数）都不超过 max_retries + 1（Req 7.3）。
    sequenced = SequencedEmbeddingClient(outcomes, dim=DIM)
    indexer_c = Indexer(
        embedding_client=sequenced,
        vector_store=FakeVectorStore(dim=DIM),
        max_retries=max_retries,
    )
    indexer_c.index([_make_chunk(text="seq-target", seq=0)])
    assert sequenced.calls <= max_retries + 1
    # 首个成功落在允许尝试窗口内时应成功（调用次数恰为该成功位置 + 1）。
    first_success = next(
        (i for i, ok in enumerate(outcomes) if ok), None
    )
    if first_success is not None and first_success <= max_retries:
        assert sequenced.calls == first_success + 1
    else:
        assert sequenced.calls == max_retries + 1


# =========================================================================== #
# 任务 5.4 — 单元测试                                                            #
# =========================================================================== #

# --- 空集合（Req 7.6） ----------------------------------------------------- #
def test_index_empty_returns_zero_and_does_not_call_embedding():
    embed = FakeEmbeddingClient(dim=DIM)
    store = FakeVectorStore(dim=DIM)
    indexer = Indexer(embedding_client=embed, vector_store=store)

    result = indexer.index([])

    assert isinstance(result, IndexResult)
    assert result.success_count == 0
    assert result.failure_count == 0
    assert result.failures == []
    assert embed.calls == 0  # 不调用 Embedding_Model
    assert store.written == []


# --- 单 Chunk 写入失败隔离（Req 7.4） -------------------------------------- #
def test_single_chunk_write_failure_isolated_others_succeed():
    chunks = [
        _make_chunk(text="first", seq=0),
        _make_chunk(text="middle", seq=1),
        _make_chunk(text="last", seq=2),
    ]
    middle_id = chunks[1].chunk_id

    embed = FakeEmbeddingClient(dim=DIM)
    store = FakeVectorStore(dim=DIM, fail_texts={"middle"})
    indexer = Indexer(embedding_client=embed, vector_store=store, max_retries=0)

    result = indexer.index(chunks)

    assert result.success_count == 2
    assert result.failure_count == 1
    assert len(result.failures) == 1
    assert result.failures[0].chunk_id == middle_id
    assert isinstance(result.failures[0], IndexFailure)
    # 仅成功的两个 Chunk 被写入。
    assert {r.text for r in store.written} == {"first", "last"}


# --- 单 Chunk 嵌入耗尽重试计为失败，其余成功（Req 7.3 / 7.4） -------------- #
def test_single_chunk_embedding_exhausts_retries_counted_as_failure():
    chunks = [
        _make_chunk(text="ok-1", seq=0),
        _make_chunk(text="bad", seq=1),
        _make_chunk(text="ok-2", seq=2),
    ]
    bad_id = chunks[1].chunk_id

    embed = FakeEmbeddingClient(dim=DIM, fail_texts={"bad"})
    store = FakeVectorStore(dim=DIM)
    indexer = Indexer(embedding_client=embed, vector_store=store, max_retries=2)

    result = indexer.index(chunks)

    assert result.success_count == 2
    assert result.failure_count == 1
    assert result.failures[0].chunk_id == bad_id
    assert {r.text for r in store.written} == {"ok-1", "ok-2"}


# --- 维度不一致计入失败而非崩溃（Req 7.4 + 21.4） -------------------------- #
def test_dimension_mismatch_from_write_counted_as_failure():
    chunks = [
        _make_chunk(text="good", seq=0),
        _make_chunk(text="wrong-dim", seq=1),
    ]
    wrong_id = chunks[1].chunk_id

    embed = FakeEmbeddingClient(dim=DIM, wrong_dim_for={"wrong-dim"})
    store = FakeVectorStore(dim=DIM, check_dim=True)
    indexer = Indexer(embedding_client=embed, vector_store=store, max_retries=0)

    result = indexer.index(chunks)

    assert result.success_count == 1
    assert result.failure_count == 1
    assert result.failures[0].chunk_id == wrong_id
    assert [r.text for r in store.written] == ["good"]


# --- 成功写入的 DOC_CHUNK 元数据 ------------------------------------------- #
def test_index_writes_doc_chunk_records_with_metadata():
    chunk = _make_chunk(text="hello world", seq=3, source_file_id="doc-7")
    embed = FakeEmbeddingClient(dim=DIM)
    store = FakeVectorStore(dim=DIM)
    indexer = Indexer(embedding_client=embed, vector_store=store, max_retries=0)

    result = indexer.index([chunk])

    assert result.success_count == 1
    assert len(store.written) == 1
    record = store.written[0]
    assert record.vector_type == VectorType.DOC_CHUNK
    assert record.source_id == "doc-7"
    assert record.text == "hello world"
    assert record.metadata["seq"] == 3
    assert record.metadata["chunk_id"] == chunk.chunk_id
    assert record.metadata["start_offset"] == chunk.start_offset
    assert record.metadata["end_offset"] == chunk.end_offset


# --- index_message 成功路径（Req 19.1） ------------------------------------ #
def test_index_message_writes_two_message_records():
    embed = FakeEmbeddingClient(dim=DIM)
    store = FakeVectorStore(dim=DIM)
    indexer = Indexer(embedding_client=embed, vector_store=store, max_retries=0)

    indexer.index_message("sess-42", "用户问题", "助手应答")

    assert len(store.written) == 2
    assert all(r.vector_type == VectorType.MESSAGE for r in store.written)
    assert all(r.source_id == "sess-42" for r in store.written)

    user_rec, asst_rec = store.written
    assert user_rec.text == "用户问题"
    assert user_rec.metadata["role"] == "USER"
    assert user_rec.metadata["session_id"] == "sess-42"
    assert asst_rec.text == "助手应答"
    assert asst_rec.metadata["role"] == "ASSISTANT"


# --- index_message 嵌入失败仅记录不中断（Req 19.2） ------------------------ #
def test_index_message_embed_failure_does_not_raise():
    embed = FakeEmbeddingClient(dim=DIM, always_fail=True)
    store = FakeVectorStore(dim=DIM)
    indexer = Indexer(embedding_client=embed, vector_store=store, max_retries=1)

    # 不应抛出任何异常；返回 None。
    assert indexer.index_message("sess-1", "u", "a") is None
    # 嵌入全部失败 -> 没有任何记录写入。
    assert store.written == []


# --- index_message 写入失败仅记录不中断（Req 19.2） ------------------------ #
def test_index_message_write_failure_does_not_raise():
    embed = FakeEmbeddingClient(dim=DIM)
    store = FakeVectorStore(dim=DIM, fail_texts={"u", "a"})
    indexer = Indexer(embedding_client=embed, vector_store=store, max_retries=0)

    assert indexer.index_message("sess-1", "u", "a") is None
    assert store.written == []


# --- index_message 部分失败：用户消息失败不影响应答写入（Req 19.2） -------- #
def test_index_message_partial_failure_still_writes_other():
    embed = FakeEmbeddingClient(dim=DIM)
    store = FakeVectorStore(dim=DIM, fail_texts={"u"})
    indexer = Indexer(embedding_client=embed, vector_store=store, max_retries=0)

    assert indexer.index_message("sess-1", "u", "a") is None
    # 用户消息写入失败但应答仍被写入。
    assert len(store.written) == 1
    assert store.written[0].text == "a"
    assert store.written[0].metadata["role"] == "ASSISTANT"

"""Retriever 测试（任务 6.1 / 6.2 / 6.3 / 6.4 / 6.5）。

包含：
- 内存替身 ``FakeEmbeddingClient``：返回确定性查询向量，可注入嵌入失败。
- 内存替身 ``FakeVectorStore``：以预置 (SearchHit) 列表为底，复刻真实 VectorStore.search
  的语义（min_score 过滤 + source_scope/vector_type 过滤 + 分数降序），并**返回全部匹配
  命中、忽略 limit**——把 Top-K 截断/边界并列逻辑完全交给 Retriever 验证。
- 任务 6.2 属性测试 Property 6（检索 Top-K 与阈值不变量，Hypothesis, max_examples=100）。
- 任务 6.3 属性测试 Property 7（混合检索去重、降序与数量上界，max_examples=100）。
- 任务 6.4 属性测试 Property 8（重排是降序排列且为输入的排列，max_examples=100）。
- 任务 6.5 属性测试 Property 9（消息向量召回的 Session 范围与数量上界，max_examples=100）。
- 单元测试：空/空白查询、嵌入失败、无命中空集、消息无历史空集、hybrid=False 忽略关键词、
  默认 vector_type=DOC_CHUNK、retrieve_messages 强制 MESSAGE + session 范围、边界并列全部返回。
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.core.retriever import (
    RetrieveOptions,
    Retriever,
    default_reranker,
    rerank_relevance,
)
from ghost_agent.models.errors import QueryEmbeddingFailedError, QueryEmptyError
from ghost_agent.models.vector_record import VectorType
from ghost_agent.vector_db.vector_store import SearchHit

DIM = 4


# --------------------------------------------------------------------------- #
# 内存替身                                                                      #
# --------------------------------------------------------------------------- #
class FakeEmbeddingClient:
    """内存版 Embedding 客户端：返回确定性查询向量，可注入失败。

    * ``always_fail`` —— 任何 ``embed`` 调用抛 :class:`QueryEmbeddingFailedError`。
    * ``return_empty`` —— ``embed`` 返回空列表（驱动"Embedding 返回空结果"分支）。
    :attr:`calls` 统计调用次数（Retriever 应在校验空查询之后才调用 embed）。
    """

    def __init__(
        self,
        *,
        dim: int = DIM,
        always_fail: bool = False,
        return_empty: bool = False,
    ) -> None:
        self._dim = dim
        self._always_fail = always_fail
        self._return_empty = return_empty
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self._always_fail:
            raise QueryEmbeddingFailedError("configured embed failure")
        if self._return_empty:
            return []
        # 查询向量值对检索不重要（FakeVectorStore 用预置分数），仅需维度正确。
        return [[float(len(t) % 7) + 0.1 for _ in range(self._dim)] for t in texts]


class FakeVectorStore:
    """内存版 vector_store：以预置命中列表为底复刻 search 语义（忽略 limit）。

    返回**全部**满足 ``score >= min_score`` 且匹配 ``source_scope`` / ``vector_type``
    的命中并按分数降序排列；**不按 limit 截断**，从而把 Top-K 截断与边界并列逻辑
    完全交由 Retriever 验证（Property 6）。
    """

    def __init__(self, hits: list[SearchHit], *, dim: int = DIM) -> None:
        self._hits = list(hits)
        self._dim = dim
        self.last_limit: int | None = None
        self.last_source_scope: str | None = None
        self.last_vector_type: VectorType | None = None

    @property
    def dim(self) -> int:
        return self._dim

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        min_score: float,
        source_scope: str | None = None,
        vector_type: VectorType | None = None,
    ) -> list[SearchHit]:
        self.last_limit = top_k
        self.last_source_scope = source_scope
        self.last_vector_type = vector_type
        matched = [
            hit
            for hit in self._hits
            if hit.score >= min_score
            and (source_scope is None or hit.source_id == source_scope)
            and (vector_type is None or hit.vector_type == vector_type)
        ]
        matched.sort(key=lambda hit: hit.score, reverse=True)
        return matched  # 忽略 top_k，由 Retriever 截断


# --------------------------------------------------------------------------- #
# 工厂                                                                          #
# --------------------------------------------------------------------------- #
def _hit(
    *,
    id: str,  # noqa: A002 - 对齐 SearchHit 字段名
    score: float,
    text: str = "chunk text",
    source_id: str = "src-1",
    vector_type: VectorType = VectorType.DOC_CHUNK,
) -> SearchHit:
    return SearchHit(
        id=id,
        text=text,
        source_id=source_id,
        vector_type=vector_type,
        score=score,
        metadata={},
    )


def _make_retriever(
    hits: list[SearchHit],
    *,
    keyword_search=None,
    reranker=None,
) -> tuple[Retriever, FakeEmbeddingClient, FakeVectorStore]:
    embed = FakeEmbeddingClient(dim=DIM)
    store = FakeVectorStore(hits, dim=DIM)
    retriever = Retriever(
        embedding_client=embed,
        vector_store=store,
        keyword_search=keyword_search,
        reranker=reranker,
    )
    return retriever, embed, store


# =========================================================================== #
# 任务 6.2 — 属性测试 Property 6                                                #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 6: 对任意向量库与非空查询，Retriever 返回结果的相似度分数均不低于配置的最小相似度阈值，结果按相似度降序排列，且在不存在边界并列的情况下返回数量不超过 Top-K（取值范围 1–100，默认 5）。
# Validates: Requirements 8.2


@settings(max_examples=200, deadline=None)
@given(
    scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=40,
        unique=True,  # 分数互异 -> 不存在边界并列
    ),
    top_k=st.integers(min_value=1, max_value=100),
    min_score=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
def test_property_6_topk_and_threshold_invariant(
    scores: list[float], top_k: int, min_score: float
):
    """Property 6：分数 >= min_score、降序、且无边界并列时数量 <= Top-K。"""
    hits = [_hit(id=f"c-{i}", score=score) for i, score in enumerate(scores)]
    retriever, _, _ = _make_retriever(hits)

    result = retriever.retrieve(
        "非空查询", RetrieveOptions(top_k=top_k, min_score=min_score)
    )

    # 全部命中分数 >= min_score。
    assert all(hit.score >= min_score for hit in result)
    # 按相似度降序排列。
    result_scores = [hit.score for hit in result]
    assert result_scores == sorted(result_scores, reverse=True)
    # 分数互异 => 不存在边界并列 => 数量 <= Top-K。
    assert len(result) <= top_k
    # 结果恰为"满足阈值命中按分数降序后取前 top_k"。
    eligible = sorted((s for s in scores if s >= min_score), reverse=True)
    assert result_scores == eligible[:top_k]


def test_property_6_boundary_ties_all_returned():
    """Req 8.3：Top-K 边界处分数并列时全部返回（可超 Top-K）——Property 6 的边界补充。"""
    # top_k=2，但前三名分数并列(0.9)，应全部返回 3 条。
    hits = [
        _hit(id="a", score=0.9),
        _hit(id="b", score=0.9),
        _hit(id="c", score=0.9),
        _hit(id="d", score=0.5),
    ]
    retriever, _, _ = _make_retriever(hits)

    result = retriever.retrieve("查询", RetrieveOptions(top_k=2, min_score=0.0))

    assert len(result) == 3  # 超过 top_k，因边界并列
    assert {hit.id for hit in result} == {"a", "b", "c"}
    assert all(hit.score == 0.9 for hit in result)


def test_property_6_no_ties_capped_exactly_at_topk():
    """无边界并列时返回数量恰被截断到 Top-K。"""
    hits = [_hit(id=f"c-{i}", score=1.0 - i * 0.1) for i in range(6)]
    retriever, _, _ = _make_retriever(hits)

    result = retriever.retrieve("查询", RetrieveOptions(top_k=3, min_score=0.0))

    assert len(result) == 3
    assert [hit.id for hit in result] == ["c-0", "c-1", "c-2"]


# =========================================================================== #
# 任务 6.3 — 属性测试 Property 7                                                #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 7: 对任意向量检索结果集合与关键词检索结果集合，启用混合检索后合并结果不含重复的 chunk_id，按融合分数降序排列，且返回数量不超过 Top-K。
# Validates: Requirements 8.4


_id_strategy = st.sampled_from([f"id-{i}" for i in range(12)])
_score_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def _id_score_pairs(draw):
    """生成 (id, score) 列表，id 在集合内唯一（同侧不重复，跨侧可重叠）。"""
    pairs = draw(
        st.lists(
            st.tuples(_id_strategy, _score_strategy),
            min_size=0,
            max_size=12,
            unique_by=lambda pair: pair[0],
        )
    )
    return pairs


@settings(max_examples=200, deadline=None)
@given(
    vector_pairs=_id_score_pairs(),
    keyword_pairs=_id_score_pairs(),
    top_k=st.integers(min_value=1, max_value=100),
)
def test_property_7_hybrid_dedup_descending_and_bound(
    vector_pairs: list[tuple[str, float]],
    keyword_pairs: list[tuple[str, float]],
    top_k: int,
):
    """Property 7：混合检索合并结果无重复 id、按融合分数降序、数量 <= Top-K。"""
    vector_hits = [_hit(id=i, score=s) for i, s in vector_pairs]
    keyword_hits = [_hit(id=i, score=s, text="kw") for i, s in keyword_pairs]

    retriever, _, _ = _make_retriever(
        vector_hits, keyword_search=lambda q, limit: keyword_hits
    )

    result = retriever.retrieve(
        "查询", RetrieveOptions(top_k=top_k, min_score=0.0, hybrid=True)
    )

    ids = [hit.id for hit in result]
    # 无重复 chunk_id。
    assert len(ids) == len(set(ids))
    # 按融合分数降序排列。
    scores = [hit.score for hit in result]
    assert scores == sorted(scores, reverse=True)
    # 数量不超过 Top-K。
    assert len(result) <= top_k

    # 融合分数应为同一 id 在两路结果中的较大值（max 融合）。
    fused: dict[str, float] = {}
    for i, s in (*vector_pairs, *keyword_pairs):
        fused[i] = max(fused.get(i, float("-inf")), s)
    expected_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]
    assert set(ids) == set(expected_ids)
    for hit in result:
        assert hit.score == fused[hit.id]


# =========================================================================== #
# 任务 6.4 — 属性测试 Property 8                                                #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 8: 对任意待重排的 Chunk 集合，重排结果按相关性分数降序排列，且重排结果恰为输入集合的一个排列（不增删元素）。
# Validates: Requirements 8.5


@settings(max_examples=200, deadline=None)
@given(
    items=st.lists(
        st.tuples(
            st.text(min_size=0, max_size=20),  # text（含空白/Unicode 由 hypothesis 覆盖）
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        ),
        min_size=0,
        max_size=30,
    ),
    query=st.text(min_size=0, max_size=20),
)
def test_property_8_rerank_is_descending_permutation(
    items: list[tuple[str, float]], query: str
):
    """Property 8：默认重排器输出按相关性降序，且恰为输入的一个排列（不增删元素）。"""
    hits = [
        _hit(id=f"c-{i}", score=score, text=text)
        for i, (text, score) in enumerate(items)
    ]

    reranked = default_reranker(query, hits)

    # 重排结果恰为输入的一个排列：id 的 multiset 完全一致（不增删元素）。
    assert sorted(h.id for h in reranked) == sorted(h.id for h in hits)
    assert len(reranked) == len(hits)
    # 按相关性键降序排列。
    keys = [rerank_relevance(query, hit) for hit in reranked]
    assert keys == sorted(keys, reverse=True)


# =========================================================================== #
# 任务 6.5 — 属性测试 Property 9                                                #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 9: 对任意跨多个 Session 的消息向量库与当前 Session 标识，在该 Session 范围内的召回结果中每条消息都归属于当前 Session，且返回数量不超过配置的历史消息 Top-K（取值范围 1–50）。
# Validates: Requirements 19.4, 10.1


_session_ids = st.sampled_from([f"sess-{i}" for i in range(5)])


@settings(max_examples=200, deadline=None)
@given(
    messages=st.lists(
        st.tuples(
            _session_ids,  # session_id (source_id)
            st.floats(
                min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        ),
        min_size=0,
        max_size=40,
    ),
    current_session=_session_ids,
    top_k=st.integers(min_value=1, max_value=50),
)
def test_property_9_message_recall_session_scope_and_bound(
    messages: list[tuple[str, float]],
    current_session: str,
    top_k: int,
):
    """Property 9：消息召回每条都归属当前 Session，且数量 <= 历史消息 Top-K。"""
    hits = [
        _hit(
            id=f"m-{i}",
            score=score,
            text=f"message-{i}",
            source_id=sess,
            vector_type=VectorType.MESSAGE,
        )
        for i, (sess, score) in enumerate(messages)
    ]
    retriever, _, store = _make_retriever(hits)

    result = retriever.retrieve_messages("当前问题", current_session, top_k=top_k)

    # 每条召回消息都归属当前 Session。
    assert all(hit.source_id == current_session for hit in result)
    # 召回的都是 MESSAGE 向量。
    assert all(hit.vector_type == VectorType.MESSAGE for hit in result)
    # 数量不超过历史消息 Top-K。
    assert len(result) <= top_k
    # 检索范围限定：传给 vector_store 的过滤条件正确。
    assert store.last_source_scope == current_session
    assert store.last_vector_type == VectorType.MESSAGE
    # 按分数降序。
    scores = [hit.score for hit in result]
    assert scores == sorted(scores, reverse=True)


# =========================================================================== #
# 单元测试                                                                      #
# =========================================================================== #

# --- 空 / 空白查询（Req 8.7） ----------------------------------------------- #
@pytest.mark.parametrize("query", ["", "   ", "\t\n  "])
def test_retrieve_empty_query_raises_query_empty(query: str):
    retriever, embed, _ = _make_retriever([_hit(id="a", score=0.9)])
    with pytest.raises(QueryEmptyError):
        retriever.retrieve(query)
    # 校验在嵌入之前完成，不调用 Embedding_Model。
    assert embed.calls == 0


def test_retrieve_messages_empty_query_raises_query_empty():
    retriever, embed, _ = _make_retriever([])
    with pytest.raises(QueryEmptyError):
        retriever.retrieve_messages("   ", "sess-1")
    assert embed.calls == 0


# --- 嵌入失败（Req 8.8） ---------------------------------------------------- #
def test_retrieve_embedding_failure_raises():
    embed = FakeEmbeddingClient(dim=DIM, always_fail=True)
    store = FakeVectorStore([_hit(id="a", score=0.9)], dim=DIM)
    retriever = Retriever(embedding_client=embed, vector_store=store)
    with pytest.raises(QueryEmbeddingFailedError):
        retriever.retrieve("有效查询")


def test_retrieve_embedding_returns_empty_raises():
    embed = FakeEmbeddingClient(dim=DIM, return_empty=True)
    store = FakeVectorStore([_hit(id="a", score=0.9)], dim=DIM)
    retriever = Retriever(embedding_client=embed, vector_store=store)
    with pytest.raises(QueryEmbeddingFailedError):
        retriever.retrieve("有效查询")


def test_retrieve_wraps_unexpected_embedding_error():
    class BoomClient:
        dim = DIM

        def embed(self, texts):
            raise RuntimeError("unexpected network error")

    store = FakeVectorStore([_hit(id="a", score=0.9)], dim=DIM)
    retriever = Retriever(embedding_client=BoomClient(), vector_store=store)
    with pytest.raises(QueryEmbeddingFailedError):
        retriever.retrieve("有效查询")


# --- 无满足阈值命中返回空集（Req 8.6） -------------------------------------- #
def test_retrieve_no_hits_above_threshold_returns_empty():
    hits = [_hit(id="a", score=0.3), _hit(id="b", score=0.4)]
    retriever, _, _ = _make_retriever(hits)
    result = retriever.retrieve("查询", RetrieveOptions(top_k=5, min_score=0.5))
    assert result == []


def test_retrieve_empty_store_returns_empty():
    retriever, _, _ = _make_retriever([])
    result = retriever.retrieve("查询", RetrieveOptions(top_k=5, min_score=0.0))
    assert result == []


# --- 消息召回无历史返回空集（Req 19.5） ------------------------------------- #
def test_retrieve_messages_no_history_returns_empty():
    # 仅存在其他 Session 的消息。
    hits = [
        _hit(id="m1", score=0.9, source_id="other", vector_type=VectorType.MESSAGE),
    ]
    retriever, _, _ = _make_retriever(hits)
    result = retriever.retrieve_messages("查询", "sess-current")
    assert result == []


def test_retrieve_messages_empty_store_returns_empty():
    retriever, _, _ = _make_retriever([])
    result = retriever.retrieve_messages("查询", "sess-1")
    assert result == []


# --- 默认 top_k / min_score 取配置（Req 8.2 默认 5 / 0.5） ------------------- #
def test_retrieve_uses_config_defaults():
    # 6 条分数互异且全部 >= 0.5（默认阈值），默认 top_k=5 应截断到 5。
    hits = [_hit(id=f"c-{i}", score=0.95 - i * 0.05) for i in range(6)]
    retriever, _, _ = _make_retriever(hits)
    result = retriever.retrieve("查询")  # 不传 opts -> 取配置默认
    assert len(result) == 5  # 默认 retrieval_top_k


def test_retrieve_default_min_score_filters_below_threshold():
    hits = [_hit(id="a", score=0.6), _hit(id="b", score=0.4)]
    retriever, _, _ = _make_retriever(hits)
    result = retriever.retrieve("查询")  # 默认 min_similarity_threshold=0.5
    assert [hit.id for hit in result] == ["a"]


# --- hybrid=False 忽略 keyword_search -------------------------------------- #
def test_retrieve_hybrid_false_ignores_keyword_search():
    called = {"count": 0}

    def keyword_search(query, limit):
        called["count"] += 1
        return [_hit(id="kw-only", score=1.0)]

    hits = [_hit(id="vec", score=0.9)]
    retriever, _, _ = _make_retriever(hits, keyword_search=keyword_search)

    result = retriever.retrieve("查询", RetrieveOptions(top_k=5, min_score=0.0))

    assert called["count"] == 0  # 未启用 hybrid 不调用关键词检索
    assert [hit.id for hit in result] == ["vec"]


def test_retrieve_hybrid_merges_vector_and_keyword():
    vector_hits = [_hit(id="shared", score=0.6), _hit(id="vonly", score=0.5)]
    keyword_hits = [_hit(id="shared", score=0.9), _hit(id="konly", score=0.7)]
    retriever, _, _ = _make_retriever(
        vector_hits, keyword_search=lambda q, limit: keyword_hits
    )

    result = retriever.retrieve(
        "查询", RetrieveOptions(top_k=5, min_score=0.0, hybrid=True)
    )

    ids = [hit.id for hit in result]
    assert len(ids) == len(set(ids))  # 去重
    assert set(ids) == {"shared", "vonly", "konly"}
    # shared 融合分数取较大值 0.9，应排在最前。
    assert result[0].id == "shared"
    assert result[0].score == 0.9
    # 降序。
    scores = [hit.score for hit in result]
    assert scores == sorted(scores, reverse=True)


# --- 默认 vector_type 为 DOC_CHUNK ----------------------------------------- #
def test_retrieve_defaults_to_doc_chunk_vector_type():
    retriever, _, store = _make_retriever([_hit(id="a", score=0.9)])
    retriever.retrieve("查询", RetrieveOptions(top_k=5, min_score=0.0))
    assert store.last_vector_type == VectorType.DOC_CHUNK


def test_retrieve_messages_forces_message_type_and_session_scope():
    hits = [
        _hit(id="m1", score=0.9, source_id="sess-1", vector_type=VectorType.MESSAGE),
        _hit(id="m2", score=0.8, source_id="sess-2", vector_type=VectorType.MESSAGE),
        _hit(id="d1", score=0.95, source_id="sess-1", vector_type=VectorType.DOC_CHUNK),
    ]
    retriever, _, store = _make_retriever(hits)

    result = retriever.retrieve_messages("查询", "sess-1", top_k=10)

    assert store.last_vector_type == VectorType.MESSAGE
    assert store.last_source_scope == "sess-1"
    # 仅返回 sess-1 的 MESSAGE 向量（doc chunk 与其他 session 被过滤）。
    assert [hit.id for hit in result] == ["m1"]


def test_retrieve_messages_respects_top_k_cap():
    hits = [
        _hit(
            id=f"m-{i}",
            score=1.0 - i * 0.05,
            source_id="sess-1",
            vector_type=VectorType.MESSAGE,
        )
        for i in range(10)
    ]
    retriever, _, _ = _make_retriever(hits)
    result = retriever.retrieve_messages("查询", "sess-1", top_k=3)
    assert len(result) == 3
    assert [hit.id for hit in result] == ["m-0", "m-1", "m-2"]


# --- rerank 启用时按相关性降序重排 ----------------------------------------- #
def test_retrieve_rerank_orders_by_relevance():
    # 查询词 "alpha"：含该词的命中关键词重叠更高，应排前，即便其向量分数更低。
    hits = [
        _hit(id="low-overlap", score=0.9, text="beta gamma"),
        _hit(id="high-overlap", score=0.6, text="alpha beta"),
    ]
    retriever, _, _ = _make_retriever(hits)

    result = retriever.retrieve(
        "alpha", RetrieveOptions(top_k=5, min_score=0.0, rerank=True)
    )

    assert [hit.id for hit in result] == ["high-overlap", "low-overlap"]
    # 重排不增删元素。
    assert len(result) == 2


def test_retrieve_returns_search_hit_instances():
    retriever, _, _ = _make_retriever([_hit(id="a", score=0.9)])
    result = retriever.retrieve("查询", RetrieveOptions(top_k=5, min_score=0.0))
    assert all(isinstance(hit, SearchHit) for hit in result)

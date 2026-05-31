"""Knowledge_Base_Agent 测试（任务 11.1–11.7）。

覆盖：
- 入库管线 ingest（Req 5.1/6.5/7.1/3.6/3.7/22.1）：happy path、解析失败、分片失败、空分片。
- 检索增强生成 answer（Req 9.1–9.6）：空召回提示、非空召回引用、生成错误透传。
- 同步/移除 sync/remove（Req 22.2–22.6）：替换、失败保持既有、删除计数往返、不存在提示。
- 属性测试（Hypothesis, max_examples>=100, deadline=None）：
  - Property 11（11.3）：答案引用来源是召回来源的子集。
  - Property 27（11.5）：同源连续同步两版后恰等于新版 Chunk 集合，不残留旧版。
  - Property 28（11.6）：移除来源返回删除数量 == N 且剩余为 0。
  - Property 29（11.7）：同步任一阶段失败后既有 Chunk 与失败前完全一致。

测试以内存替身隔离外部依赖：
- ``FakeEmbeddingClient``：确定性定长向量，可注入失败。
- ``FakeVectorStore``：内存版向量库，支持 write / write_many / delete_by_source / search，
  按 source_id 跟踪 Chunk。
- ``SpyChatModel``：返回可配置 Completion 或抛出生成错误，记录调用。
- ``StubRetriever``：返回预置 SearchHit 列表，便于精确控制召回集合。
真实 Loader + Transformer 用于 ingest/sync（二者纯离线），Chunk 生成确定性。
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.agents.knowledge_base_agent import (
    AnswerResult,
    IngestResult,
    KnowledgeBaseAgent,
    RemoveResult,
    SyncResult,
)
from ghost_agent.core.chat_model import Completion
from ghost_agent.core.loader import Loader
from ghost_agent.core.transformer import Transformer
from ghost_agent.models.errors import (
    GenerationError,
    QueryEmbeddingFailedError,
    SplitFailedError,
)
from ghost_agent.models.ingest_task import IngestTaskStatus
from ghost_agent.models.vector_record import VectorRecord, VectorType
from ghost_agent.vector_db.vector_store import SearchHit

DIM = 4


# --------------------------------------------------------------------------- #
# 内存替身                                                                      #
# --------------------------------------------------------------------------- #
class FakeEmbeddingClient:
    """内存版 Embedding 客户端：返回确定性定长向量，可注入失败。

    * ``always_fail`` —— 任意 ``embed`` 调用抛 :class:`QueryEmbeddingFailedError`。
    * ``fail_texts``  —— 当批次中包含命中文本时整体失败（用于 sync 全量嵌入失败）。
    :attr:`calls` 统计调用次数。
    """

    def __init__(
        self,
        *,
        dim: int = DIM,
        always_fail: bool = False,
        fail_texts: set[str] | None = None,
    ) -> None:
        self._dim = dim
        self._always_fail = always_fail
        self._fail_texts = set(fail_texts or [])
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self._always_fail:
            raise QueryEmbeddingFailedError("always fail")
        if any(t in self._fail_texts for t in texts):
            raise QueryEmbeddingFailedError("configured text failure")
        return [self._vector_for(t) for t in texts]

    def _vector_for(self, text: str) -> list[float]:
        return [float((len(text) + i) % 7) + 0.5 for i in range(self._dim)]


class FakeVectorStore:
    """内存版 vector_store：按 source_id 跟踪写入的记录。

    支持 :meth:`write` / :meth:`write_many` / :meth:`delete_by_source` /
    :meth:`search`，足以驱动 ingest（write）、sync（delete + write_many）、
    remove（delete_by_source）与默认 Retriever（search，本测试不依赖）。
    """

    def __init__(self, *, dim: int = DIM) -> None:
        self._dim = dim
        self.records: list[VectorRecord] = []

    @property
    def dim(self) -> int:
        return self._dim

    def write(self, record: VectorRecord) -> None:
        self.records.append(record)

    def write_many(self, records: list[VectorRecord]) -> int:
        self.records.extend(records)
        return len(records)

    def delete_by_source(self, source_file_id: str) -> int:
        before = len(self.records)
        self.records = [r for r in self.records if r.source_id != source_file_id]
        return before - len(self.records)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        min_score: float,
        source_scope: str | None = None,
        vector_type: VectorType | None = None,
    ) -> list[SearchHit]:  # pragma: no cover - 本测试通过 StubRetriever 控制召回
        return []

    # 测试辅助
    def texts_for(self, source_file_id: str) -> list[str]:
        return sorted(r.text for r in self.records if r.source_id == source_file_id)

    def count_for(self, source_file_id: str) -> int:
        return sum(1 for r in self.records if r.source_id == source_file_id)


class SpyChatModel:
    """记录调用的 Chat_Model 替身：返回可配置 Completion 或抛出错误。"""

    def __init__(self, *, content: str = "生成的答案", error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.calls: list[list] = []

    def generate(self, messages, *, temperature: float | None = None) -> Completion:
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        return Completion(content=self._content)


class StubRetriever:
    """返回预置命中列表的 Retriever 替身。"""

    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = list(hits)
        self.calls = 0

    def retrieve(self, query: str, opts=None) -> list[SearchHit]:
        self.calls += 1
        return list(self._hits)


class StubSplitFailTransformer:
    """split 始终抛 :class:`SplitFailedError` 的 Transformer 替身（驱动分片失败阶段）。"""

    def split(self, parse_result, strategy=None):
        raise SplitFailedError("configured split failure")


# --------------------------------------------------------------------------- #
# 工厂                                                                          #
# --------------------------------------------------------------------------- #
def _make_hit(*, id: str, source_id: str, score: float = 0.9, text: str = "片段") -> SearchHit:  # noqa: A002
    return SearchHit(
        id=id,
        text=text,
        source_id=source_id,
        vector_type=VectorType.DOC_CHUNK,
        score=score,
        metadata={},
    )


def _small_transformer() -> Transformer:
    """长度较小的 Transformer，便于由普通文本产生多个确定性 Chunk。"""
    return Transformer(min_length=1, max_length=30, max_input_length=30)


def _build_kb(
    *,
    loader: Loader | None = None,
    transformer=None,
    embedding_client: FakeEmbeddingClient | None = None,
    vector_store: FakeVectorStore | None = None,
    chat_model=None,
    retriever=None,
) -> KnowledgeBaseAgent:
    """构造 KBA：注入 embedding/store 后默认 Indexer/Retriever 亦为离线。"""
    return KnowledgeBaseAgent(
        loader=loader if loader is not None else Loader(),
        transformer=transformer,
        embedding_client=embedding_client
        if embedding_client is not None
        else FakeEmbeddingClient(dim=DIM),
        vector_store=vector_store if vector_store is not None else FakeVectorStore(dim=DIM),
        chat_model=chat_model,
        retriever=retriever,
    )


# 生成可被 Loader 解析（非空）的文本内容：以固定前缀保证去空白后非空。
_content_chars = st.characters(
    whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
    whitelist_characters="\n。，",
)
_content_strategy = st.text(_content_chars, min_size=0, max_size=160).map(
    lambda s: "DOC" + s
)


# =========================================================================== #
# 11.1 — ingest 单元测试                                                        #
# =========================================================================== #
def test_ingest_happy_path_completed_with_chunk_count_and_source_id():
    """ingest 成功路径：COMPLETED、chunk_count 与产出一致、每个 Chunk 附来源标识。"""
    store = FakeVectorStore(dim=DIM)
    embed = FakeEmbeddingClient(dim=DIM)
    kb = _build_kb(embedding_client=embed, vector_store=store)

    result = kb.ingest(
        content="DOC 这是一段运维知识库的正文内容。",
        file_name="manual.txt",
        file_format="txt",
        source_file_id="src-ingest-1",
    )

    assert isinstance(result, IngestResult)
    assert result.status is IngestTaskStatus.COMPLETED
    assert result.chunk_count is not None and result.chunk_count >= 1
    # 写入记录数与 chunk_count 一致，且每条记录都附带来源文件标识（Req 22.1）。
    assert store.count_for("src-ingest-1") == result.chunk_count
    assert all(r.source_id == "src-ingest-1" for r in store.records)
    assert all(r.vector_type == VectorType.DOC_CHUNK for r in store.records)


def test_ingest_parse_failure_returns_failed_with_reason():
    """ingest 解析失败（不支持的类型）：FAILED 并附失败原因（Req 3.7/5.3）。"""
    store = FakeVectorStore(dim=DIM)
    kb = _build_kb(vector_store=store)

    result = kb.ingest(
        content=b"\x00\x01binary",
        file_name="bad.bin",
        file_format="bin",
        source_file_id="src-bad",
    )

    assert result.status is IngestTaskStatus.FAILED
    assert result.failure_reason is not None and result.failure_reason.strip()
    assert result.chunk_count is None
    assert store.records == []  # 解析失败不写入任何 Chunk


def test_ingest_empty_content_returns_failed():
    """ingest 内容为空（仅空白）：Loader 判定内容为空 → FAILED（Req 5.6）。"""
    store = FakeVectorStore(dim=DIM)
    kb = _build_kb(vector_store=store)

    result = kb.ingest(
        content="    \n\t  ",
        file_name="empty.txt",
        file_format="txt",
        source_file_id="src-empty",
    )

    assert result.status is IngestTaskStatus.FAILED
    assert result.failure_reason
    assert store.records == []


def test_ingest_split_failure_returns_failed():
    """ingest 分片失败：Transformer 抛 SplitFailedError → FAILED（Req 6.7）。"""
    store = FakeVectorStore(dim=DIM)
    kb = _build_kb(transformer=StubSplitFailTransformer(), vector_store=store)

    result = kb.ingest(
        content="DOC 正文内容",
        file_name="x.txt",
        file_format="txt",
        source_file_id="src-split",
    )

    assert result.status is IngestTaskStatus.FAILED
    assert result.failure_reason
    assert store.records == []


def test_ingest_empty_chunks_returns_failed():
    """ingest 未产生任何分片：FAILED，reason 指明未产生分片（Req 6.6）。"""

    class EmptyTransformer:
        def split(self, parse_result, strategy=None):
            return []

    store = FakeVectorStore(dim=DIM)
    kb = _build_kb(transformer=EmptyTransformer(), vector_store=store)

    result = kb.ingest(
        content="DOC 正文",
        file_name="x.txt",
        file_format="txt",
        source_file_id="src-noc",
    )

    assert result.status is IngestTaskStatus.FAILED
    assert "分片" in (result.failure_reason or "")
    assert store.records == []


def test_ingest_all_chunks_fail_index_returns_failed():
    """ingest 全部分片嵌入失败：success_count==0 → FAILED（不写入任何记录）。"""
    store = FakeVectorStore(dim=DIM)
    embed = FakeEmbeddingClient(dim=DIM, always_fail=True)
    kb = _build_kb(embedding_client=embed, vector_store=store)

    result = kb.ingest(
        content="DOC 一段正文",
        file_name="x.txt",
        file_format="txt",
        source_file_id="src-allfail",
    )

    assert result.status is IngestTaskStatus.FAILED
    assert result.failure_reason
    assert store.records == []


# =========================================================================== #
# 11.2 — answer 单元测试                                                        #
# =========================================================================== #
def test_answer_empty_recall_returns_not_found_without_calling_chat_model():
    """空召回：返回未检索到提示、cited_sources 为空、不调用 Chat_Model（Req 9.3）。"""
    spy = SpyChatModel()
    kb = _build_kb(retriever=StubRetriever([]), chat_model=spy)

    result = kb.answer("如何处理告警？")

    assert isinstance(result, AnswerResult)
    assert "未在知识库" in result.answer
    assert result.cited_sources == []
    assert spy.calls == []  # 未调用 Chat_Model，不臆造


def test_answer_with_recall_cites_unique_sources_and_builds_prompt():
    """非空召回：cited_sources == 召回去重来源；提示词含查询与各来源标识（Req 9.1/9.4）。"""
    hits = [
        _make_hit(id="h1", source_id="srcA", text="片段一"),
        _make_hit(id="h2", source_id="srcB", text="片段二"),
        _make_hit(id="h3", source_id="srcA", text="片段三"),
    ]
    spy = SpyChatModel(content="基于片段的答案")
    kb = _build_kb(retriever=StubRetriever(hits), chat_model=spy)

    result = kb.answer("查询关键字XYZ")

    assert result.answer == "基于片段的答案"
    assert result.cited_sources == ["srcA", "srcB"]
    # Chat_Model 被调用且提示词包含用户查询与每个来源标识。
    assert len(spy.calls) == 1
    prompt_text = spy.calls[0][0].content
    assert "查询关键字XYZ" in prompt_text
    assert "srcA" in prompt_text and "srcB" in prompt_text


def test_answer_generation_error_propagates_without_partial_answer():
    """生成错误：透传 GenerationError，不返回部分/臆造答案（Req 9.5）。"""
    hits = [_make_hit(id="h1", source_id="srcA")]
    spy = SpyChatModel(error=GenerationError("模型炸了"))
    kb = _build_kb(retriever=StubRetriever(hits), chat_model=spy)

    with pytest.raises(GenerationError):
        kb.answer("查询")


# =========================================================================== #
# 11.4 — sync / remove 单元测试                                                 #
# =========================================================================== #
def test_sync_replaces_old_chunks_and_keeps_other_sources():
    """sync 同源替换：旧版被替换为新版；其他来源不受影响（Req 22.2）。"""
    store = FakeVectorStore(dim=DIM)
    loader = Loader()
    transformer = _small_transformer()
    kb = _build_kb(loader=loader, transformer=transformer, vector_store=store)

    # 预置另一来源，验证不被误删。
    store.write(
        VectorRecord(
            vector=[0.1] * DIM,
            text="other-chunk",
            source_id="OTHER",
            vector_type=VectorType.DOC_CHUNK,
        )
    )

    r1 = kb.sync(content="DOC 第一版内容 alpha beta gamma", file_name="f.txt", file_format="txt", source_file_id="S1")
    assert r1.status == "COMPLETED"
    v1_texts = store.texts_for("S1")
    assert v1_texts

    r2 = kb.sync(content="DOC 第二版完全不同的内容 delta epsilon", file_name="f.txt", file_format="txt", source_file_id="S1")
    assert r2.status == "COMPLETED"

    expected_v2 = sorted(t.text for t in transformer.split(loader.parse(content="DOC 第二版完全不同的内容 delta epsilon", file_name="f.txt", file_format="txt", source_file_id="S1")))
    assert store.texts_for("S1") == expected_v2
    # 其他来源保持不变。
    assert store.count_for("OTHER") == 1


def test_sync_failure_keeps_existing_chunks():
    """sync 嵌入阶段失败：保持既有 Chunk 不变，返回 FAILED + 失败阶段（Req 22.5）。"""
    store = FakeVectorStore(dim=DIM)
    loader = Loader()
    transformer = _small_transformer()
    kb_good = _build_kb(loader=loader, transformer=transformer, vector_store=store)

    r1 = kb_good.sync(content="DOC 初始版本内容", file_name="f.txt", file_format="txt", source_file_id="S2")
    assert r1.status == "COMPLETED"
    snapshot = store.texts_for("S2")
    assert snapshot

    kb_fail = _build_kb(
        loader=loader,
        transformer=transformer,
        embedding_client=FakeEmbeddingClient(dim=DIM, always_fail=True),
        vector_store=store,
    )
    res = kb_fail.sync(content="DOC 新版内容将无法嵌入", file_name="f.txt", file_format="txt", source_file_id="S2")

    assert res.status == "FAILED"
    assert res.failed_stage == "embed_write"
    assert res.failure_reason
    # 既有 Chunk 与失败前完全一致。
    assert store.texts_for("S2") == snapshot


def test_remove_existing_source_returns_count_and_found():
    """remove 已存在来源：返回删除数量与 found=True（Req 22.3）。"""
    store = FakeVectorStore(dim=DIM)
    for i in range(3):
        store.write(
            VectorRecord(
                vector=[0.2] * DIM,
                text=f"chunk-{i}",
                source_id="S3",
                vector_type=VectorType.DOC_CHUNK,
            )
        )
    kb = _build_kb(vector_store=store)

    result = kb.remove("S3")

    assert isinstance(result, RemoveResult)
    assert result.deleted_count == 3
    assert result.found is True
    assert store.count_for("S3") == 0


def test_remove_nonexistent_source_returns_not_found():
    """remove 不存在来源：不删除任何 Chunk，found=False（Req 22.6）。"""
    store = FakeVectorStore(dim=DIM)
    kb = _build_kb(vector_store=store)

    result = kb.remove("missing")

    assert result.deleted_count == 0
    assert result.found is False


# =========================================================================== #
# 11.3 — 属性测试 Property 11                                                    #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 11: 对任意召回 Chunk 集合与据此生成的答案，答案附带的引用来源文件标识列表是召回集合来源标识集合的子集（不包含召回集合以外的来源）。
# Validates: Requirements 9.4

_source_pool = [f"src-{i}" for i in range(4)]


@st.composite
def _recalled_hits(draw):
    n = draw(st.integers(min_value=0, max_value=8))
    hits = []
    for i in range(n):
        source_id = draw(st.sampled_from(_source_pool))
        score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        text = draw(st.text(min_size=0, max_size=20))
        hits.append(_make_hit(id=f"h{i}", source_id=source_id, score=score, text=text))
    return hits


@settings(max_examples=150, deadline=None)
@given(hits=_recalled_hits(), chat_content=st.text(min_size=0, max_size=40), query=st.text(min_size=1, max_size=20))
def test_property_11_cited_sources_subset_of_recalled(hits, chat_content, query):
    """Property 11：cited_sources ⊆ 召回命中的来源标识集合（不含召回以外来源）。"""
    spy = SpyChatModel(content=chat_content)
    kb = _build_kb(retriever=StubRetriever(hits), chat_model=spy)

    result = kb.answer(query)

    recalled_sources = {hit.source_id for hit in hits}
    # 引用来源是召回来源集合的子集（核心断言）。
    assert set(result.cited_sources) <= recalled_sources
    # 空召回时不引用任何来源且不调用 Chat_Model。
    if not hits:
        assert result.cited_sources == []
        assert spy.calls == []


# =========================================================================== #
# 11.5 — 属性测试 Property 27                                                    #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 27: 对任意来源文件，对其连续同步两版内容后，Vector_Database 中归属该来源文件的 Chunk 集合恰等于新一版生成的 Chunk 集合，不残留任何旧版 Chunk。
# Validates: Requirements 22.2

_sid_pool = [f"S-{i}" for i in range(4)]


@settings(max_examples=100, deadline=None)
@given(v1=_content_strategy, v2=_content_strategy, sid=st.sampled_from(_sid_pool))
def test_property_27_resync_replaces_without_leftover(v1: str, v2: str, sid: str):
    """Property 27：连续同步两版后，该来源的 Chunk 集合恰等于第二版生成的 Chunk 集合。"""
    store = FakeVectorStore(dim=DIM)
    loader = Loader()
    transformer = _small_transformer()
    kb = _build_kb(loader=loader, transformer=transformer, vector_store=store)

    r1 = kb.sync(content=v1, file_name="f.txt", file_format="txt", source_file_id=sid)
    r2 = kb.sync(content=v2, file_name="f.txt", file_format="txt", source_file_id=sid)
    assert r1.status == "COMPLETED"
    assert r2.status == "COMPLETED"

    expected_v2 = sorted(
        c.text
        for c in transformer.split(
            loader.parse(content=v2, file_name="f.txt", file_format="txt", source_file_id=sid)
        )
    )
    # 该来源现存 Chunk 恰等于第二版 Chunk 集合，无旧版残留。
    assert store.texts_for(sid) == expected_v2
    assert store.count_for(sid) == r2.success_count


# =========================================================================== #
# 11.6 — 属性测试 Property 28                                                    #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 28: 对任意已写入 N 个 Chunk 的来源文件，移除该来源后返回的删除数量等于 N，且该来源在 Vector_Database 中剩余 Chunk 数为 0。
# Validates: Requirements 22.3


@settings(max_examples=100, deadline=None)
@given(n=st.integers(min_value=0, max_value=25), other_n=st.integers(min_value=0, max_value=10))
def test_property_28_remove_count_roundtrip(n: int, other_n: int):
    """Property 28：移除来源后删除数量 == N 且剩余为 0；其他来源不受影响。"""
    store = FakeVectorStore(dim=DIM)
    for i in range(n):
        store.write(
            VectorRecord(
                vector=[0.3] * DIM,
                text=f"t-{i}",
                source_id="TARGET",
                vector_type=VectorType.DOC_CHUNK,
            )
        )
    for j in range(other_n):
        store.write(
            VectorRecord(
                vector=[0.4] * DIM,
                text=f"o-{j}",
                source_id="OTHER",
                vector_type=VectorType.DOC_CHUNK,
            )
        )
    kb = _build_kb(vector_store=store)

    result = kb.remove("TARGET")

    assert result.deleted_count == n
    assert store.count_for("TARGET") == 0
    assert result.found is (n > 0)
    # 其他来源保持不变。
    assert store.count_for("OTHER") == other_n


# =========================================================================== #
# 11.7 — 属性测试 Property 29                                                    #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 29: 对任意在加载、分片、嵌入或写入任一阶段失败的同步流程，失败后该来源文件在 Vector_Database 中已有的 Chunk 集合与失败前完全一致（失败不破坏既有状态）。
# Validates: Requirements 22.5


@settings(max_examples=100, deadline=None)
@given(
    v1=_content_strategy,
    v2=_content_strategy,
    sid=st.sampled_from(_sid_pool),
    stage=st.sampled_from(["load", "split", "embed"]),
)
def test_property_29_sync_failure_keeps_existing(v1: str, v2: str, sid: str, stage: str):
    """Property 29：同步任一阶段失败后，该来源既有 Chunk 与失败前完全一致。"""
    store = FakeVectorStore(dim=DIM)
    loader = Loader()
    transformer = _small_transformer()

    # 先成功同步第一版，建立既有状态。
    kb_good = _build_kb(loader=loader, transformer=transformer, vector_store=store)
    r1 = kb_good.sync(content=v1, file_name="f.txt", file_format="txt", source_file_id=sid)
    assert r1.status == "COMPLETED"
    snapshot = store.texts_for(sid)
    assert snapshot  # 第一版至少产生一个 Chunk

    # 在指定阶段注入失败，再次同步第二版。
    if stage == "load":
        # 不支持的文件类型 → Loader 解析失败（load 阶段）。
        res = kb_good.sync(content=v2, file_name="f.bin", file_format="bin", source_file_id=sid)
        expected_stage = "load"
    elif stage == "split":
        kb_fail = _build_kb(loader=loader, transformer=StubSplitFailTransformer(), vector_store=store)
        res = kb_fail.sync(content=v2, file_name="f.txt", file_format="txt", source_file_id=sid)
        expected_stage = "split"
    else:  # embed
        kb_fail = _build_kb(
            loader=loader,
            transformer=transformer,
            embedding_client=FakeEmbeddingClient(dim=DIM, always_fail=True),
            vector_store=store,
        )
        res = kb_fail.sync(content=v2, file_name="f.txt", file_format="txt", source_file_id=sid)
        expected_stage = "embed_write"

    assert res.status == "FAILED"
    assert res.failed_stage == expected_stage
    # 失败不破坏既有状态：该来源 Chunk 集合与失败前完全一致。
    assert store.texts_for(sid) == snapshot

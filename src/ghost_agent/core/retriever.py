"""核心组件：Retriever（检索召回，Req 8, 19）。

Retriever 在 :class:`VectorStore` 之上做检索编排：把非空查询转换为查询向量
（Req 8.1），从 Vector_Database 召回相似度 ≥ ``min_score`` 且最高的 Chunk
（数量 ≤ Top-K，Req 8.2），并在此基础上叠加以下能力：

* **Top-K 边界并列全部返回（Req 8.3）**：当第 Top-K 名与其后若干命中的相似度分数
  相等时，全部返回（此时返回数量可超过 Top-K）。
* **混合检索（Hybrid_Retrieval，Req 8.4）**：合并向量检索结果与关键词检索结果，
  按 ``id`` 去重（融合分数取二者较大值），按融合分数降序，返回 ≤ Top-K。
* **重排（Rerank，Req 8.5）**：对召回集合按相关性分数降序重排，结果恰为输入集合的
  一个排列（不增删元素）。
* **消息向量召回（Req 19.3–19.5）**：:meth:`retrieve_messages` 限定当前 Session 范围
  召回相似度最高的历史消息（数量由历史消息 Top-K 决定），无历史消息返回空集。

错误处理：
* 查询为空或去除首尾空白后为空 → :class:`QueryEmptyError`（Req 8.7）。
* 查询向量生成失败 → :class:`QueryEmbeddingFailedError`（Req 8.8）。
* 无满足阈值的命中 → 返回空集（非错误，Req 8.6 / 19.5）。

设计要点：
- **职责分层**：:class:`VectorStore.search` 已完成 ``min_score`` 过滤、降序与按 limit
  截断；Retriever 在其上追加查询嵌入、Top-K 边界并列、混合合并、重排与消息范围召回。
- **可注入 seam**：``embedding_client`` / ``vector_store`` / ``reranker`` /
  ``keyword_search`` 均可经构造函数注入；默认实现确定性、无网络依赖，便于属性测试。
  关键词检索默认返回 ``[]``（无关键词后端），生产可注入真实关键词后端。
- **惰性构造**：默认 ``DoubaoEmbeddingClient`` 与 ``VectorStore`` 构造期均不触网。
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from ghost_agent.clients.doubao_client import DoubaoEmbeddingClient
from ghost_agent.config import get_settings
from ghost_agent.models.errors import QueryEmbeddingFailedError, QueryEmptyError
from ghost_agent.models.vector_record import VectorType
from ghost_agent.vector_db.vector_store import SearchHit, VectorStore

__all__ = [
    "Retriever",
    "RetrieveOptions",
    "default_reranker",
    "default_keyword_search",
    "rerank_relevance",
]

# 为在 Top-K 边界纳入并列项而向 vector_store 额外多取的候选数量。
# 内存替身会忽略 limit 直接返回全部命中；该 buffer 仅对真实 Milvus 生效，
# 用于尽量纳入边界并列项（极端情况下并列项超过 buffer 时无法全部纳入，已知近似）。
_TIE_OVER_FETCH_BUFFER = 64

# 消息向量召回不设相似度阈值（Req 19.4 仅要求"相似度最高的历史消息"）。
# cosine 取值域为 [-1, 1]，-1.0 可接受全部有效命中。
_MESSAGE_RECALL_MIN_SCORE = -1.0


# --------------------------------------------------------------------------- #
# 确定性关键词检索 / 重排默认实现（可注入 seam）                                  #
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> set[str]:
    """将文本切分为小写词元集合（按空白切分），用于关键词重叠度量。"""
    return {token for token in text.lower().split() if token}


def rerank_relevance(query: str, hit: SearchHit) -> tuple[int, float]:
    """计算单条命中相对查询的相关性键（用于重排降序排序）。

    相关性键为二元组 ``(关键词重叠数, 原始相似度分数)``：先按查询与命中文本的关键词
    重叠数比较，重叠数相等时再按原始相似度分数比较。该键是确定性的纯函数，
    便于属性测试复算并验证降序。
    """
    overlap = len(_tokenize(query) & _tokenize(hit.text))
    return (overlap, hit.score)


def default_reranker(query: str, hits: list[SearchHit]) -> list[SearchHit]:
    """默认重排器：按相关性键降序重排，返回输入集合的一个排列（不增删元素，Req 8.5）。

    使用稳定排序按 :func:`rerank_relevance` 降序排列；因仅重排不增删，输出恰为输入的
    一个排列（Property 8）。
    """
    return sorted(hits, key=lambda hit: rerank_relevance(query, hit), reverse=True)


def default_keyword_search(query: str, limit: int) -> list[SearchHit]:  # noqa: ARG001
    """默认关键词检索：无关键词后端，返回空列表。

    生产环境可注入真实关键词检索后端；默认返回 ``[]`` 时启用混合检索等价于仅向量检索
    （仍保证去重与降序）。
    """
    return []


# --------------------------------------------------------------------------- #
# 选项                                                                          #
# --------------------------------------------------------------------------- #
class RetrieveOptions(BaseModel):
    """:meth:`Retriever.retrieve` 的检索选项。"""

    model_config = ConfigDict(extra="forbid")

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="召回数量上界（1–100）；为 None 时取 settings.retrieval_top_k（默认 5），Req 8.2。",
    )
    min_score: float | None = Field(
        default=None,
        description="最小相似度阈值；为 None 时取 settings.min_similarity_threshold（默认 0.5），Req 8.2。",
    )
    hybrid: bool = Field(default=False, description="是否启用混合检索（Req 8.4）。")
    rerank: bool = Field(default=False, description="是否启用重排（Req 8.5）。")
    source_scope: str | None = Field(
        default=None,
        description="可选，仅检索 source_id == source_scope 的记录。",
    )
    vector_type: VectorType | None = Field(
        default=None,
        description="向量类型；为 None 时默认 DOC_CHUNK（文档检索）。",
    )


# 关键词检索与重排器的可注入函数签名。
KeywordSearch = Callable[[str, int], list[SearchHit]]
Reranker = Callable[[str, list[SearchHit]], list[SearchHit]]


class Retriever:
    """检索器：查询嵌入 + 向量召回 + Top-K 边界并列 + 混合检索 + 重排 + 消息召回。

    Args:
        embedding_client: Doubao Embedding 客户端；为 ``None`` 时构造默认
            :class:`DoubaoEmbeddingClient`（构造期不触网）。
        vector_store: 向量存储；为 ``None`` 时构造默认 :class:`VectorStore`
            （构造期不连接 Milvus）。
        reranker: 重排器（query, hits -> 重排后的 hits）；为 ``None`` 时用
            :func:`default_reranker`。
        keyword_search: 关键词检索（query, limit -> hits）；为 ``None`` 时用
            :func:`default_keyword_search`（返回 ``[]``）。
    """

    def __init__(
        self,
        *,
        embedding_client: DoubaoEmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        keyword_search: KeywordSearch | None = None,
    ) -> None:
        self._embedding_client = (
            embedding_client if embedding_client is not None else DoubaoEmbeddingClient()
        )
        self._vector_store = vector_store if vector_store is not None else VectorStore()
        self._reranker: Reranker = reranker if reranker is not None else default_reranker
        self._keyword_search: KeywordSearch = (
            keyword_search if keyword_search is not None else default_keyword_search
        )

    # ------------------------------------------------------------------ #
    # 文档检索（Req 8）                                                    #
    # ------------------------------------------------------------------ #
    def retrieve(
        self, query: str, opts: RetrieveOptions | None = None
    ) -> list[SearchHit]:
        """根据查询召回相关 Chunk。

        流程：校验查询非空（Req 8.7）→ 生成查询向量（Req 8.1，失败抛 Req 8.8）→
        向量召回（Req 8.2）→ Top-K 边界并列全部返回（Req 8.3）/ 启用混合检索时合并去重
        降序截断（Req 8.4）→ 启用重排时按相关性降序重排（Req 8.5）。无满足阈值命中时
        返回空集（Req 8.6）。

        Args:
            query: 用户查询；为空或 trim 后为空时抛 :class:`QueryEmptyError`。
            opts: 检索选项；为 ``None`` 时使用默认选项（取配置默认值）。

        Returns:
            召回的 :class:`SearchHit` 列表，按相似度（或重排相关性）降序。

        Raises:
            QueryEmptyError: 查询为空或去空白后为空（Req 8.7）。
            QueryEmbeddingFailedError: 查询向量生成失败（Req 8.8）。
        """
        options = opts if opts is not None else RetrieveOptions()
        settings = get_settings()
        top_k = options.top_k if options.top_k is not None else settings.retrieval_top_k
        min_score = (
            options.min_score
            if options.min_score is not None
            else settings.min_similarity_threshold
        )
        vector_type = (
            options.vector_type
            if options.vector_type is not None
            else VectorType.DOC_CHUNK
        )

        query_vector = self._embed_query(query)

        if options.hybrid:
            hits = self._hybrid_merge(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                min_score=min_score,
                source_scope=options.source_scope,
                vector_type=vector_type,
            )
        else:
            base = self._vector_search(
                query_vector=query_vector,
                limit=top_k + _TIE_OVER_FETCH_BUFFER,
                min_score=min_score,
                source_scope=options.source_scope,
                vector_type=vector_type,
            )
            hits = self._cap_with_ties(base, top_k)

        if options.rerank:
            hits = list(self._reranker(query, hits))

        return hits

    # ------------------------------------------------------------------ #
    # 消息向量召回（Req 19.3–19.5）                                         #
    # ------------------------------------------------------------------ #
    def retrieve_messages(
        self, query: str, session_id: str, top_k: int | None = None
    ) -> list[SearchHit]:
        """在当前 Session 范围内召回相似度最高的历史消息（Req 19.4）。

        将当前用户消息转换为查询向量（Req 19.3），限定 ``vector_type=MESSAGE`` 且
        ``source_id == session_id`` 召回，数量由历史消息 Top-K 决定。当前 Session 无可
        召回历史消息时返回空集（Req 19.5）。

        Args:
            query: 当前用户消息；为空或 trim 后为空时抛 :class:`QueryEmptyError`。
            session_id: 当前会话标识（召回范围限定键）。
            top_k: 召回数量上界；为 ``None`` 时取 ``settings.history_message_top_k``（1–50）。

        Returns:
            归属当前 Session 的历史消息命中列表（数量 ≤ Top-K），按相似度降序。

        Raises:
            QueryEmptyError: 查询为空或去空白后为空（Req 8.7）。
            QueryEmbeddingFailedError: 查询向量生成失败（Req 8.8）。
        """
        settings = get_settings()
        limit = top_k if top_k is not None else settings.history_message_top_k

        query_vector = self._embed_query(query)
        hits = self._vector_search(
            query_vector=query_vector,
            limit=limit,
            min_score=_MESSAGE_RECALL_MIN_SCORE,
            source_scope=session_id,
            vector_type=VectorType.MESSAGE,
        )
        # vector_store 已按分数降序并按 limit 截断；此处再截断以兜底内存替身忽略 limit 的情形。
        return hits[:limit]

    # ------------------------------------------------------------------ #
    # 内部工具                                                            #
    # ------------------------------------------------------------------ #
    def _embed_query(self, query: str) -> list[float]:
        """校验查询非空并生成查询向量。

        Raises:
            QueryEmptyError: 查询为 ``None`` 或去除首尾空白后长度为 0（Req 8.7）。
            QueryEmbeddingFailedError: 嵌入调用失败或返回空结果（Req 8.8）。
        """
        if query is None or not query.strip():
            raise QueryEmptyError("用户查询为空或去除首尾空白后长度为 0")
        try:
            vectors = self._embedding_client.embed([query])
        except QueryEmbeddingFailedError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一封装为查询向量生成失败（Req 8.8）
            raise QueryEmbeddingFailedError("查询向量生成失败") from exc
        if not vectors:
            raise QueryEmbeddingFailedError("查询向量生成失败：Embedding 返回空结果")
        return list(vectors[0])

    def _vector_search(
        self,
        *,
        query_vector: list[float],
        limit: int,
        min_score: float,
        source_scope: str | None,
        vector_type: VectorType,
    ) -> list[SearchHit]:
        """调用 vector_store 检索，返回 ≥ min_score、按分数降序的命中。"""
        return self._vector_store.search(
            query_vector,
            top_k=limit,
            min_score=min_score,
            source_scope=source_scope,
            vector_type=vector_type,
        )

    @staticmethod
    def _cap_with_ties(hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """截断到 Top-K，但在边界存在分数并列时全部纳入（可超 Top-K，Req 8.3）。

        ``hits`` 须已按分数降序排列。不存在边界并列时返回数量恰为 ``min(len, top_k)``，
        从而保证"无边界并列时数量不超过 Top-K"（Property 6）。
        """
        if top_k <= 0:
            return []
        if len(hits) <= top_k:
            return list(hits)
        boundary_score = hits[top_k - 1].score
        result = list(hits[:top_k])
        for hit in hits[top_k:]:
            if hit.score == boundary_score:
                result.append(hit)
            else:
                break
        return result

    def _hybrid_merge(
        self,
        *,
        query: str,
        query_vector: list[float],
        top_k: int,
        min_score: float,
        source_scope: str | None,
        vector_type: VectorType,
    ) -> list[SearchHit]:
        """合并向量检索与关键词检索结果：按 id 去重、融合分数降序、截断到 Top-K（Req 8.4）。

        融合分数取同一 ``id`` 在两路结果中的较大分数（``max`` 融合）；按融合分数降序排列；
        严格截断到 ``top_k``（混合检索不适用 Top-K 边界并列例外）。
        """
        vector_hits = self._vector_search(
            query_vector=query_vector,
            limit=top_k,
            min_score=min_score,
            source_scope=source_scope,
            vector_type=vector_type,
        )
        keyword_hits = self._keyword_search(query, top_k)

        best_by_id: dict[str, SearchHit] = {}
        for hit in (*vector_hits, *keyword_hits):
            existing = best_by_id.get(hit.id)
            # 融合分数取较大值：保留分数更高的代表命中，从而无重复 id。
            if existing is None or hit.score > existing.score:
                best_by_id[hit.id] = hit

        merged = sorted(
            best_by_id.values(), key=lambda hit: hit.score, reverse=True
        )
        return merged[:top_k]

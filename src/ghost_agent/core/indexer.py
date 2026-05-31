"""核心组件：Indexer（嵌入与索引，Req 7, 19）。

Indexer 负责把上游 Transformer 产出的 :class:`Chunk` 集合转换为向量并写入
:class:`Vector_Database`（Milvus），同时为对话轮次的消息生成向量并入库以支撑
消息向量召回（Req 19.1）。

对外提供两个能力：

* :meth:`Indexer.index` —— 为每个 Chunk 调用 Embedding_Model 生成向量（Req 7.1）
  并写入 vector_store（Req 7.2）。嵌入失败按配置最大重试次数（0–5，默认 3）重试
  （Req 7.3）；达上限仍失败或写入失败则记录失败、计入失败数并继续处理其余 Chunk
  （Req 7.4，失败隔离）。完成后返回成功数与失败数，且二者之和恒等于接收到的 Chunk
  总数（Req 7.5）；空集合不调用 Embedding_Model 并返回 (0, 0)（Req 7.6）。
* :meth:`Indexer.index_message` —— 为本轮用户消息与应答生成向量并连同 session_id
  写入（vector_type=MESSAGE，Req 19.1）。任何失败仅记录、绝不抛出，以保证不中断
  对话流程（Req 19.2）。

设计要点：
- **失败隔离与计数守恒**：``index`` 逐个处理 Chunk，对单个 Chunk 的 embed+write
  以 try/except 包裹，每个 Chunk 恰好增加一次成功或失败计数，从而由构造保证
  ``success_count + failure_count == len(chunks)``（Property 4 / Req 7.5）。
- **重试次数上界**：:meth:`_embed_with_retry` 以 ``range(max_retries + 1)`` 控制
  总尝试次数，使"重试次数"（不含首次尝试）严格不超过 ``max_retries``
  （Property 5 / Req 7.3）。``max_retries`` 在构造期被防御性钳制到 [0, 5]。
- **可测试性 seam**：embedding_client 与 vector_store 均可经构造函数注入；测试
  以内存替身隔离，无需任何网络或真实 Milvus。
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ghost_agent.clients.doubao_client import DoubaoEmbeddingClient
from ghost_agent.config import get_settings
from ghost_agent.models.chunk import Chunk
from ghost_agent.models.errors import QueryEmbeddingFailedError
from ghost_agent.models.memory import Role
from ghost_agent.models.vector_record import VectorRecord, VectorType
from ghost_agent.vector_db.vector_store import VectorStore

logger = logging.getLogger(__name__)

# 嵌入最大重试次数的取值范围（Req 7.3）。构造期对越界配置做防御性钳制。
_MIN_RETRIES = 0
_MAX_RETRIES = 5

__all__ = ["Indexer", "IndexResult", "IndexFailure"]


class IndexFailure(BaseModel):
    """单个 Chunk 的索引失败信息（Req 7.4）。"""

    chunk_id: str = Field(..., description="失败 Chunk 的唯一标识。")
    reason: str = Field(..., description="失败原因描述（嵌入失败或写入失败）。")


class IndexResult(BaseModel):
    """``index`` 的返回结果（Req 7.5）。

    不变量：``success_count + failure_count`` 恒等于接收到的 Chunk 总数；
    ``len(failures) == failure_count``。
    """

    success_count: int = Field(..., ge=0, description="成功写入 Vector_Database 的 Chunk 数量。")
    failure_count: int = Field(..., ge=0, description="嵌入或写入失败的 Chunk 数量。")
    failures: list[IndexFailure] = Field(
        default_factory=list,
        description="失败 Chunk 的明细列表。",
    )


class Indexer:
    """索引器：为 Chunk / 消息生成嵌入向量并写入 Vector_Database。

    Args:
        embedding_client: Doubao Embedding 客户端；为 ``None`` 时惰性构造默认
            :class:`DoubaoEmbeddingClient`（构造期不触网）。
        vector_store: 向量存储；为 ``None`` 时惰性构造默认 :class:`VectorStore`
            （构造期不连接 Milvus）。
        max_retries: 嵌入失败最大重试次数；为 ``None`` 时取
            ``settings.embedding_max_retries``。无论来源如何，均被钳制到 [0, 5]
            （Req 7.3）。
    """

    def __init__(
        self,
        *,
        embedding_client: DoubaoEmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self._embedding_client = (
            embedding_client if embedding_client is not None else DoubaoEmbeddingClient()
        )
        self._vector_store = vector_store if vector_store is not None else VectorStore()
        raw_retries = (
            max_retries if max_retries is not None else settings.embedding_max_retries
        )
        # 防御性钳制：即便配置层失效，重试次数也不会超出 [0, 5]（Req 7.3）。
        self._max_retries: int = max(_MIN_RETRIES, min(_MAX_RETRIES, int(raw_retries)))

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def max_retries(self) -> int:
        """单个 Chunk 嵌入失败的最大重试次数（已钳制到 [0, 5]）。"""
        return self._max_retries

    # ------------------------------------------------------------------ #
    # 文档分片索引                                                        #
    # ------------------------------------------------------------------ #
    def index(self, chunks: list[Chunk]) -> IndexResult:
        """为 Chunk 集合生成向量并写入 Vector_Database。

        逐个独立处理每个 Chunk（失败隔离，Req 7.4）：嵌入成功则构造
        :class:`VectorRecord` 写入；嵌入达上限仍失败或写入失败则记录失败、计入
        失败数并继续其余 Chunk。每个 Chunk 恰好触发一次成功或失败计数，从而保证
        ``success_count + failure_count == len(chunks)``（Req 7.5）。

        Args:
            chunks: 待索引的 Chunk 集合；为空（或 ``None``）时不调用 Embedding_Model
                并返回 ``(0, 0)``（Req 7.6）。

        Returns:
            :class:`IndexResult`，含成功数、失败数与失败明细。
        """
        if not chunks:
            # Req 7.6：空集合不调用 Embedding_Model，直接返回 (0, 0)。
            return IndexResult(success_count=0, failure_count=0)

        success_count = 0
        failures: list[IndexFailure] = []

        for chunk in chunks:
            try:
                vector = self._embed_with_retry(chunk.text)
                record = VectorRecord(
                    vector=vector,
                    text=chunk.text,
                    source_id=chunk.source_file_id,
                    vector_type=VectorType.DOC_CHUNK,
                    metadata={
                        "seq": chunk.seq,
                        "start_offset": chunk.start_offset,
                        "end_offset": chunk.end_offset,
                        "chunk_id": chunk.chunk_id,
                        "parent_chunk_id": chunk.parent_chunk_id,
                    },
                )
                self._vector_store.write(record)
            except Exception as exc:  # noqa: BLE001 - 失败隔离：记录并继续（Req 7.4）
                logger.warning(
                    "索引 Chunk 失败（chunk_id=%s）：%s", chunk.chunk_id, exc
                )
                failures.append(IndexFailure(chunk_id=chunk.chunk_id, reason=str(exc)))
            else:
                success_count += 1

        return IndexResult(
            success_count=success_count,
            failure_count=len(failures),
            failures=failures,
        )

    # ------------------------------------------------------------------ #
    # 消息向量索引（Req 19.1, 19.2）                                       #
    # ------------------------------------------------------------------ #
    def index_message(self, session_id: str, user_msg: str, answer: str) -> None:
        """为本轮用户消息与应答生成向量并连同 session_id 写入（vector_type=MESSAGE）。

        分别为用户消息与应答各写入一条 MESSAGE 向量记录（source_id=session_id，
        Req 19.1）。任一嵌入或写入失败仅记录日志，绝不向上抛出，以保证不中断对话
        流程（Req 19.2）。

        Args:
            session_id: 本轮消息所属会话标识。
            user_msg: 本轮用户消息文本。
            answer: 本轮应答文本。

        Returns:
            ``None``（无论成功或失败均不抛出）。
        """
        self._index_single_message(session_id, user_msg, Role.USER)
        self._index_single_message(session_id, answer, Role.ASSISTANT)

    def _index_single_message(self, session_id: str, content: str, role: Role) -> None:
        """为单条消息生成向量并写入；失败仅记录不抛出（Req 19.2）。"""
        try:
            vector = self._embed_with_retry(content)
            record = VectorRecord(
                vector=vector,
                text=content,
                source_id=session_id,
                vector_type=VectorType.MESSAGE,
                metadata={"role": role.value, "session_id": session_id},
            )
            self._vector_store.write(record)
        except Exception as exc:  # noqa: BLE001 - 仅记录，不中断对话流程（Req 19.2）
            logger.warning(
                "消息向量入库失败（session_id=%s, role=%s）：%s",
                session_id,
                role.value,
                exc,
            )

    # ------------------------------------------------------------------ #
    # 内部工具：带重试的嵌入（Req 7.3 / Property 5）                       #
    # ------------------------------------------------------------------ #
    def _embed_with_retry(self, text: str) -> list[float]:
        """对单段文本生成嵌入向量，失败时按 ``max_retries`` 重试。

        总尝试次数 = ``max_retries + 1``（1 次首次尝试 + 至多 ``max_retries`` 次重试），
        因此"重试次数"严格不超过配置的最大重试次数（Req 7.3 / Property 5）。

        Args:
            text: 待嵌入文本。

        Returns:
            长度等于 Embedding_Model 输出维度的向量。

        Raises:
            QueryEmbeddingFailedError: 达到最大重试次数后仍失败时抛出，保留最后一次
                的原始异常为 ``__cause__``。
        """
        last_exc: Exception | None = None
        for _attempt in range(self._max_retries + 1):
            try:
                vectors = self._embedding_client.embed([text])
            except Exception as exc:  # noqa: BLE001 - 统一捕获以计入一次失败尝试
                last_exc = exc
                continue
            if not vectors:
                last_exc = QueryEmbeddingFailedError("Embedding 返回空结果")
                continue
            return list(vectors[0])

        raise QueryEmbeddingFailedError(
            f"嵌入在 {self._max_retries} 次重试后仍失败"
        ) from last_exc

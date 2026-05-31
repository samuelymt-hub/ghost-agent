"""Knowledge_Base_Agent（知识库 Agent，RAG，Req 5–9, 22, 3.6, 3.7）。

本模块实现知识库 Agent，编排核心组件完成两条主管线与两类知识库运维能力：

* **入库管线（ingest，Req 5.1/6.5/7.1/3.6/3.7/22.1）**：编排
  ``Loader → Transformer → Indexer``，按阶段流转 :class:`IngestTask` 状态；成功置
  ``COMPLETED`` 并提供 ``chunk_count``，失败置 ``FAILED`` 并提供 ``failure_reason``；
  每个 Chunk 经 Transformer 从 ``ParseResult.meta`` 继承来源文件标识（Req 22.1）。
* **检索增强生成（answer，Req 9.1–9.6）**：编排
  ``Retriever → Prompt_Module → Chat_Model``；非空召回时要求 Chat_Model 仅依据召回
  Chunk 作答并附**仅来自召回集合**的引用来源文件标识列表（Req 9.4 / Property 11）；
  召回为空时明确告知未检索到且不臆造（Req 9.3）；生成错误/超时直接向上抛出，绝不返回
  部分或臆造答案（Req 9.5/9.6）。
* **同步（sync，Req 22.1/22.2/22.4/22.5）**：执行 ``加载→分片→嵌入→写入`` 完整流程。
  为满足 **Property 29（失败不破坏既有状态）**，采用 **"先全量嵌入、后原子替换"
  （embed-all-then-swap）** 的顺序：先把全部新版 Chunk 嵌入为内存中的
  :class:`VectorRecord`（此阶段不触碰 Vector_Database），**仅当**全部嵌入成功后，才
  ``delete_by_source`` 删除旧版 Chunk 并写入新版。如此一来，加载/分片/嵌入任一阶段失败
  时尚未发生任何删除或写入，Vector_Database 中该来源的既有 Chunk 与失败前完全一致
  （Req 22.5）。同源再次同步时，先删后写保证用新版完全替换、不残留旧版（Req 22.2 /
  Property 27）。
* **移除（remove，Req 22.3/22.6）**：按来源文件标识删除其全部 Chunk 并返回删除数量
  （Req 22.3 / Property 28）；移除不存在的来源时不删除任何 Chunk 并返回不存在提示
  （Req 22.6）。

设计要点：
- **协作者注入（Dependency Injection）**：构造函数注入全部协作者
  （loader / transformer / indexer / retriever / prompt_module / chat_model /
  vector_store / embedding_client），缺省时惰性构造默认实现（构造期均不触网/不连库），
  从而支持完全离线、确定性的属性化测试。
- **sync 直接持有 embedding_client 与 vector_store**：入库（ingest）复用 Indexer 的
  逐 Chunk 失败隔离语义；而同步（sync）需要"全量嵌入成功后再原子替换"的更细粒度控制，
  故 KBA 直接持有 ``embedding_client`` 与 ``vector_store`` 自行实现 swap 顺序不变量。
- **cited_sources 严格派生自召回集合（Property 11）**：引用来源仅由召回命中的
  ``source_id`` 去重得到，绝不引入召回集合以外的来源。
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ghost_agent.clients.doubao_client import DoubaoEmbeddingClient
from ghost_agent.core.chat_model import ChatMessage, ChatModel
from ghost_agent.core.indexer import Indexer
from ghost_agent.core.loader import Loader
from ghost_agent.core.prompt_module import PromptModule
from ghost_agent.core.retriever import Retriever, RetrieveOptions
from ghost_agent.core.transformer import Transformer
from ghost_agent.models.chunk import Chunk
from ghost_agent.models.errors import (
    EmptyContentError,
    ParseFailedError,
    ParseTimeoutError,
    SplitFailedError,
)
from ghost_agent.models.ingest_task import IngestTask, IngestTaskStatus
from ghost_agent.models.vector_record import VectorRecord, VectorType
from ghost_agent.vector_db.vector_store import VectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "KnowledgeBaseAgent",
    "IngestResult",
    "AnswerResult",
    "SyncResult",
    "RemoveResult",
]

#: 召回为空时返回给用户的"未检索到"提示语（Req 9.3，不臆造知识库以外内容）。
_NOT_FOUND_MESSAGE = "未在知识库中检索到相关内容，无法据此作答。"

#: 同步流程失败阶段标识。
_STAGE_LOAD = "load"
_STAGE_SPLIT = "split"
_STAGE_EMBED_WRITE = "embed_write"


# --------------------------------------------------------------------------- #
# 返回结构                                                                      #
# --------------------------------------------------------------------------- #
class IngestResult(BaseModel):
    """入库结果（镜像 :class:`IngestTask` 的终态，Req 3.6/3.7）。"""

    task_id: str = Field(..., description="入库任务唯一标识。")
    status: IngestTaskStatus = Field(..., description="入库任务终态。")
    chunk_count: int | None = Field(
        default=None,
        ge=0,
        description="COMPLETED 时提供的成功写入分片数（Req 3.6）。",
    )
    failure_reason: str | None = Field(
        default=None,
        description="FAILED 时提供的失败原因（Req 3.7）。",
    )


class AnswerResult(BaseModel):
    """检索增强生成结果（Req 9.4）。"""

    answer: str = Field(..., description="最终答案文本。")
    cited_sources: list[str] = Field(
        default_factory=list,
        description="所引用 Chunk 的来源文件标识列表（仅来自召回集合，Property 11）。",
    )


class SyncResult(BaseModel):
    """知识库同步结果（Req 22.4/22.5）。"""

    source_file_id: str = Field(..., description="被同步的来源文件标识。")
    status: str = Field(..., description='同步状态："COMPLETED" 或 "FAILED"。')
    success_count: int = Field(default=0, ge=0, description="成功写入的 Chunk 数量（Req 22.4）。")
    failure_count: int = Field(default=0, ge=0, description="失败的 Chunk 数量（Req 22.4）。")
    failed_stage: str | None = Field(
        default=None,
        description='失败阶段："load" | "split" | "embed_write"（Req 22.5）。',
    )
    failure_reason: str | None = Field(
        default=None, description="失败原因（Req 22.5）。"
    )


class RemoveResult(BaseModel):
    """知识库来源移除结果（Req 22.3/22.6）。"""

    source_file_id: str = Field(..., description="被移除的来源文件标识。")
    deleted_count: int = Field(..., ge=0, description="被删除的 Chunk 数量（Req 22.3）。")
    found: bool = Field(
        ..., description="该来源是否存在；为 False 表示未找到、未删除任何 Chunk（Req 22.6）。"
    )


# --------------------------------------------------------------------------- #
# Knowledge_Base_Agent                                                          #
# --------------------------------------------------------------------------- #
class KnowledgeBaseAgent:
    """知识库 Agent：编排入库、检索增强生成与知识库同步/移除。

    Args:
        loader: 文档加载器；为 ``None`` 时构造默认 :class:`Loader`。
        transformer: 分片器；为 ``None`` 时构造默认 :class:`Transformer`。
        indexer: 索引器（用于 ingest 的逐 Chunk 失败隔离索引）；为 ``None`` 时构造
            默认 :class:`Indexer`，复用本 KBA 的 ``embedding_client`` 与 ``vector_store``。
        retriever: 检索器；为 ``None`` 时构造默认 :class:`Retriever`，复用本 KBA 的
            ``embedding_client`` 与 ``vector_store``。
        prompt_module: 提示词模块；为 ``None`` 时构造默认 :class:`PromptModule`。
        chat_model: 对话模型；为 ``None`` 时构造默认 :class:`ChatModel`（惰性连接）。
        vector_store: 向量存储；为 ``None`` 时构造默认 :class:`VectorStore`（惰性连接）。
        embedding_client: 嵌入客户端（用于 sync 的全量嵌入）；为 ``None`` 时构造默认
            :class:`DoubaoEmbeddingClient`（惰性连接）。
        rag_template_name: RAG 增强提示词使用的基础模板名（默认 ``"rag_answer"``）。
    """

    def __init__(
        self,
        *,
        loader: Loader | None = None,
        transformer: Transformer | None = None,
        indexer: Indexer | None = None,
        retriever: Retriever | None = None,
        prompt_module: PromptModule | None = None,
        chat_model: ChatModel | None = None,
        vector_store: VectorStore | None = None,
        embedding_client: DoubaoEmbeddingClient | None = None,
        rag_template_name: str = "rag_answer",
    ) -> None:
        # 先构造共享基础设施（向量库 / 嵌入客户端），供 Indexer / Retriever 复用。
        self._vector_store = vector_store if vector_store is not None else VectorStore()
        self._embedding_client = (
            embedding_client if embedding_client is not None else DoubaoEmbeddingClient()
        )
        self._loader = loader if loader is not None else Loader()
        self._transformer = transformer if transformer is not None else Transformer()
        self._indexer = (
            indexer
            if indexer is not None
            else Indexer(
                embedding_client=self._embedding_client,
                vector_store=self._vector_store,
            )
        )
        self._retriever = (
            retriever
            if retriever is not None
            else Retriever(
                embedding_client=self._embedding_client,
                vector_store=self._vector_store,
            )
        )
        self._prompt_module = (
            prompt_module if prompt_module is not None else PromptModule()
        )
        self._chat_model = chat_model if chat_model is not None else ChatModel()
        self._rag_template_name = rag_template_name

    # ------------------------------------------------------------------ #
    # 入库管线（Req 5.1, 6.5, 7.1, 3.6, 3.7, 22.1）                         #
    # ------------------------------------------------------------------ #
    def ingest(
        self,
        *,
        content: bytes | str,
        file_name: str,
        file_format: str,
        source_file_id: str | None = None,
    ) -> IngestResult:
        """执行文档入库：Loader → Transformer → Indexer，并流转 IngestTask 状态。

        各阶段失败按 Req 3.7 / 5.3 / 5.5 / 5.6 / 6.6 / 6.7 将任务置 ``FAILED`` 并附
        失败原因；全部成功（或部分 Chunk 成功）按 Req 3.6 置 ``COMPLETED`` 并提供
        ``chunk_count``（成功写入的 Chunk 数）。每个 Chunk 经 Transformer 从
        ``ParseResult.meta`` 继承来源文件标识（Req 22.1）。

        Args:
            content: 原始文件内容（bytes 或 str）。
            file_name: 原始文件名。
            file_format: 文件格式/扩展名。
            source_file_id: 可选来源文件标识；为 ``None`` 时由 Loader 生成 UUID。

        Returns:
            :class:`IngestResult`，镜像入库任务终态。
        """
        task = IngestTask(file_name=file_name, file_format=file_format)
        task.status = IngestTaskStatus.RUNNING

        # 阶段一：加载解析。Loader 在失败时已将 task 标记为 FAILED 并抛出。
        try:
            parse_result = self._loader.parse(
                content=content,
                file_name=file_name,
                file_format=file_format,
                source_file_id=source_file_id,
                ingest_task=task,
            )
        except (ParseFailedError, ParseTimeoutError, EmptyContentError) as exc:
            return self._failed_ingest(task, exc.message)

        # 阶段二：分片。
        try:
            chunks = self._transformer.split(parse_result)
        except SplitFailedError as exc:
            return self._failed_ingest(task, exc.message)

        if not chunks:
            # Req 6.6：内容为空/未产生任何分片，视为入库失败。
            return self._failed_ingest(task, "未产生任何分片")

        # 阶段三：索引（逐 Chunk 失败隔离，Req 7.4）。
        index_result = self._indexer.index(chunks)
        if index_result.success_count == 0:
            reason = (
                f"全部 {len(chunks)} 个分片嵌入或写入失败，未成功写入任何 Chunk"
            )
            return self._failed_ingest(task, reason)

        # Req 3.6：COMPLETED 必须提供 chunk_count（先写 chunk_count 再置状态）。
        task.chunk_count = index_result.success_count
        task.status = IngestTaskStatus.COMPLETED
        return IngestResult(
            task_id=task.task_id,
            status=IngestTaskStatus.COMPLETED,
            chunk_count=index_result.success_count,
        )

    # ------------------------------------------------------------------ #
    # 检索增强生成（Req 9.1–9.6）                                           #
    # ------------------------------------------------------------------ #
    def answer(self, query: str) -> AnswerResult:
        """基于召回的知识片段生成答案（RAG）。

        编排 ``Retriever → Prompt_Module → Chat_Model``：

        * 召回为空（Req 9.3）：返回明确的"未检索到"提示，``cited_sources`` 为空，
          且**不调用** Chat_Model，绝不臆造知识库以外内容。
        * 召回非空（Req 9.1/9.2/9.4）：构造增强提示词并要求 Chat_Model 仅依据召回
          Chunk 作答；引用来源严格取召回命中 ``source_id`` 的去重集合（Property 11）。
        * 生成错误/超时（Req 9.5/9.6）：:class:`GenerationError` /
          :class:`GenerationTimeoutError` 直接向上抛出，不返回部分或臆造答案。

        Args:
            query: 用户查询文本。

        Returns:
            :class:`AnswerResult`，含答案与引用来源文件标识列表。

        Raises:
            QueryEmptyError: 查询为空或去空白后为空（来自 Retriever，Req 8.7）。
            QueryEmbeddingFailedError: 查询向量生成失败（来自 Retriever，Req 8.8）。
            GenerationError / GenerationTimeoutError: 生成失败/超时（Req 9.5/9.6）。
        """
        recalled = self._retriever.retrieve(query, RetrieveOptions())

        if not recalled:
            # Req 9.3：明确告知未检索到，不调用 Chat_Model、不臆造。
            return AnswerResult(answer=_NOT_FOUND_MESSAGE, cited_sources=[])

        prompt = self._prompt_module.build_rag_prompt(
            query, recalled, template_name=self._rag_template_name
        )
        # Req 9.5/9.6：生成错误/超时由 Chat_Model 抛出，直接透传，不做部分返回。
        completion = self._chat_model.generate(
            [ChatMessage(role="user", content=prompt.text)]
        )

        # Property 11：cited_sources 严格派生自召回命中的来源标识，绝不引入召回以外来源。
        cited_sources = sorted({hit.source_id for hit in recalled})
        return AnswerResult(answer=completion.content, cited_sources=cited_sources)

    # ------------------------------------------------------------------ #
    # 知识库同步（Req 22.1, 22.2, 22.4, 22.5）                              #
    # ------------------------------------------------------------------ #
    def sync(
        self,
        *,
        content: bytes | str,
        file_name: str,
        file_format: str,
        source_file_id: str,
    ) -> SyncResult:
        """同步单个来源文件至 Vector_Database（embed-all-then-swap）。

        采用"先全量嵌入、后原子替换"的顺序以满足 Property 29：

        1. **加载**（失败 → ``failed_stage="load"``）。
        2. **分片**（失败或未产生分片 → ``failed_stage="split"``）。
        3. **全量嵌入**到内存 :class:`VectorRecord`（**不触碰向量库**；任一嵌入失败 →
           ``failed_stage="embed_write"``，既有 Chunk 保持不变）。
        4. **原子替换**：``delete_by_source`` 删除旧版后写入新版（Req 22.2，不残留旧版）。

        由于删除/写入仅在前三阶段全部成功后才发生，加载/分片/嵌入任一阶段失败时
        Vector_Database 中该来源的既有 Chunk 与失败前完全一致（Req 22.5 / Property 29）。

        Args:
            content: 原始文件内容（bytes 或 str）。
            file_name: 原始文件名。
            file_format: 文件格式/扩展名。
            source_file_id: 来源文件标识（替换/写入的归属键）。

        Returns:
            :class:`SyncResult`，含同步状态与成功/失败 Chunk 数（Req 22.4）。
        """
        # 阶段一：加载。
        try:
            parse_result = self._loader.parse(
                content=content,
                file_name=file_name,
                file_format=file_format,
                source_file_id=source_file_id,
            )
        except (ParseFailedError, ParseTimeoutError, EmptyContentError) as exc:
            return self._failed_sync(source_file_id, _STAGE_LOAD, exc.message)

        # 阶段二：分片。
        try:
            chunks = self._transformer.split(parse_result)
        except SplitFailedError as exc:
            return self._failed_sync(source_file_id, _STAGE_SPLIT, exc.message)

        if not chunks:
            return self._failed_sync(source_file_id, _STAGE_SPLIT, "未产生任何分片")

        # 阶段三：全量嵌入到内存（不触碰向量库），任一失败即终止且不破坏既有状态。
        try:
            records = self._embed_chunks(chunks)
        except Exception as exc:  # noqa: BLE001 - 嵌入阶段任意异常统一归为 embed_write
            logger.warning(
                "同步嵌入阶段失败（source_file_id=%s）：%s", source_file_id, exc
            )
            return self._failed_sync(source_file_id, _STAGE_EMBED_WRITE, str(exc))

        # 阶段四：原子替换（先删后写）。仅在全量嵌入成功后执行，保证既有状态不被破坏。
        try:
            self._vector_store.delete_by_source(source_file_id)
            written = self._write_records(records)
        except Exception as exc:  # noqa: BLE001 - 写入阶段异常归为 embed_write
            logger.warning(
                "同步写入阶段失败（source_file_id=%s）：%s", source_file_id, exc
            )
            return self._failed_sync(source_file_id, _STAGE_EMBED_WRITE, str(exc))

        return SyncResult(
            source_file_id=source_file_id,
            status="COMPLETED",
            success_count=written,
            failure_count=0,
        )

    # ------------------------------------------------------------------ #
    # 知识库移除（Req 22.3, 22.6）                                          #
    # ------------------------------------------------------------------ #
    def remove(self, source_file_id: str) -> RemoveResult:
        """移除某来源文件的全部 Chunk。

        删除该来源在 Vector_Database 中的全部 Chunk 并返回删除数量（Req 22.3）；
        若该来源不存在（删除数量为 0），返回 ``found=False`` 作为不存在提示
        （Req 22.6），且不删除任何 Chunk。

        Args:
            source_file_id: 待移除的来源文件标识。

        Returns:
            :class:`RemoveResult`，含删除数量与是否找到该来源。
        """
        deleted = self._vector_store.delete_by_source(source_file_id)
        return RemoveResult(
            source_file_id=source_file_id,
            deleted_count=deleted,
            found=deleted > 0,
        )

    # ------------------------------------------------------------------ #
    # 内部工具                                                            #
    # ------------------------------------------------------------------ #
    def _embed_chunks(self, chunks: list[Chunk]) -> list[VectorRecord]:
        """将 Chunk 集合全量嵌入为 :class:`VectorRecord`（不写入向量库）。

        采用整体（all-or-nothing）语义：任一文本嵌入失败即抛出，使同步在写入前终止，
        从而保证既有 Chunk 不被破坏（Property 29）。

        Raises:
            QueryEmbeddingFailedError: 嵌入调用失败（来自 embedding_client）。
            ValueError: 返回向量数量与 Chunk 数量不一致。
        """
        vectors = self._embedding_client.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(
                f"嵌入返回向量数量({len(vectors)})与 Chunk 数量({len(chunks)})不一致"
            )
        records: list[VectorRecord] = []
        for chunk, vector in zip(chunks, vectors):
            records.append(
                VectorRecord(
                    vector=list(vector),
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
            )
        return records

    def _write_records(self, records: list[VectorRecord]) -> int:
        """将内存中的 :class:`VectorRecord` 批量写入向量库，返回写入数量。"""
        if not records:
            return 0
        write_many = getattr(self._vector_store, "write_many", None)
        if callable(write_many):
            return write_many(records)
        for record in records:
            self._vector_store.write(record)
        return len(records)

    @staticmethod
    def _failed_ingest(task: IngestTask, reason: str) -> IngestResult:
        """将入库任务置为 FAILED（先写 failure_reason 再置 status）并返回结果。"""
        if task.status is not IngestTaskStatus.FAILED:
            task.failure_reason = reason
            task.status = IngestTaskStatus.FAILED
        return IngestResult(
            task_id=task.task_id,
            status=IngestTaskStatus.FAILED,
            failure_reason=task.failure_reason or reason,
        )

    @staticmethod
    def _failed_sync(source_file_id: str, stage: str, reason: str) -> SyncResult:
        """构造一个失败的同步结果（既有 Chunk 保持不变）。"""
        return SyncResult(
            source_file_id=source_file_id,
            status="FAILED",
            success_count=0,
            failure_count=0,
            failed_stage=stage,
            failure_reason=reason,
        )

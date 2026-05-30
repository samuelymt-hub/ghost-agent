"""Vector_Database 层：基于 Milvus 的统一向量存储（vector_store）。

本模块封装 :class:`MilvusClientWrapper`，在单一 collection 中统一存储文档分片向量
（``DOC_CHUNK``）与对话消息向量（``MESSAGE``），通过 ``vector_type`` 标量字段区分
（Req 21.1）。对上层提供写入、检索、按来源删除三类能力：

* :meth:`VectorStore.write` —— 写入前校验 ``len(vector) == dim``，不一致拒绝并抛
  :class:`DimensionMismatchError`（Req 21.4）；通过校验则持久化原始文本、来源标识
  ``source_id``、向量类型 ``vector_type`` 与 ``metadata``（Req 21.3）。
* :meth:`VectorStore.search` —— 向量相似度检索，按分数降序返回 ≥ ``min_score``、
  数量 ≤ ``top_k`` 的命中，支持按来源与向量类型过滤（Req 19.1 消息召回的底座）。
* :meth:`VectorStore.delete_by_source` —— 按来源标识删除并返回删除数量（支撑 Req 22.3）。

设计要点：
- **惰性连接 / 惰性建表（Lazy）**：构造函数不连接 Milvus、不创建 collection、不导入
  ``pymilvus``。真正的连接在首次访问底层 client 时建立；collection 在首次 IO 时惰性
  创建并通过 :attr:`_collection_ready` 缓存，仅创建一次。这使模块可在无运行中 Milvus
  的环境下被 ``import`` 与单元测试。
- **统一异常封装**：除写入前的维度校验（:class:`DimensionMismatchError`，在任何 DB
  调用之前抛出，保证"未写入数据不丢失"）外，连接/操作过程中的任何异常均被包装为
  :class:`VectorDatabaseUnavailableError`（Req 21.5），并保留原始异常作为 ``__cause__``。
- **可测试性 seam**：所有底层访问都经由 ``self._milvus.get_client()``。测试可构造一个
  :class:`MilvusClientWrapper` 并将其 ``_build_client`` 替换为返回内存版假客户端，从而
  在离线环境下确定性地验证写入/检索/删除/阈值/过滤等行为。
- **小而明确的 client 调用面**：production 仅调用底层 client 的
  ``has_collection`` / ``create_schema`` / ``prepare_index_params`` / ``create_collection``
  / ``insert`` / ``search`` / ``delete`` 七个方法（以及 schema/index 对象的 ``add_field`` /
  ``add_index``），便于测试用假客户端精确复刻。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ghost_agent.clients.milvus_client import MilvusClientWrapper
from ghost_agent.config import get_settings
from ghost_agent.models.errors import (
    DimensionMismatchError,
    VectorDatabaseUnavailableError,
)
from ghost_agent.models.vector_record import VectorRecord, VectorType

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    pass

__all__ = ["VectorStore", "SearchHit"]


# VARCHAR 字段长度上限（仅对真实 Milvus 生效；内存假客户端忽略）。
_ID_MAX_LENGTH = 128
_SOURCE_ID_MAX_LENGTH = 512
_VECTOR_TYPE_MAX_LENGTH = 32
_TEXT_MAX_LENGTH = 65535

# search 需要回填的字段（不含 ``vector``，避免无谓地回传大向量）。
_OUTPUT_FIELDS = ["id", "text", "source_id", "vector_type", "metadata"]


class SearchHit(BaseModel):
    """单条相似度检索命中结果。

    承载回填检索结果所需的全部字段：主键、原始文本、来源标识、向量类型、相似度
    分数与附加元数据。
    """

    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    id: str = Field(..., description="命中记录主键。")
    text: str = Field(..., description="原始文本 (Req 21.3)。")
    source_id: str = Field(..., description="来源标识 (Req 21.3)。")
    vector_type: VectorType = Field(..., description="向量类型 (Req 21.3)。")
    score: float = Field(..., description="相似度分数（COSINE，越大越相似）。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据。")


class VectorStore:
    """Milvus 统一向量存储（惰性连接 / 惰性建表）。

    Args:
        milvus: Milvus 连接封装；为 ``None`` 时使用默认 :class:`MilvusClientWrapper`。
        dim: 向量维度；为 ``None`` 时取 ``settings.embedding_dim``（Req 21.2/21.4）。
        collection_name: 统一 collection 名称；为 ``None`` 时取
            ``settings.milvus_collection_name``（Req 21.1）。
    """

    def __init__(
        self,
        *,
        milvus: MilvusClientWrapper | None = None,
        dim: int | None = None,
        collection_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self._milvus: MilvusClientWrapper = (
            milvus if milvus is not None else MilvusClientWrapper()
        )
        self._dim: int = dim if dim is not None else settings.embedding_dim
        self._collection_name: str = (
            collection_name
            if collection_name is not None
            else settings.milvus_collection_name
        )
        # 惰性建表标志；首次成功 ensure_collection 后置 True，避免重复建表。
        self._collection_ready: bool = False

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        """向量维度（Req 21.2/21.4）。"""
        return self._dim

    @property
    def collection_name(self) -> str:
        """统一向量 collection 名称（Req 21.1）。"""
        return self._collection_name

    # ------------------------------------------------------------------ #
    # collection 生命周期                                                 #
    # ------------------------------------------------------------------ #
    def ensure_collection(self) -> None:
        """惰性创建统一 collection（仅在不存在时创建，且整个进程仅创建一次）。

        schema 字段：``id``(VARCHAR 主键)、``vector``(FLOAT_VECTOR, dim=:attr:`dim`)、
        ``text``(VARCHAR)、``source_id``(VARCHAR)、``vector_type``(VARCHAR)、
        ``metadata``(JSON)；并为 ``vector`` 建立 COSINE 度量索引，使相似度分数有意义。

        Raises:
            VectorDatabaseUnavailableError: 连接或建表失败时抛出（Req 21.5），
                保留原始异常为 ``__cause__``。
        """
        if self._collection_ready:
            return
        try:
            client = self._milvus.get_client()
            if not client.has_collection(self._collection_name):
                self._create_collection(client)
        except VectorDatabaseUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一封装为向量库不可用
            raise VectorDatabaseUnavailableError(
                f"创建/检查 Milvus collection 失败（collection={self._collection_name}）",
                details={"collection": self._collection_name},
            ) from exc
        self._collection_ready = True

    def _create_collection(self, client: Any) -> None:
        """使用高层 schema API 创建统一 collection 及其向量索引。"""
        from pymilvus import DataType  # 惰性导入，import 期不依赖 SDK

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=_ID_MAX_LENGTH,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._dim,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=_TEXT_MAX_LENGTH,
        )
        schema.add_field(
            field_name="source_id",
            datatype=DataType.VARCHAR,
            max_length=_SOURCE_ID_MAX_LENGTH,
        )
        schema.add_field(
            field_name="vector_type",
            datatype=DataType.VARCHAR,
            max_length=_VECTOR_TYPE_MAX_LENGTH,
        )
        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )

    # ------------------------------------------------------------------ #
    # 写入                                                                #
    # ------------------------------------------------------------------ #
    def write(self, record: VectorRecord) -> None:
        """写入单条向量记录。

        写入前先做维度校验：``len(record.vector) != dim`` 时直接抛
        :class:`DimensionMismatchError`（Req 21.4），且不触发任何 DB 调用，
        从而保证"未写入数据不丢失"（Req 21.5）。校验通过后持久化原始文本、
        ``source_id``、``vector_type`` 与 ``metadata``（Req 21.3）。

        Raises:
            DimensionMismatchError: 向量维度与配置维度不一致。
            VectorDatabaseUnavailableError: 连接或写入失败（Req 21.5）。
        """
        self._check_dim(len(record.vector))
        self.ensure_collection()
        row = self._record_to_row(record)
        try:
            client = self._milvus.get_client()
            client.insert(collection_name=self._collection_name, data=[row])
        except (VectorDatabaseUnavailableError, DimensionMismatchError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorDatabaseUnavailableError(
                f"写入 Milvus 失败（collection={self._collection_name}）",
                details={"collection": self._collection_name},
            ) from exc

    def write_many(self, records: list[VectorRecord]) -> int:
        """批量写入向量记录。

        先对全部记录做维度校验（任一不一致即 fail-fast，不写入任何记录），
        再一次性插入，返回成功写入的记录数。空列表返回 0 且不触发 DB 调用。

        Raises:
            DimensionMismatchError: 任一记录维度与配置维度不一致。
            VectorDatabaseUnavailableError: 连接或写入失败（Req 21.5）。
        """
        for record in records:
            self._check_dim(len(record.vector))
        if not records:
            return 0
        self.ensure_collection()
        rows = [self._record_to_row(record) for record in records]
        try:
            client = self._milvus.get_client()
            client.insert(collection_name=self._collection_name, data=rows)
        except (VectorDatabaseUnavailableError, DimensionMismatchError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorDatabaseUnavailableError(
                f"批量写入 Milvus 失败（collection={self._collection_name}）",
                details={"collection": self._collection_name},
            ) from exc
        return len(rows)

    # ------------------------------------------------------------------ #
    # 检索                                                                #
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        min_score: float,
        source_scope: str | None = None,
        vector_type: VectorType | None = None,
    ) -> list[SearchHit]:
        """相似度检索，返回分数 ≥ ``min_score``、按分数降序、数量 ≤ ``top_k`` 的命中。

        Args:
            query_vector: 查询向量，维度须等于 :attr:`dim`。
            top_k: 返回数量上界。
            min_score: 最小相似度阈值（COSINE，越大越相似）。
            source_scope: 可选，仅检索 ``source_id == source_scope`` 的记录。
            vector_type: 可选，仅检索指定向量类型的记录。

        Raises:
            DimensionMismatchError: 查询向量维度与配置维度不一致（Req 21.4）。
            VectorDatabaseUnavailableError: 连接或检索失败（Req 21.5）。
        """
        self._check_dim(len(query_vector))
        self.ensure_collection()
        filter_expr = self._build_filter(source_scope, vector_type)
        try:
            client = self._milvus.get_client()
            raw = client.search(
                collection_name=self._collection_name,
                data=[list(query_vector)],
                limit=top_k,
                output_fields=list(_OUTPUT_FIELDS),
                filter=filter_expr,
                search_params={"metric_type": "COSINE", "params": {}},
            )
        except VectorDatabaseUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorDatabaseUnavailableError(
                f"检索 Milvus 失败（collection={self._collection_name}）",
                details={"collection": self._collection_name},
            ) from exc

        hits = [hit for hit in self._parse_search_result(raw) if hit.score >= min_score]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    # ------------------------------------------------------------------ #
    # 删除                                                                #
    # ------------------------------------------------------------------ #
    def delete_by_source(self, source_file_id: str) -> int:
        """删除指定来源标识的全部记录，返回删除数量（支撑 Req 22.3）。

        Raises:
            VectorDatabaseUnavailableError: 连接或删除失败（Req 21.5）。
        """
        self.ensure_collection()
        filter_expr = f"source_id == '{self._escape(source_file_id)}'"
        try:
            client = self._milvus.get_client()
            result = client.delete(
                collection_name=self._collection_name, filter=filter_expr
            )
        except VectorDatabaseUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorDatabaseUnavailableError(
                f"删除 Milvus 记录失败（collection={self._collection_name}）",
                details={"collection": self._collection_name},
            ) from exc
        return self._extract_delete_count(result)

    # ------------------------------------------------------------------ #
    # 内部工具                                                            #
    # ------------------------------------------------------------------ #
    def _check_dim(self, actual_dim: int) -> None:
        """维度一致性校验，不一致抛 :class:`DimensionMismatchError`（Req 21.4）。"""
        if actual_dim != self._dim:
            raise DimensionMismatchError(
                f"向量维度({actual_dim})与 Embedding_Model 输出维度({self._dim})不一致",
                details={"expected_dim": self._dim, "actual_dim": actual_dim},
            )

    def _record_to_row(self, record: VectorRecord) -> dict[str, Any]:
        """将 :class:`VectorRecord` 转换为 Milvus 行（含文本/来源/类型/元数据，Req 21.3）。"""
        return {
            "id": record.id,
            "vector": list(record.vector),
            "text": record.text,
            "source_id": record.source_id,
            "vector_type": self._vector_type_value(record.vector_type),
            "metadata": dict(record.metadata),
        }

    @staticmethod
    def _vector_type_value(vector_type: Any) -> str:
        """归一化向量类型为其字符串值。"""
        if isinstance(vector_type, VectorType):
            return vector_type.value
        return str(vector_type)

    def _build_filter(
        self, source_scope: str | None, vector_type: VectorType | None
    ) -> str:
        """构造 Milvus 过滤表达式；无条件时返回空串（即不过滤）。

        以 ``&&`` 连接多个等值条件，并对字面量做单引号/反斜杠转义以防注入。
        """
        clauses: list[str] = []
        if source_scope is not None:
            clauses.append(f"source_id == '{self._escape(source_scope)}'")
        if vector_type is not None:
            clauses.append(
                f"vector_type == '{self._escape(self._vector_type_value(vector_type))}'"
            )
        return " && ".join(clauses)

    @staticmethod
    def _escape(value: Any) -> str:
        """转义过滤表达式字面量中的反斜杠与单引号，避免表达式注入。"""
        return str(value).replace("\\", "\\\\").replace("'", "\\'")

    def _parse_search_result(self, raw: Any) -> list[SearchHit]:
        """将底层 client 的 search 返回解析为 :class:`SearchHit` 列表。

        兼容高层 ``MilvusClient.search`` 的返回形态：外层按查询向量分组，内层每个命中
        为含 ``id`` / ``distance`` / ``entity`` 的 dict。
        """
        hits: list[SearchHit] = []
        if not raw:
            return hits
        first = raw[0]
        for item in first:
            entity = item.get("entity", {}) if isinstance(item, dict) else {}
            hit_id = item.get("id") if isinstance(item, dict) else None
            if hit_id is None:
                hit_id = entity.get("id")
            score = item.get("distance") if isinstance(item, dict) else None
            metadata = entity.get("metadata") or {}
            hits.append(
                SearchHit(
                    id=str(hit_id),
                    text=entity.get("text", ""),
                    source_id=entity.get("source_id", ""),
                    vector_type=VectorType(entity.get("vector_type")),
                    score=float(score) if score is not None else 0.0,
                    metadata=dict(metadata),
                )
            )
        return hits

    @staticmethod
    def _extract_delete_count(result: Any) -> int:
        """从底层 delete 返回值中稳健提取删除数量。"""
        if result is None:
            return 0
        if isinstance(result, bool):  # 防御：bool 是 int 子类，单独处理
            return int(result)
        if isinstance(result, int):
            return result
        if isinstance(result, dict):
            if "delete_count" in result:
                return int(result["delete_count"])
            if "delete_cnt" in result:
                return int(result["delete_cnt"])
        count = getattr(result, "delete_count", None)
        if count is not None:
            return int(count)
        try:
            return len(result)
        except TypeError:
            return 0

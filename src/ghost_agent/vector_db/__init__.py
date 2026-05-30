"""向量数据库层：基于 Milvus 的 vector_store（统一存储文档分片向量与消息向量）。"""

from ghost_agent.vector_db.vector_store import SearchHit, VectorStore

__all__ = ["VectorStore", "SearchHit"]

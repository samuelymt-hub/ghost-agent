"""基础设施客户端封装：Doubao Embedding 客户端、Milvus 客户端。"""

from ghost_agent.clients.doubao_client import DoubaoEmbeddingClient
from ghost_agent.clients.milvus_client import MilvusClientWrapper

__all__ = ["DoubaoEmbeddingClient", "MilvusClientWrapper"]

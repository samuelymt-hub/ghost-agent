"""pytest 公共 fixtures。

外部依赖（Doubao Embedding / Milvus / Chat_Model / MCP / send_msg）后续任务会在这里
以 mock/stub 替身隔离，使属性测试（Hypothesis, max_examples>=100）可低成本运行。
"""
import pytest


@pytest.fixture
def fake_embedding():
    """占位：任务 3.x 提供可控的 Embedding 替身（固定维度、可注入失败）。"""
    raise NotImplementedError("将在任务 3.x 实现")


@pytest.fixture
def fake_vector_store():
    """占位：内存版 vector_store，供 Indexer / Retriever / KBA 属性测试使用。"""
    raise NotImplementedError("将在任务 3.x 实现")

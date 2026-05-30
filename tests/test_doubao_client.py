"""DoubaoEmbeddingClient 单元测试（任务 3.1）。

覆盖范围（基础设施客户端封装，按设计采用示例化测试，不使用 Hypothesis）：
- 构造期不连接（惰性），``dim`` / ``max_input_len`` 反映 settings 默认值与覆盖值。
- ``embed([])`` 直接返回 ``[]`` 且不构建客户端。
- monkeypatch ``_build_client`` 注入假客户端，乱序 ``.index`` 的返回被按输入顺序重排。
- 假客户端在 ``embeddings.create`` 抛异常 -> ``embed`` 抛 QueryEmbeddingFailedError，
  且 ``__cause__`` 为原始异常。
- 空 api_key 且不 monkeypatch -> ``embed`` 抛 QueryEmbeddingFailedError
  （覆盖"未配置 API Key"或 SDK 不可用路径，二者皆为合法触发原因）。
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ghost_agent.clients import DoubaoEmbeddingClient
from ghost_agent.models.errors import QueryEmbeddingFailedError


# --------------------------------------------------------------------------- #
# 假 SDK 客户端                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class _FakeEmbeddingItem:
    embedding: list[float]
    index: int


class _FakeEmbeddingsResponse:
    def __init__(self, data: list[_FakeEmbeddingItem]) -> None:
        self.data = data


class _FakeEmbeddingsNamespace:
    """模拟 ``client.embeddings``，记录调用并返回乱序结果。"""

    def __init__(self, vectors_by_index: dict[int, list[float]], *, raises: Exception | None = None) -> None:
        self._vectors_by_index = vectors_by_index
        self._raises = raises
        self.calls: list[dict] = []

    def create(self, *, model: str, input: list[str]):  # noqa: A002 - 对齐 SDK 入参名
        self.calls.append({"model": model, "input": list(input)})
        if self._raises is not None:
            raise self._raises
        # 故意以乱序 index 返回，验证 embed 会按 index 重排回输入顺序。
        items = [
            _FakeEmbeddingItem(embedding=self._vectors_by_index[i], index=i)
            for i in range(len(input))
        ]
        shuffled = list(reversed(items))
        return _FakeEmbeddingsResponse(shuffled)


class _FakeArkClient:
    def __init__(self, embeddings: _FakeEmbeddingsNamespace) -> None:
        self.embeddings = embeddings


# --------------------------------------------------------------------------- #
# 构造与属性                                                                    #
# --------------------------------------------------------------------------- #
def test_construction_without_api_key_does_not_raise():
    """无 API Key 构造不应抛错（惰性连接）。"""
    client = DoubaoEmbeddingClient(api_key="")
    assert client is not None
    assert client._client is None  # 尚未构建底层 SDK 客户端


def test_dim_and_max_input_len_reflect_settings_defaults():
    from ghost_agent.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    client = DoubaoEmbeddingClient()
    assert client.dim == settings.embedding_dim
    assert client.max_input_len == settings.embedding_max_input_length


def test_dim_and_max_input_len_overrides():
    client = DoubaoEmbeddingClient(dim=128, max_input_len=512, model="custom-model")
    assert client.dim == 128
    assert client.max_input_len == 512
    assert client.model == "custom-model"


# --------------------------------------------------------------------------- #
# embed 行为                                                                    #
# --------------------------------------------------------------------------- #
def test_embed_empty_list_returns_empty_without_building_client(monkeypatch):
    client = DoubaoEmbeddingClient(api_key="")

    def _should_not_be_called():
        raise AssertionError("空输入不应构建底层客户端")

    monkeypatch.setattr(client, "_build_client", _should_not_be_called)
    assert client.embed([]) == []
    assert client._client is None


def test_embed_realigns_vectors_to_input_order(monkeypatch):
    vectors_by_index = {0: [0.0, 0.1], 1: [1.0, 1.1], 2: [2.0, 2.1]}
    fake_namespace = _FakeEmbeddingsNamespace(vectors_by_index)
    fake_client = _FakeArkClient(fake_namespace)

    client = DoubaoEmbeddingClient(api_key="dummy-key")
    monkeypatch.setattr(client, "_build_client", lambda: fake_client)

    result = client.embed(["a", "b", "c"])

    # 尽管假客户端乱序返回，结果应按输入顺序对齐 index。
    assert result == [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]]
    assert fake_namespace.calls[0]["input"] == ["a", "b", "c"]


def test_embed_caches_client_across_calls(monkeypatch):
    vectors_by_index = {0: [0.0]}
    fake_client = _FakeArkClient(_FakeEmbeddingsNamespace(vectors_by_index))

    client = DoubaoEmbeddingClient(api_key="dummy-key")
    build_calls = {"count": 0}

    def _build():
        build_calls["count"] += 1
        return fake_client

    monkeypatch.setattr(client, "_build_client", _build)
    client.embed(["x"])
    client.embed(["x"])
    assert build_calls["count"] == 1  # 客户端只构建一次并被缓存


def test_embed_wraps_sdk_exception(monkeypatch):
    original = RuntimeError("network down")
    fake_namespace = _FakeEmbeddingsNamespace({}, raises=original)
    fake_client = _FakeArkClient(fake_namespace)

    client = DoubaoEmbeddingClient(api_key="dummy-key")
    monkeypatch.setattr(client, "_build_client", lambda: fake_client)

    with pytest.raises(QueryEmbeddingFailedError) as exc_info:
        client.embed(["a"])
    assert exc_info.value.__cause__ is original


def test_embed_with_empty_key_and_no_monkeypatch_raises(monkeypatch):
    """空 API Key 且无注入时，embed 必须抛 QueryEmbeddingFailedError。

    触发原因可能是"未配置 API Key"或底层 SDK 不可用，二者均为合法的失败路径，
    因此断言只校验异常类型而非具体原因。
    """
    client = DoubaoEmbeddingClient(api_key="")
    with pytest.raises(QueryEmbeddingFailedError):
        client.embed(["hello"])

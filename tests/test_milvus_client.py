"""MilvusClientWrapper 单元测试（任务 3.1）。

覆盖范围（基础设施连接封装，按设计采用示例化测试，不使用 Hypothesis）：
- 构造期不连接（无运行中 Milvus 也不报错）。
- ``is_lite`` 正确区分本地文件路径与 http(s) standalone 地址。
- monkeypatch ``_build_client`` 注入假客户端，``get_client`` 返回并缓存（二次调用同实例，
  ``_build_client`` 仅调用一次）。
- ``_build_client`` 抛异常 -> ``connect`` / ``get_client`` 抛 VectorDatabaseUnavailableError
  （保留 ``__cause__``）。
- ``close()`` 清除缓存，后续 ``get_client`` 重新连接。
"""
from __future__ import annotations

import pytest

from ghost_agent.clients import MilvusClientWrapper
from ghost_agent.models.errors import VectorDatabaseUnavailableError


class _FakeMilvusClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# 构造与 is_lite                                                                #
# --------------------------------------------------------------------------- #
def test_construction_does_not_connect():
    wrapper = MilvusClientWrapper(uri="http://localhost:19530")
    assert wrapper._client is None  # 构造期不连接


@pytest.mark.parametrize(
    ("uri", "expected_lite"),
    [
        ("./milvus_dev.db", True),
        ("milvus_dev.db", True),
        ("/var/data/milvus_dev.db", True),
        ("http://localhost:19530", False),
        ("https://in03-xxxx.api.gcp-us-west1.zillizcloud.com", False),
        ("HTTP://LOCALHOST:19530", False),
    ],
)
def test_is_lite_classification(uri: str, expected_lite: bool):
    wrapper = MilvusClientWrapper(uri=uri)
    assert wrapper.is_lite is expected_lite
    assert wrapper.uri == uri


# --------------------------------------------------------------------------- #
# 连接缓存                                                                      #
# --------------------------------------------------------------------------- #
def test_get_client_returns_and_caches(monkeypatch):
    wrapper = MilvusClientWrapper(uri="./milvus_dev.db")
    fake = _FakeMilvusClient()
    build_calls = {"count": 0}

    def _build():
        build_calls["count"] += 1
        return fake

    monkeypatch.setattr(wrapper, "_build_client", _build)

    first = wrapper.get_client()
    second = wrapper.get_client()
    assert first is fake
    assert second is fake
    assert build_calls["count"] == 1  # 仅构建一次并缓存


def test_connect_wraps_failure(monkeypatch):
    wrapper = MilvusClientWrapper(uri="http://unreachable:19530", timeout=3.0)
    original = ConnectionError("refused")

    def _build():
        raise original

    monkeypatch.setattr(wrapper, "_build_client", _build)

    with pytest.raises(VectorDatabaseUnavailableError) as exc_info:
        wrapper.connect()
    assert exc_info.value.__cause__ is original
    # 错误详情应携带 uri 与超时配置（Req 21.5）。
    assert exc_info.value.details == {
        "uri": "http://unreachable:19530",
        "timeout_seconds": 3.0,
    }


def test_get_client_propagates_connect_failure(monkeypatch):
    wrapper = MilvusClientWrapper(uri="http://unreachable:19530")
    monkeypatch.setattr(
        wrapper, "_build_client", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(VectorDatabaseUnavailableError):
        wrapper.get_client()


def test_close_clears_cached_client(monkeypatch):
    wrapper = MilvusClientWrapper(uri="./milvus_dev.db")
    clients = [_FakeMilvusClient(), _FakeMilvusClient()]
    build_calls = {"count": 0}

    def _build():
        client = clients[build_calls["count"]]
        build_calls["count"] += 1
        return client

    monkeypatch.setattr(wrapper, "_build_client", _build)

    first = wrapper.get_client()
    wrapper.close()
    assert first.closed is True
    assert wrapper._client is None

    # close 后再次获取应重建连接（返回第二个假客户端）。
    second = wrapper.get_client()
    assert second is not first
    assert build_calls["count"] == 2


def test_close_without_connection_is_noop():
    wrapper = MilvusClientWrapper(uri="./milvus_dev.db")
    wrapper.close()  # 不应抛错
    assert wrapper._client is None

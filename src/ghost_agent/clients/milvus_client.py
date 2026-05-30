"""Milvus 向量库连接封装。

封装 ``pymilvus`` 高层 ``MilvusClient`` 的连接管理（Req 21.1, 21.5）：统一处理连接
地址、鉴权 token、database 名称与连接超时配置，并对外提供已连接的客户端句柄。

设计要点：
- **惰性连接（Lazy connection）**：构造函数不连接 Milvus、不进行任何网络/磁盘 IO，
  仅记录配置。真正的连接在首次 :meth:`connect` / :meth:`get_client` 调用时建立并缓存。
  这使模块可在无运行中 Milvus 的环境下被 ``import`` 与单元测试。
- **Milvus Lite 与 standalone 兼容**：``MilvusClient`` 既支持本地文件路径（Milvus Lite，
  如 ``./milvus_dev.db``），也支持 ``http(s)://host:port`` 的 standalone/cloud 地址。
  :attr:`is_lite` 根据 URI 形态做轻量判别。
- **统一异常封装**：连接失败统一包装为 :class:`VectorDatabaseUnavailableError`（Req 21.5），
  错误详情携带配置的连接超时，并保留原始异常作为 ``__cause__``。
- **可测试性 seam**：内部通过 :meth:`_build_client` 构建底层 SDK 客户端，测试可
  monkeypatch 注入假客户端，从而在离线环境验证连接缓存与异常封装行为。

本文件只负责"连接 + 客户端句柄"，**不**创建 collection（collection/schema 由
任务 3.2 的 vector_store 负责）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ghost_agent.config import get_settings
from ghost_agent.models.errors import VectorDatabaseUnavailableError

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查，不在运行期导入 SDK
    from pymilvus import MilvusClient

__all__ = ["MilvusClientWrapper"]


class MilvusClientWrapper:
    """Milvus 连接封装（惰性连接）。

    Args:
        uri: 连接地址；为 ``None`` 时取 ``settings.milvus_uri``。
        token: 鉴权 token；为 ``None`` 时取 ``settings.milvus_token``。
        db_name: database 名称；为 ``None`` 时取 ``settings.milvus_db_name``。
        timeout: 连接超时（秒）；为 ``None`` 时取
            ``settings.milvus_connection_timeout_seconds``（Req 21.5）。
    """

    def __init__(
        self,
        *,
        uri: str | None = None,
        token: str | None = None,
        db_name: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._uri: str = uri if uri is not None else settings.milvus_uri
        self._token: str = token if token is not None else settings.milvus_token
        self._db_name: str = db_name if db_name is not None else settings.milvus_db_name
        self._timeout: float = (
            timeout
            if timeout is not None
            else settings.milvus_connection_timeout_seconds
        )
        # 惰性建立并缓存底层客户端；构造期间不连接。
        self._client: Any | None = None

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def uri(self) -> str:
        """配置的 Milvus 连接地址。"""
        return self._uri

    @property
    def db_name(self) -> str:
        """配置的 Milvus database 名称。"""
        return self._db_name

    @property
    def timeout(self) -> float:
        """配置的连接超时（秒，Req 21.5）。"""
        return self._timeout

    @property
    def is_lite(self) -> bool:
        """URI 是否指向 Milvus Lite（本地文件）而非 standalone/cloud。

        判别规则：以 ``http://`` 或 ``https://`` 开头视为 standalone/cloud（非 Lite），
        其余（本地文件路径，如 ``./milvus_dev.db``）视为 Milvus Lite。
        """
        normalized = self._uri.strip().lower()
        return not (normalized.startswith("http://") or normalized.startswith("https://"))

    # ------------------------------------------------------------------ #
    # 测试 seam：构建底层 SDK 客户端                                       #
    # ------------------------------------------------------------------ #
    def _build_client(self) -> Any:
        """构建底层 ``pymilvus.MilvusClient``。

        SDK 在方法内部惰性导入，避免 import 期或无关测试受 SDK 安装/配置影响。
        测试可通过 ``monkeypatch.setattr(instance, "_build_client", fake)`` 注入
        假客户端。
        """
        from pymilvus import MilvusClient

        return MilvusClient(
            uri=self._uri,
            token=self._token,
            db_name=self._db_name,
            timeout=self._timeout,
        )

    # ------------------------------------------------------------------ #
    # 公共 API                                                            #
    # ------------------------------------------------------------------ #
    def connect(self) -> "MilvusClient":
        """建立（并缓存）到 Milvus 的连接。

        Returns:
            已连接的底层 ``MilvusClient`` 句柄。

        Raises:
            VectorDatabaseUnavailableError: 在配置的连接超时内无法连接或连接失败时抛出
                （Req 21.5），错误详情携带 ``uri`` 与 ``timeout_seconds``，
                并保留原始异常为 ``__cause__``。
        """
        try:
            self._client = self._build_client()
        except VectorDatabaseUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一封装连接异常
            raise VectorDatabaseUnavailableError(
                f"无法在 {self._timeout} 秒内连接 Milvus 向量库（uri={self._uri}）",
                details={"uri": self._uri, "timeout_seconds": self._timeout},
            ) from exc
        return self._client

    def get_client(self) -> "MilvusClient":
        """返回缓存的连接，必要时惰性建立连接。"""
        if self._client is None:
            return self.connect()
        return self._client

    def close(self) -> None:
        """关闭并清除缓存的连接（若存在），后续 ``get_client`` 将重新连接。"""
        client = self._client
        self._client = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - 关闭失败不应向上传播
                pass

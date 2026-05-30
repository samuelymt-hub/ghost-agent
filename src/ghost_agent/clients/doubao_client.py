"""Doubao（火山方舟）Embedding 客户端封装。

封装对 ``Doubao-embedding-text-240715`` 嵌入模型的调用（Req 21.2, 23.4），
对上层提供稳定、与具体 SDK 解耦的 ``embed(texts) -> vectors`` 接口，并暴露
输出维度 ``dim`` 与最大输入长度 ``max_input_len`` 供索引/检索层做维度与长度校验。

设计要点：
- **惰性连接（Lazy connection）**：构造函数不创建任何 SDK 客户端、不进行任何
  网络调用。真正的 ``volcenginesdkarkruntime.Ark`` 客户端在首次 ``embed`` 调用时
  才构建并缓存。这样模块可在无 API Key、无网络环境下被 ``import`` 与单元测试。
- **统一异常封装**：任何 SDK 导入失败、未配置 API Key 或网络/调用异常，均被包装为
  :class:`QueryEmbeddingFailedError`（Req 8.8）并保留原始异常作为 ``__cause__``，
  方便上层做统一重试与错误上报。
- **可测试性 seam**：内部通过 :meth:`_build_client` 构建底层 SDK 客户端，测试可
  monkeypatch 该方法注入假客户端，从而在离线环境验证 ``embed`` 行为。
"""

from __future__ import annotations

from typing import Any

from ghost_agent.config import get_settings
from ghost_agent.models.errors import QueryEmbeddingFailedError

__all__ = ["DoubaoEmbeddingClient"]


class DoubaoEmbeddingClient:
    """Doubao 嵌入模型客户端（惰性连接）。

    Args:
        api_key: 火山引擎 API Key；为 ``None`` 时取 ``settings.doubao_api_key``。
        base_url: 火山方舟 API Base URL；为 ``None`` 时取 ``settings.doubao_base_url``。
        model: Embedding 模型 ID；为 ``None`` 时取 ``settings.doubao_embedding_model``。
        dim: 输出向量维度；为 ``None`` 时取 ``settings.embedding_dim``（Req 21.2/21.4）。
        max_input_len: 最大输入长度；为 ``None`` 时取
            ``settings.embedding_max_input_length``（Req 6.4）。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        max_input_len: int | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key: str = api_key if api_key is not None else settings.doubao_api_key
        self._base_url: str = (
            base_url if base_url is not None else settings.doubao_base_url
        )
        self._model: str = model if model is not None else settings.doubao_embedding_model
        self._dim: int = dim if dim is not None else settings.embedding_dim
        self._max_input_len: int = (
            max_input_len
            if max_input_len is not None
            else settings.embedding_max_input_length
        )
        # 惰性构建并缓存底层 SDK 客户端；构造期间不触碰网络。
        self._client: Any | None = None

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        """Embedding 输出向量维度（Req 21.2/21.4）。"""
        return self._dim

    @property
    def max_input_len(self) -> int:
        """单次嵌入的最大输入长度（Req 6.4）。"""
        return self._max_input_len

    @property
    def model(self) -> str:
        """当前使用的 Embedding 模型 ID。"""
        return self._model

    # ------------------------------------------------------------------ #
    # 测试 seam：构建底层 SDK 客户端                                       #
    # ------------------------------------------------------------------ #
    def _build_client(self) -> Any:
        """构建底层 volcengine Ark 客户端。

        SDK 在方法内部惰性导入，避免 import 期或无关测试受 SDK 安装/配置影响。
        测试可通过 ``monkeypatch.setattr(instance, "_build_client", fake)`` 注入
        假客户端。

        Raises:
            QueryEmbeddingFailedError: SDK 不可导入或未配置 API Key 时抛出。
        """
        if not self._api_key:
            raise QueryEmbeddingFailedError(
                "未配置 Doubao API Key，无法调用 Embedding 服务（请设置 DOUBAO_API_KEY）"
            )
        try:
            from volcenginesdkarkruntime import Ark
        except Exception as exc:  # noqa: BLE001 - SDK 导入失败统一封装
            raise QueryEmbeddingFailedError(
                "Doubao(volcengine) SDK 导入失败，无法构建 Embedding 客户端"
            ) from exc
        return Ark(api_key=self._api_key, base_url=self._base_url)

    def _get_client(self) -> Any:
        """返回缓存的底层客户端，必要时惰性构建。"""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # ------------------------------------------------------------------ #
    # 公共 API                                                            #
    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str]) -> list[list[float]]:
        """将一批文本转换为嵌入向量，按输入顺序返回。

        Args:
            texts: 待嵌入的文本列表。空列表直接返回 ``[]`` 且不构建客户端。

        Returns:
            与 ``texts`` 一一对应、顺序一致的嵌入向量列表。

        Raises:
            QueryEmbeddingFailedError: 输入非法、SDK 不可用、未配置 API Key，
                或底层嵌入调用发生任何异常时抛出（保留原始异常为 ``__cause__``）。
        """
        if texts is None:
            raise QueryEmbeddingFailedError("嵌入输入不能为 None")
        if len(texts) == 0:
            return []

        client = self._get_client()
        try:
            response = client.embeddings.create(model=self._model, input=texts)
        except QueryEmbeddingFailedError:
            # 来自 _get_client/_build_client 的封装异常，原样向上抛出。
            raise
        except Exception as exc:  # noqa: BLE001 - 统一封装 SDK/网络异常供上层重试
            raise QueryEmbeddingFailedError(
                "调用 Doubao Embedding 服务失败"
            ) from exc

        return self._extract_vectors(response, expected_count=len(texts))

    # ------------------------------------------------------------------ #
    # 内部工具                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_vectors(response: Any, *, expected_count: int) -> list[list[float]]:
        """从 SDK 响应中提取向量并按 ``.index`` 重排，保证与输入顺序一致。

        Args:
            response: SDK 返回对象，``.data`` 为含 ``.embedding`` 与 ``.index`` 的条目列表。
            expected_count: 期望的向量数量（用于校验返回完整性）。

        Raises:
            QueryEmbeddingFailedError: 响应结构非法或数量不匹配时抛出。
        """
        try:
            data = list(response.data)
        except Exception as exc:  # noqa: BLE001
            raise QueryEmbeddingFailedError(
                "Doubao Embedding 响应缺少 data 字段或结构非法"
            ) from exc

        if len(data) != expected_count:
            raise QueryEmbeddingFailedError(
                f"Doubao Embedding 返回数量({len(data)})与输入数量({expected_count})不一致"
            )

        try:
            ordered = sorted(data, key=lambda item: item.index)
            return [list(item.embedding) for item in ordered]
        except Exception as exc:  # noqa: BLE001
            raise QueryEmbeddingFailedError(
                "Doubao Embedding 响应条目缺少 index/embedding 字段"
            ) from exc

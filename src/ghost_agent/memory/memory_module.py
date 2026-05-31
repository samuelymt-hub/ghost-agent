"""Memory_Module（记忆模块，Req 18）。

本模块实现多轮对话记忆的运行时管理组件 :class:`MemoryModule`，对上层 Agent
（Conversation_Agent / Prompt_Module）提供与具体持久化解耦的稳定接口：

* :meth:`MemoryModule.append` —— 一轮应答完成后，将本轮用户消息与应答按时间
  先后顺序追加写入对应 Session 的 Short_Term_Memory（Req 18.1）；当短期记忆消息
  数超过配置的保留条数上限（≥1）时，将溢出的较早消息总结后写入 Long_Term_Memory
  （Req 18.2），并从短期记忆移除已总结的较早消息使其数量 <= 上限（Req 18.3）；
  总结或写入长期记忆失败时，将相关消息保留在短期记忆并记录失败（Req 18.4）。
* :meth:`MemoryModule.load` —— 按时间先后顺序返回某 Session 的短期/长期记忆
  （Req 18.5）；每个 Session 的记忆仅来源于且仅作用于该 Session（隔离，Req 18.6）。

设计要点：

- **本实现为内存版参考实现**：以 ``session_id`` 为键在进程内字典中保存短期/长期
  记忆，便于单元/属性测试在离线环境验证记忆语义。后续可在不改变接口的前提下替换
  为持久化实现（如数据库 / LangGraph Checkpointer）。
- **总结器 seam（Summarizer）**：总结逻辑通过可注入的 ``summarizer`` 回调隔离，
  签名为 ``Callable[[list[Message]], str]``。**默认总结器为离线确定性实现**
  （拼接/截断历史消息文本），因此 ``MemoryModule()`` 无需 API Key、无需网络即可使用
  与测试。真正基于大模型的总结器由 :func:`llm_summarizer` 工厂在装配阶段按需构建，
  默认不启用。
- **时间顺序稳健性**：``created_at`` 通过可注入的 ``clock`` 生成，并由内部
  ``_next_timestamp`` 保证单调不减（即便时钟分辨率较粗或回拨）。由于列表追加顺序由
  本模块控制，短期记忆的列表顺序始终等于追加顺序（Property 25）。
- **记忆隔离**：所有读写操作只触碰目标 ``session_id`` 对应的列表，不同 Session
  之间的记忆相互独立（Property 23）。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from ghost_agent.config import get_settings
from ghost_agent.models.errors import MemorySummarizeFailedError
from ghost_agent.models.memory import (
    LongTermSummary,
    Message,
    Role,
    ShortTermMemory,
)

__all__ = [
    "Summarizer",
    "MemoryModule",
    "llm_summarizer",
]

logger = logging.getLogger(__name__)

#: 总结器签名：输入一批待归档的较早消息，返回其总结文本（非空）。
Summarizer = Callable[[list[Message]], str]


def _default_summarizer(messages: list[Message]) -> str:
    """默认离线总结器：拼接并截断历史消息内容（确定性、无网络）。

    仅用于让 :class:`MemoryModule` 在无 API Key / 无网络环境下可用与可测；真正的
    大模型总结请使用 :func:`llm_summarizer` 构建的总结器。返回值保证非空（满足
    :class:`LongTermSummary` 的 ``summary_text`` 非空约束）。
    """
    joined = "；".join(message.content for message in messages)
    text = joined[:200]
    if not text:
        # 理论上不会触发（归档时消息数 >= 1 且 content 非空），保留兜底以确保非空。
        text = f"总结({len(messages)}条历史消息)"
    return text


class MemoryModule:
    """多轮对话记忆管理器（内存版参考实现，Req 18）。

    Args:
        retention_limit: 短期记忆保留条数上限（≥1）；为 ``None`` 时取
            ``settings.short_term_memory_limit``。小于 1 时构造即拒绝（Req 18.2）。
        summarizer: 较早消息总结器；为 ``None`` 时使用默认离线确定性总结器。
        clock: 生成 ``created_at`` 的时钟；为 ``None`` 时使用
            ``datetime.now(timezone.utc)``。注入以便测试获得确定性时间顺序。

    Attributes:
        last_summarize_error: 最近一次总结/写入长期记忆失败的错误（Req 18.4）；
            无失败时为 ``None``。
        failures: 历次总结/写入长期记忆失败记录列表（Req 18.4）。
    """

    def __init__(
        self,
        *,
        retention_limit: int | None = None,
        summarizer: Summarizer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if retention_limit is None:
            retention_limit = get_settings().short_term_memory_limit
        if retention_limit < 1:
            raise ValueError(
                f"retention_limit 必须为 >= 1 的正整数 (Req 18.2)，实际收到: {retention_limit}"
            )

        self._retention_limit: int = retention_limit
        self._summarizer: Summarizer = (
            summarizer if summarizer is not None else _default_summarizer
        )
        self._clock: Callable[[], datetime] = (
            clock if clock is not None else (lambda: datetime.now(timezone.utc))
        )

        # 以 session_id 为键的进程内记忆存储（隔离，Req 18.6）。
        self._short_term: dict[str, list[Message]] = {}
        self._long_term: dict[str, list[LongTermSummary]] = {}

        # 单调不减时间戳生成的内部状态。
        self._last_ts: datetime | None = None

        # 失败记录（Req 18.4）。
        self.last_summarize_error: MemorySummarizeFailedError | None = None
        self.failures: list[MemorySummarizeFailedError] = []

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def retention_limit(self) -> int:
        """短期记忆保留条数上限（≥1）。"""
        return self._retention_limit

    # ------------------------------------------------------------------ #
    # 公共 API                                                            #
    # ------------------------------------------------------------------ #
    def append(self, session_id: str, user_msg: str, answer: str) -> None:
        """追加一轮对话（用户消息 + 应答）到对应 Session 的短期记忆（Req 18.1）。

        用户消息与应答按发生先后顺序（先 USER 后 ASSISTANT）以单调不减的
        ``created_at`` 追加。追加后若短期记忆消息数超过保留条数上限，则触发溢出归档
        （Req 18.2, 18.3）。

        Args:
            session_id: 目标会话标识。
            user_msg: 本轮用户消息内容（非空）。
            answer: 本轮应答内容（非空）。
        """
        user_message = Message(
            session_id=session_id,
            role=Role.USER,
            content=user_msg,
            created_at=self._next_timestamp(),
        )
        assistant_message = Message(
            session_id=session_id,
            role=Role.ASSISTANT,
            content=answer,
            created_at=self._next_timestamp(),
        )

        short_term = self._short_term.setdefault(session_id, [])
        # 按时间顺序追加（Req 18.1）：先用户消息，后应答。
        short_term.append(user_message)
        short_term.append(assistant_message)

        self._maybe_archive(session_id)

    def load(self, session_id: str) -> dict:
        """按时间顺序返回某 Session 的短期与长期记忆（Req 18.5, 18.6）。

        Args:
            session_id: 目标会话标识。

        Returns:
            形如 ``{"short_term": ShortTermMemory, "long_term": list[LongTermSummary]}``
            的字典；未知 Session 返回空短期记忆与空长期记忆列表。
        """
        short_messages = list(self._short_term.get(session_id, []))
        long_summaries = list(self._long_term.get(session_id, []))
        return {
            "short_term": ShortTermMemory(
                session_id=session_id, messages=short_messages
            ),
            "long_term": long_summaries,
        }

    # ------------------------------------------------------------------ #
    # 便捷访问器（供测试与上层使用）                                        #
    # ------------------------------------------------------------------ #
    def short_term_messages(self, session_id: str) -> list[Message]:
        """返回某 Session 短期记忆中按时间顺序排列的消息（副本）。"""
        return list(self._short_term.get(session_id, []))

    def long_term_summaries(self, session_id: str) -> list[LongTermSummary]:
        """返回某 Session 长期记忆中按时间顺序排列的总结（副本）。"""
        return list(self._long_term.get(session_id, []))

    # ------------------------------------------------------------------ #
    # 内部：溢出归档                                                       #
    # ------------------------------------------------------------------ #
    def _maybe_archive(self, session_id: str) -> None:
        """当短期记忆超出上限时，将最早的溢出消息总结并写入长期记忆。

        溢出条数 ``overflow = len(short_term) - retention_limit``：取最早的
        ``overflow`` 条消息进行总结（Req 18.2），成功写入 :class:`LongTermSummary`
        后再从短期记忆移除这些消息使其数量 <= 上限（Req 18.3）。

        若总结或写入长期记忆过程中失败，则保留相关消息于短期记忆并记录失败
        （Req 18.4），不向上层抛出异常（避免对话流程中断与消息丢失）；此失败场景下
        短期记忆可能暂时超过上限，这是 Req 18.4 要求的"宁可保留、不可丢失"权衡。
        """
        short_term = self._short_term[session_id]
        overflow = len(short_term) - self._retention_limit
        if overflow <= 0:
            return

        to_archive = list(short_term[:overflow])
        try:
            summary_text = self._summarizer(to_archive)
            summary = LongTermSummary(
                session_id=session_id,
                summary_text=summary_text,
                covered_message_ids=[message.message_id for message in to_archive],
                created_at=self._next_timestamp(),
            )
        except Exception as exc:  # noqa: BLE001 - 统一按 Req 18.4 记录并保留消息
            self._record_failure(session_id, to_archive, exc)
            return

        # 写入成功后再移除，保证失败时消息不丢失（Req 18.3, 18.4）。
        self._long_term.setdefault(session_id, []).append(summary)
        del short_term[:overflow]

    def _record_failure(
        self,
        session_id: str,
        messages: list[Message],
        cause: Exception,
    ) -> None:
        """记录一次总结/写入长期记忆失败（Req 18.4），不向上层抛出。"""
        error = MemorySummarizeFailedError(
            f"会话 {session_id!r} 的较早消息总结/写入长期记忆失败：{cause}",
            details={
                "session_id": session_id,
                "message_ids": [message.message_id for message in messages],
            },
        )
        self.last_summarize_error = error
        self.failures.append(error)
        logger.warning(
            "Memory_Module 总结/写入长期记忆失败（消息保留在短期记忆）：session=%s, count=%d, cause=%r",
            session_id,
            len(messages),
            cause,
        )

    # ------------------------------------------------------------------ #
    # 内部：单调不减时间戳                                                 #
    # ------------------------------------------------------------------ #
    def _next_timestamp(self) -> datetime:
        """返回单调不减的 ``created_at``。

        以注入的 ``clock`` 取当前时间；若取值不晚于上一次（时钟粗粒度或回拨），
        则复用上一次的值以保证非单调递减（满足 :class:`ShortTermMemory` 的
        ``created_at`` 升序校验）。短期记忆的列表顺序由追加顺序决定，与时间戳是否相等
        无关，因此追加顺序始终被保留（Req 18.1, 18.5）。
        """
        timestamp = self._clock()
        if self._last_ts is not None and timestamp <= self._last_ts:
            timestamp = self._last_ts
        self._last_ts = timestamp
        return timestamp


def llm_summarizer(
    chat_model,
    prompt_module,
    *,
    template_name: str = "memory_summary",
) -> Summarizer:
    """构建基于 Chat_Model 的较早消息总结器（按需启用的 LLM 路径，Req 18.2）。

    返回的总结器使用 Prompt_Module 的 ``memory_summary`` 模板构造系统提示词，
    并将待归档历史消息作为用户消息交给 Chat_Model 生成摘要。任何 LLM 调用错误或
    空摘要都会被包装为 :class:`MemorySummarizeFailedError` 抛出，从而触发
    :class:`MemoryModule` 的 Req 18.4 失败处理（保留消息、记录失败）。

    本工厂**不会**被 :class:`MemoryModule` 默认使用；默认总结器为离线确定性实现，
    以保证 ``MemoryModule()`` 在无网络环境下可用。仅在装配阶段（任务 11/12）显式注入。

    Args:
        chat_model: 提供 ``generate(messages) -> Completion`` 的对话模型
            （``ghost_agent.core.chat_model.ChatModel`` 或兼容替身）。
        prompt_module: 提供 ``build(template_name, variables)`` 的提示词模块。
        template_name: 总结所用模板名称，默认 ``"memory_summary"``。

    Returns:
        一个 :data:`Summarizer` 回调。
    """
    # 延迟导入，避免 memory 包导入期强依赖 core 层。
    from ghost_agent.core.chat_model import ChatMessage

    def _summarize(messages: list[Message]) -> str:
        history_text = "\n".join(
            f"{message.role.value}: {message.content}" for message in messages
        )
        prompt = prompt_module.build(template_name)
        chat_messages = [
            ChatMessage(role="system", content=prompt.text),
            ChatMessage(role="user", content=history_text),
        ]
        try:
            completion = chat_model.generate(chat_messages)
        except Exception as exc:  # noqa: BLE001 - 统一封装为记忆总结失败（Req 18.4）
            raise MemorySummarizeFailedError(
                "调用 Chat_Model 总结历史消息失败"
            ) from exc

        content = (completion.content or "").strip()
        if not content:
            raise MemorySummarizeFailedError("Chat_Model 返回空摘要，无法写入长期记忆")
        return content

    return _summarize

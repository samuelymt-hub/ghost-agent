"""会话与多轮对话记忆相关数据模型 (Req 18.x)。

包含:

* :class:`Role`             —— 消息角色枚举。
* :class:`Session`          —— 会话主体。
* :class:`Message`          —— 单条对话消息。
* :class:`ShortTermMemory`  —— 短期记忆 (Req 18.1, 18.3, 18.6)。
* :class:`LongTermSummary`  —— 较早消息总结 (Req 18.2)。

记忆隔离 (Req 18.6) 在 :class:`ShortTermMemory` 中通过校验所有 ``messages`` 的
``session_id`` 与外层一致来强制；时间顺序 (Req 18.1, 18.5) 通过校验
``messages`` 按 ``created_at`` 单调不减来强制。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _new_id() -> str:
    """生成会话/消息的唯一标识。"""
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    """返回带 UTC 时区的当前时间。"""
    return datetime.now(timezone.utc)


class Role(str, Enum):
    """消息角色枚举 (Req 18.1)。"""

    USER = "USER"
    ASSISTANT = "ASSISTANT"


class Session(BaseModel):
    """会话主体：标识一组多轮交互上下文。"""

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    session_id: str = Field(
        default_factory=_new_id,
        description="唯一会话标识。",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="会话创建时间 (UTC)。",
    )


class Message(BaseModel):
    """单条对话消息 (Req 18.1, 18.5, 18.6)。"""

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    message_id: str = Field(
        default_factory=_new_id,
        description="唯一消息标识。",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description="所属会话标识，记忆隔离的归属键 (Req 18.6)。",
    )
    role: Role = Field(
        ...,
        description="消息角色 (USER / ASSISTANT)。",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="消息内容；不允许为空字符串。",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="消息创建时间，用于按时间先后排序 (Req 18.1, 18.5)。",
    )


class ShortTermMemory(BaseModel):
    """短期记忆：保存近期历史消息 (Req 18.1, 18.3, 18.6)。

    本模型只承载结构性约束：

    * 所有内含 ``Message`` 的 ``session_id`` 必须等于本对象的 ``session_id``
      （Req 18.6 记忆隔离）。
    * ``messages`` 必须按 ``created_at`` 单调不减排序（Req 18.1, 18.5）。

    保留条数上限 (Req 18.2, 18.3) 由 :mod:`ghost_agent.memory.memory_module`
    在运行时按配置裁剪，本模型不在此处强制。
    """

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    session_id: str = Field(
        ...,
        min_length=1,
        description="本短期记忆所属会话标识。",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="按时间先后排序的消息列表。",
    )

    @model_validator(mode="after")
    def _check_messages(self) -> "ShortTermMemory":
        # Req 18.6：记忆隔离——所有消息必须归属同一 session
        for idx, msg in enumerate(self.messages):
            if msg.session_id != self.session_id:
                raise ValueError(
                    f"messages[{idx}].session_id ({msg.session_id!r}) "
                    f"与 ShortTermMemory.session_id ({self.session_id!r}) 不一致"
                )

        # Req 18.1, 18.5：按时间先后排序（单调不减）
        for prev, curr in zip(self.messages, self.messages[1:]):
            if curr.created_at < prev.created_at:
                raise ValueError(
                    "messages 必须按 created_at 升序排列 "
                    f"(发现 {prev.created_at} -> {curr.created_at} 逆序)"
                )

        return self


class LongTermSummary(BaseModel):
    """长期记忆：对较早消息的总结 (Req 18.2)。"""

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    session_id: str = Field(
        ...,
        min_length=1,
        description="所属会话标识 (Req 18.6)。",
    )
    summary_text: str = Field(
        ...,
        min_length=1,
        description="较早消息的总结文本 (Req 18.2)。",
    )
    covered_message_ids: list[str] = Field(
        default_factory=list,
        description="本次总结所覆盖的消息标识列表。",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="总结生成时间 (UTC)。",
    )


__all__ = [
    "Role",
    "Session",
    "Message",
    "ShortTermMemory",
    "LongTermSummary",
]

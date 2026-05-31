"""Memory_Module 测试（任务 9.1 / 9.2 / 9.3 / 9.4）。

包含：
- 任务 9.2 属性测试 Property 23（记忆按 Session 隔离，Hypothesis, max_examples>=100）。
- 任务 9.3 属性测试 Property 24（短期记忆容量上界与溢出归档，max_examples>=100）。
- 任务 9.4 属性测试 Property 25（消息追加保持时间顺序，max_examples>=100）。
- 单元测试：未越限不归档、越限归档/移除/计数、总结失败保留消息并记录失败、
  Session 隔离、未知 Session 空记忆、retention_limit<1 拒绝、llm_summarizer 行为。

外部依赖（Chat_Model）以轻量替身隔离；总结器通过注入控制确定性与失败。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.core.chat_model import ChatMessage, Completion
from ghost_agent.memory import MemoryModule, llm_summarizer
from ghost_agent.models.errors import MemorySummarizeFailedError
from ghost_agent.models.memory import Message, Role


# --------------------------------------------------------------------------- #
# 测试工具                                                                      #
# --------------------------------------------------------------------------- #
_BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _monotonic_clock():
    """返回一个每次调用都严格递增的时钟（用于确定性时间顺序）。"""
    counter = {"n": 0}

    def _clock() -> datetime:
        counter["n"] += 1
        return _BASE_TIME + timedelta(seconds=counter["n"])

    return _clock


def _ok_summarizer(messages: list[Message]) -> str:
    """确定性、不失败的总结器（覆盖被归档消息）。"""
    return "摘要:" + "|".join(m.message_id for m in messages)


def _always_fail_summarizer(messages: list[Message]) -> str:
    """总是抛错的总结器，用于触发 Req 18.4 失败处理。"""
    raise RuntimeError("boom: 总结失败")


# =========================================================================== #
# 任务 9.2 — 属性测试 Property 23                                               #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 23: 对任意跨多个 Session 交错写入的对话消息，加载某一 Session 的记忆时返回的所有内容均归属于该 Session，不同 Session 之间记忆内容互不泄漏。
# Validates: Requirements 18.6

_session_id_strategy = st.sampled_from(["s1", "s2", "s3", "s4"])
_content_strategy = st.text(min_size=1, max_size=20).filter(lambda s: s.strip() != "")


@settings(max_examples=150, deadline=None)
@given(
    rounds=st.lists(
        st.tuples(_session_id_strategy, _content_strategy, _content_strategy),
        min_size=0,
        max_size=40,
    ),
    retention_limit=st.integers(min_value=1, max_value=6),
)
def test_property_23_memory_isolated_by_session(
    rounds: list[tuple[str, str, str]], retention_limit: int
):
    """Property 23：加载某 Session 的记忆，所有内容均归属该 Session，互不泄漏。"""
    memory = MemoryModule(
        retention_limit=retention_limit,
        summarizer=_ok_summarizer,
        clock=_monotonic_clock(),
    )

    # 跨多个 Session 交错写入。
    for session_id, user_msg, answer in rounds:
        memory.append(session_id, user_msg, answer)

    # 记录每个 Session 自己写入的消息内容（多重集合，用于校验不丢失/不串扰）。
    own_contents: dict[str, list[str]] = {}
    for session_id, user_msg, answer in rounds:
        own_contents.setdefault(session_id, []).extend([user_msg, answer])

    all_sessions = {"s1", "s2", "s3", "s4"}
    for session_id in all_sessions:
        loaded = memory.load(session_id)
        short_term = loaded["short_term"]
        long_term = loaded["long_term"]

        # 短期记忆所有消息均归属该 Session。
        assert short_term.session_id == session_id
        for msg in short_term.messages:
            assert msg.session_id == session_id

        # 长期记忆所有总结均归属该 Session。
        for summary in long_term:
            assert summary.session_id == session_id

        # 该 Session 短期内出现的消息 id 集合，与长期覆盖 id 集合，
        # 必须全部来自该 Session 自己写入的消息（无跨 Session 串扰）。
        short_ids = {m.message_id for m in short_term.messages}
        covered_ids: set[str] = set()
        for summary in long_term:
            covered_ids.update(summary.covered_message_ids)

        # 短期消息内容必须是该 Session 自己写入内容的子多重集合。
        own = own_contents.get(session_id, [])
        own_multiset: dict[str, int] = {}
        for c in own:
            own_multiset[c] = own_multiset.get(c, 0) + 1
        seen: dict[str, int] = {}
        for msg in short_term.messages:
            seen[msg.content] = seen.get(msg.content, 0) + 1
            assert seen[msg.content] <= own_multiset.get(msg.content, 0), (
                f"Session {session_id} 短期记忆含非本 Session 内容: {msg.content!r}"
            )

        # 短期 id 与长期覆盖 id 不应重叠（已归档的消息从短期移除）。
        assert short_ids.isdisjoint(covered_ids)


# =========================================================================== #
# 任务 9.3 — 属性测试 Property 24                                               #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 24: 对任意任意轮数的对话消息序列，短期记忆中的消息数量在任意时刻均不超过配置的保留条数上限（≥1），溢出的较早消息被总结并纳入长期记忆覆盖范围。
# Validates: Requirements 18.2, 18.3


@settings(max_examples=150, deadline=None)
@given(
    n_rounds=st.integers(min_value=0, max_value=40),
    retention_limit=st.integers(min_value=1, max_value=8),
)
def test_property_24_short_term_capacity_and_overflow_archived(
    n_rounds: int, retention_limit: int
):
    """Property 24：短期记忆任意时刻 <= 上限，溢出消息被长期记忆覆盖。"""
    session_id = "s"
    memory = MemoryModule(
        retention_limit=retention_limit,
        summarizer=_ok_summarizer,
        clock=_monotonic_clock(),
    )

    all_message_ids: list[str] = []
    for i in range(n_rounds):
        memory.append(session_id, f"u{i}", f"a{i}")

        # 任意时刻短期记忆数量不超过上限（非失败路径）。
        short_now = memory.short_term_messages(session_id)
        assert len(short_now) <= retention_limit

        all_message_ids.extend(m.message_id for m in short_now)

    short_term = memory.short_term_messages(session_id)
    long_term = memory.long_term_summaries(session_id)

    short_ids = {m.message_id for m in short_term}
    covered_ids: set[str] = set()
    for summary in long_term:
        covered_ids.update(summary.covered_message_ids)

    total_written = 2 * n_rounds
    # 写入的全部消息 id（最终态由短期保留 + 长期覆盖共同构成）。
    final_short_ids = short_ids
    # 所有曾被移出短期的消息都应被长期覆盖。
    assert len(final_short_ids) <= retention_limit
    # 长期覆盖 + 当前短期 == 写入总数（无丢失、无重复归档）。
    assert len(covered_ids) + len(final_short_ids) == total_written
    # 长期覆盖与短期保留不重叠。
    assert covered_ids.isdisjoint(final_short_ids)


# =========================================================================== #
# 任务 9.4 — 属性测试 Property 25                                               #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 25: 对任意多轮对话序列，写入后短期记忆中的消息按时间先后顺序排列，且每轮的用户消息与应答均按发生顺序被追加。
# Validates: Requirements 18.1


@settings(max_examples=150, deadline=None)
@given(
    rounds=st.lists(
        st.tuples(_content_strategy, _content_strategy),
        min_size=0,
        max_size=30,
    ),
    retention_limit=st.integers(min_value=1, max_value=10),
)
def test_property_25_append_preserves_time_order(
    rounds: list[tuple[str, str]], retention_limit: int
):
    """Property 25：短期记忆按时间顺序排列；每轮 USER 先于其 ASSISTANT。"""
    session_id = "s"
    memory = MemoryModule(
        retention_limit=retention_limit,
        summarizer=_ok_summarizer,
        clock=_monotonic_clock(),
    )

    for user_msg, answer in rounds:
        memory.append(session_id, user_msg, answer)

    short_term = memory.short_term_messages(session_id)
    long_term = memory.long_term_summaries(session_id)

    # created_at 单调不减（时间先后顺序，Req 18.1/18.5）。
    for prev, curr in zip(short_term, short_term[1:]):
        assert prev.created_at <= curr.created_at

    # 全局追加顺序：每轮先 USER（全局偶数下标）后 ASSISTANT（全局奇数下标），
    # 时钟严格递增。短期记忆是被保留的最新一段（最早的消息先被归档移出），
    # 因此短期记忆是全局序列的连续后缀。计算该后缀的起始全局下标并校验角色奇偶，
    # 即同时验证“时间顺序保留”“USER 先于其应答”“溢出按最早优先归档”（适配奇数上限）。
    total_written = 2 * len(rounds)
    covered_count = sum(len(s.covered_message_ids) for s in long_term)
    start_index = total_written - len(short_term)

    # 被移出短期的消息恰好都进入长期覆盖（无丢失、无重复）。
    assert covered_count == start_index

    for offset, msg in enumerate(short_term):
        global_index = start_index + offset
        expected_role = Role.USER if global_index % 2 == 0 else Role.ASSISTANT
        assert msg.role == expected_role


# =========================================================================== #
# 单元测试                                                                      #
# =========================================================================== #
def test_append_below_limit_keeps_all_no_summary():
    """未超过上限：保留全部消息、不产生长期总结。"""
    memory = MemoryModule(retention_limit=4, summarizer=_ok_summarizer)
    memory.append("s", "u1", "a1")  # 2 条 <= 4
    short = memory.short_term_messages("s")
    assert len(short) == 2
    assert [m.role for m in short] == [Role.USER, Role.ASSISTANT]
    assert [m.content for m in short] == ["u1", "a1"]
    assert memory.long_term_summaries("s") == []


def test_append_crossing_limit_archives_oldest_overflow():
    """超过上限：归档最早的溢出消息、从短期移除、长期覆盖被移除消息 id（18.2/18.3）。"""
    memory = MemoryModule(
        retention_limit=2, summarizer=_ok_summarizer, clock=_monotonic_clock()
    )
    memory.append("s", "u1", "a1")  # short=[u1,a1] len2 <=2
    removed_ids_before = [m.message_id for m in memory.short_term_messages("s")]

    memory.append("s", "u2", "a2")  # short 临时变 4 -> 归档最早 2 条 -> short=[u2,a2]
    short = memory.short_term_messages("s")
    long_term = memory.long_term_summaries("s")

    assert len(short) == 2
    assert [m.content for m in short] == ["u2", "a2"]
    assert len(long_term) == 1
    # 归档覆盖的正是被移除的最早 2 条消息 id。
    assert set(long_term[0].covered_message_ids) == set(removed_ids_before)
    assert long_term[0].session_id == "s"
    assert long_term[0].summary_text  # 非空


def test_summarizer_failure_keeps_messages_and_records_failure():
    """总结失败：消息保留在短期记忆（不丢失），记录失败，不抛异常（18.4）。"""
    memory = MemoryModule(
        retention_limit=2, summarizer=_always_fail_summarizer, clock=_monotonic_clock()
    )
    memory.append("s", "u1", "a1")
    # 触发溢出归档，但总结失败。
    memory.append("s", "u2", "a2")

    short = memory.short_term_messages("s")
    # 消息全部保留（4 条），未丢失；短期暂时超过上限是 18.4 的允许权衡。
    assert len(short) == 4
    assert [m.content for m in short] == ["u1", "a1", "u2", "a2"]
    # 未写入任何长期总结。
    assert memory.long_term_summaries("s") == []
    # 失败被记录。
    assert memory.last_summarize_error is not None
    assert isinstance(memory.last_summarize_error, MemorySummarizeFailedError)
    assert len(memory.failures) == 1


def test_session_isolation_two_sessions():
    """两个 Session 互不串扰（18.6）。"""
    memory = MemoryModule(retention_limit=4, summarizer=_ok_summarizer)
    memory.append("s1", "u1", "a1")
    memory.append("s2", "u2", "a2")
    memory.append("s1", "u3", "a3")

    s1 = memory.short_term_messages("s1")
    s2 = memory.short_term_messages("s2")
    assert [m.content for m in s1] == ["u1", "a1", "u3", "a3"]
    assert [m.content for m in s2] == ["u2", "a2"]
    assert all(m.session_id == "s1" for m in s1)
    assert all(m.session_id == "s2" for m in s2)


def test_load_unknown_session_returns_empty():
    """未知 Session：空短期记忆、空长期记忆。"""
    memory = MemoryModule(retention_limit=3, summarizer=_ok_summarizer)
    loaded = memory.load("nope")
    assert loaded["short_term"].session_id == "nope"
    assert loaded["short_term"].messages == []
    assert loaded["long_term"] == []


def test_retention_limit_below_one_rejected():
    """retention_limit < 1 构造时拒绝（18.2，>=1 正整数）。"""
    with pytest.raises(ValueError):
        MemoryModule(retention_limit=0, summarizer=_ok_summarizer)
    with pytest.raises(ValueError):
        MemoryModule(retention_limit=-3, summarizer=_ok_summarizer)


def test_load_returns_time_ordered_short_and_long():
    """load 返回的短期消息按时间顺序；长期总结存在并归属该 Session（18.5）。"""
    memory = MemoryModule(
        retention_limit=2, summarizer=_ok_summarizer, clock=_monotonic_clock()
    )
    for i in range(4):
        memory.append("s", f"u{i}", f"a{i}")

    loaded = memory.load("s")
    short = loaded["short_term"].messages
    long_term = loaded["long_term"]
    # 短期保留最后一轮的 2 条。
    assert len(short) == 2
    for prev, curr in zip(short, short[1:]):
        assert prev.created_at <= curr.created_at
    # 已归档若干轮到长期记忆。
    assert len(long_term) >= 1
    assert all(s.session_id == "s" for s in long_term)


# --------------------------------------------------------------------------- #
# llm_summarizer 单元测试                                                       #
# --------------------------------------------------------------------------- #
class _FakeChatModel:
    """返回固定 Completion 的 Chat_Model 替身。"""

    def __init__(self, content: str):
        self._content = content
        self.seen_messages: list[ChatMessage] | None = None

    def generate(self, messages, *, temperature=None) -> Completion:
        self.seen_messages = messages
        return Completion(content=self._content, finish_reason="stop")


class _RaisingChatModel:
    """generate 抛错的 Chat_Model 替身。"""

    def generate(self, messages, *, temperature=None) -> Completion:
        raise RuntimeError("LLM 调用失败")


class _FakePromptModule:
    """返回带 .text 属性对象的 Prompt_Module 替身。"""

    def build(self, template_name, variables=None):
        class _P:
            text = f"[system:{template_name}]"

        return _P()


def _msgs() -> list[Message]:
    return [
        Message(session_id="s", role=Role.USER, content="问题1"),
        Message(session_id="s", role=Role.ASSISTANT, content="回答1"),
    ]


def test_llm_summarizer_returns_completion_content():
    """llm_summarizer：Chat_Model 返回内容时产出该内容。"""
    chat = _FakeChatModel("摘要X")
    summarizer = llm_summarizer(chat, _FakePromptModule())
    result = summarizer(_msgs())
    assert result == "摘要X"
    # system 提示词来自模板，user 消息携带历史。
    assert chat.seen_messages is not None
    assert chat.seen_messages[0].role == "system"
    assert chat.seen_messages[1].role == "user"


def test_llm_summarizer_raises_on_chat_error():
    """llm_summarizer：Chat_Model 抛错时总结器抛 MemorySummarizeFailedError。"""
    summarizer = llm_summarizer(_RaisingChatModel(), _FakePromptModule())
    with pytest.raises(MemorySummarizeFailedError):
        summarizer(_msgs())


def test_llm_summarizer_raises_on_empty_summary():
    """llm_summarizer：Chat_Model 返回空摘要时抛 MemorySummarizeFailedError。"""
    summarizer = llm_summarizer(_FakeChatModel("   "), _FakePromptModule())
    with pytest.raises(MemorySummarizeFailedError):
        summarizer(_msgs())


def test_memory_module_with_failing_llm_summarizer_keeps_messages():
    """集成：MemoryModule 使用会失败的 llm_summarizer 时，溢出消息保留并记录失败（18.4）。"""
    summarizer = llm_summarizer(_RaisingChatModel(), _FakePromptModule())
    memory = MemoryModule(
        retention_limit=2, summarizer=summarizer, clock=_monotonic_clock()
    )
    memory.append("s", "u1", "a1")
    memory.append("s", "u2", "a2")  # 触发归档但 LLM 失败
    assert len(memory.short_term_messages("s")) == 4
    assert memory.long_term_summaries("s") == []
    assert memory.last_summarize_error is not None

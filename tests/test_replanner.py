"""Replanner_Agent 测试（任务 13.5–13.7，Req 13.1–13.6）。

覆盖：
- evaluate 单元测试（Req 13.2–13.6）：
  - COMPLETED（全成功，默认评估器）→ summary 非空、终止（13.2）。
  - CONTINUE → 无 new_plan、非终止（13.3）。
  - REPLAN 未达上限 → replan_count + 1、new_plan 非空、非终止（13.4）。
  - REPLAN 达上限 → 终止并含"因达到最大重规划次数而未完成"说明（13.5）。
  - 评估器抛错（GenerationError）→ 终止、保留已执行结果、含失败原因说明、不崩溃（13.6）。
  - max_replan_count 钳制到 [1, 50]。
- 属性测试（Hypothesis, max_examples>=100, deadline=None）：
  - Property 16（13.6）：任意 (计划, 执行结果) 组合下，verdict 恒为三态之一。
  - Property 17（13.7）：持续 REPLAN 时重规划次数恒不超过上限，达上限终止且含达上限说明。

测试以离线确定性替身隔离全部外部依赖，无任何网络调用。
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.agents.executor import ExecutionOutcome
from ghost_agent.agents.replanner import (
    EVALUATION_FAILED_PREFIX,
    REPLAN_LIMIT_NOTE,
    ReplannerAgent,
    ReplanResult,
)
from ghost_agent.models.errors import GenerationError
from ghost_agent.models.plan import (
    Plan,
    ReplanVerdict,
    Step,
    StepFailure,
    StepResult,
)


# --------------------------------------------------------------------------- #
# 工厂 / 辅助                                                                   #
# --------------------------------------------------------------------------- #
def _plan(n: int = 2, *, grounded: bool = True) -> Plan:
    """构造含 n 个连续步骤（order 0..n-1）的计划。"""
    return Plan(
        grounded=grounded,
        steps=[
            Step(order=i, tool_name="query_cls_log", goal=f"步骤{i}")
            for i in range(n)
        ],
    )


def _outcome(
    *,
    completed: bool = False,
    failure: StepFailure | None = None,
    results: list[StepResult] | None = None,
) -> ExecutionOutcome:
    return ExecutionOutcome(
        completed=completed,
        failure=failure,
        results=list(results or []),
    )


def _const_evaluator(verdict: ReplanVerdict):
    """返回一个忽略输入、恒返回给定 verdict 的评估器。"""

    def _ev(plan, outcome):
        return verdict

    return _ev


# =========================================================================== #
# 13.5 — evaluate 单元测试                                                      #
# =========================================================================== #
def test_completed_verdict_terminates_with_summary():
    """全成功（默认评估器）→ COMPLETED、summary 非空、终止（13.2）。"""
    agent = ReplannerAgent()
    outcome = _outcome(
        completed=True,
        results=[StepResult(step_id="s0", tool_response={"ok": True})],
    )

    result = agent.evaluate(_plan(), outcome, replan_count=0)

    assert isinstance(result, ReplanResult)
    assert result.verdict is ReplanVerdict.COMPLETED
    assert result.is_terminal
    assert result.summary is not None
    # 基于已执行结果生成总结（已执行操作记录非占位）。
    assert "s0" in result.summary.executed_actions
    assert result.new_plan is None
    assert result.replan_count == 0


def test_continue_verdict_no_new_plan_not_terminal():
    """剩余计划仍适用 → CONTINUE、无 new_plan、非终止（13.3）。"""
    agent = ReplannerAgent()
    # 默认评估器：未完成且无失败 → CONTINUE。
    outcome = _outcome(completed=False, failure=None)

    result = agent.evaluate(_plan(), outcome, replan_count=2)

    assert result.verdict is ReplanVerdict.CONTINUE
    assert result.new_plan is None
    assert result.summary is None
    assert not result.is_terminal
    # CONTINUE 不改变重规划次数。
    assert result.replan_count == 2


def test_replan_verdict_below_cap_increments_and_returns_new_plan():
    """剩余计划不再适用且未达上限 → REPLAN、replan_count+1、new_plan 非空（13.4）。"""
    agent = ReplannerAgent(max_replan_count=10)
    outcome = _outcome(failure=StepFailure(step_id="s1", failure_reason="工具失败"))

    result = agent.evaluate(_plan(), outcome, replan_count=3)

    assert result.verdict is ReplanVerdict.REPLAN
    assert result.replan_count == 4
    assert result.new_plan is not None
    # 新计划从第一步开始（order 0），且应为可执行计划。
    assert result.new_plan.steps[0].order == 0
    assert result.summary is None
    assert not result.is_terminal


def test_replan_at_cap_terminates_with_limit_note():
    """重规划次数达上限 → 终止并生成含达上限说明的总结（13.5）。"""
    agent = ReplannerAgent(max_replan_count=5)
    outcome = _outcome(
        failure=StepFailure(step_id="s1", failure_reason="仍失败"),
        results=[StepResult(step_id="s0", tool_response="日志若干")],
    )

    # replan_count=4，本次 REPLAN 自增至 5 == 上限 → 终止。
    result = agent.evaluate(_plan(), outcome, replan_count=4)

    assert result.is_terminal
    assert result.summary is not None
    assert REPLAN_LIMIT_NOTE in result.summary.root_cause
    # 重规划次数不超过上限。
    assert result.replan_count == 5
    assert result.replan_count <= agent.max_replan_count
    # 终止时不再下发新计划。
    assert result.new_plan is None
    # 保留已执行结果。
    assert "s0" in result.summary.executed_actions


def test_evaluator_error_terminates_preserving_results():
    """评估器抛 GenerationError → 终止、保留已执行结果、含失败原因说明、不崩溃（13.6）。"""

    def _boom(plan, outcome):
        raise GenerationError("模型评估失败")

    agent = ReplannerAgent(evaluator=_boom)
    preserved = [
        StepResult(step_id="s0", tool_response="结果A"),
        StepResult(step_id="s1", tool_response="结果B"),
    ]
    outcome = _outcome(results=preserved)

    result = agent.evaluate(_plan(), outcome, replan_count=2)

    # 不抛出，终止流程。
    assert result.is_terminal
    assert result.summary is not None
    # 含评估失败原因说明。
    assert EVALUATION_FAILED_PREFIX in result.summary.root_cause
    assert "模型评估失败" in result.summary.root_cause
    # 保留已执行步骤的工具结果。
    assert "s0" in result.summary.executed_actions
    assert "s1" in result.summary.executed_actions
    # verdict 仍为三态之一（Property 16 封闭）。
    assert result.verdict in set(ReplanVerdict)


def test_replanner_error_during_new_plan_generation_terminates():
    """生成新计划过程中出错 → 按 13.6 终止、保留已执行结果、含失败原因说明。"""

    def _bad_replanner(plan, outcome):
        raise GenerationError("生成新计划失败")

    agent = ReplannerAgent(
        evaluator=_const_evaluator(ReplanVerdict.REPLAN),
        replanner=_bad_replanner,
        max_replan_count=10,
    )
    outcome = _outcome(results=[StepResult(step_id="s0", tool_response="x")])

    result = agent.evaluate(_plan(), outcome, replan_count=1)

    assert result.is_terminal
    assert result.summary is not None
    assert EVALUATION_FAILED_PREFIX in result.summary.root_cause
    assert "生成新计划失败" in result.summary.root_cause


def test_illegal_evaluator_return_handled_as_error():
    """评估器返回非法值（非三态）→ 按 13.6 终止，verdict 仍为三态之一（Property 16）。"""

    def _illegal(plan, outcome):
        return "NOT_A_VERDICT"

    agent = ReplannerAgent(evaluator=_illegal)
    result = agent.evaluate(_plan(), _outcome(), replan_count=0)

    assert result.is_terminal
    assert result.verdict in set(ReplanVerdict)


def test_negative_replan_count_clamped_to_zero():
    """传入负的 replan_count 被钳制为 0。"""
    agent = ReplannerAgent()
    result = agent.evaluate(_plan(), _outcome(completed=True), replan_count=-5)
    assert result.replan_count == 0


def test_max_replan_count_clamped_to_range():
    """max_replan_count 钳制到 [1, 50]。"""
    assert ReplannerAgent(max_replan_count=0).max_replan_count == 1
    assert ReplannerAgent(max_replan_count=999).max_replan_count == 50
    assert ReplannerAgent(max_replan_count=7).max_replan_count == 7


def test_default_max_replan_count_from_settings():
    """未显式指定时取配置默认值（默认 10）。"""
    assert ReplannerAgent().max_replan_count == 10


# =========================================================================== #
# 13.6 — 属性测试 Property 16                                                    #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 16: 对任意当前计划与执行结果组合，Replanner_Agent 输出的评估结果恒为 {任务已完成, 任务未完成且剩余计划仍适用, 任务未完成且剩余计划不再适用} 三个取值之一。
# Validates: Requirements 13.1

_VERDICTS = set(ReplanVerdict)


@st.composite
def _outcomes(draw) -> ExecutionOutcome:
    """生成随机执行态汇总：随机 completed / 失败信息 / 已执行结果列表。"""
    completed = draw(st.booleans())
    has_failure = draw(st.booleans())
    failure = (
        StepFailure(
            step_id=draw(st.text(min_size=1, max_size=8)),
            failure_reason=draw(st.text(min_size=1, max_size=16)),
        )
        if has_failure
        else None
    )
    results = draw(
        st.lists(
            st.builds(
                StepResult,
                step_id=st.text(min_size=1, max_size=8),
                tool_response=st.one_of(
                    st.text(max_size=16),
                    st.integers(),
                    st.dictionaries(
                        st.text(min_size=1, max_size=4),
                        st.integers(),
                        max_size=3,
                    ),
                ),
            ),
            max_size=4,
        )
    )
    return ExecutionOutcome(completed=completed, failure=failure, results=results)


@settings(max_examples=200, deadline=None)
@given(
    n_steps=st.integers(min_value=1, max_value=6),
    grounded=st.booleans(),
    outcome=_outcomes(),
    replan_count=st.integers(min_value=0, max_value=60),
)
def test_property_16_default_evaluator_verdict_closed(
    n_steps: int, grounded: bool, outcome: ExecutionOutcome, replan_count: int
):
    """Property 16：默认评估器下，任意 (计划, 执行结果) 组合的 verdict 恒为三态之一。"""
    agent = ReplannerAgent()
    result = agent.evaluate(_plan(n_steps, grounded=grounded), outcome, replan_count=replan_count)
    assert result.verdict in _VERDICTS


@settings(max_examples=100, deadline=None)
@given(
    forced=st.sampled_from(list(ReplanVerdict)),
    outcome=_outcomes(),
    replan_count=st.integers(min_value=0, max_value=60),
)
def test_property_16_injected_evaluator_verdict_closed(
    forced: ReplanVerdict, outcome: ExecutionOutcome, replan_count: int
):
    """Property 16：注入任意（来自枚举的）评估器返回值，verdict 仍恒为三态之一。"""
    agent = ReplannerAgent(evaluator=_const_evaluator(forced))
    result = agent.evaluate(_plan(2), outcome, replan_count=replan_count)
    assert result.verdict in _VERDICTS


# =========================================================================== #
# 13.7 — 属性测试 Property 17                                                    #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 17: 对任意持续触发重规划的排查流程，重规划次数不超过配置的最大重规划次数上限（默认 10，范围 1–50），且达上限时终止流程并生成含"因达到最大重规划次数而未完成"说明的分析结果总结。
# Validates: Requirements 13.5


@settings(max_examples=200, deadline=None)
@given(
    max_replan_count=st.integers(min_value=1, max_value=50),
    n_steps=st.integers(min_value=1, max_value=5),
)
def test_property_17_continuous_replan_respects_upper_bound(
    max_replan_count: int, n_steps: int
):
    """Property 17：持续 REPLAN 时重规划次数恒不超过上限，达上限终止且含达上限说明。"""
    # 注入恒返回 REPLAN 的评估器，驱动"持续触发重规划"。
    agent = ReplannerAgent(
        evaluator=_const_evaluator(ReplanVerdict.REPLAN),
        max_replan_count=max_replan_count,
    )
    plan = _plan(n_steps)
    outcome = _outcome(
        failure=StepFailure(step_id="s0", failure_reason="持续失败"),
        results=[StepResult(step_id="s0", tool_response="部分结果")],
    )

    replan_count = 0
    iterations = 0
    # 循环上界防御：理论上至多 max_replan_count 次即终止。
    max_iterations = max_replan_count + 5
    result: ReplanResult | None = None
    while iterations < max_iterations:
        iterations += 1
        result = agent.evaluate(plan, outcome, replan_count=replan_count)
        # 不变式：重规划次数恒不超过配置上限。
        assert result.replan_count <= max_replan_count
        replan_count = result.replan_count
        if result.is_terminal:
            break
        # 非终止的 REPLAN 应下发新计划，并从第一步执行。
        assert result.verdict is ReplanVerdict.REPLAN
        assert result.new_plan is not None
        plan = result.new_plan

    assert result is not None
    # 终止发生。
    assert result.is_terminal
    # 终止时重规划次数恰为上限。
    assert result.replan_count == max_replan_count
    # 终止总结含"因达到最大重规划次数而未完成"说明。
    assert result.summary is not None
    assert REPLAN_LIMIT_NOTE in result.summary.root_cause


@settings(max_examples=100, deadline=None)
@given(
    max_replan_count=st.integers(min_value=1, max_value=50),
    n_steps=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_17_terminates_at_exactly_cap_iterations(
    max_replan_count: int, n_steps: int, seed: int
):
    """Property 17（计数语义）：从 0 起持续 REPLAN，恰在第 max 次评估终止，计数恒不超过上限。"""
    agent = ReplannerAgent(
        evaluator=_const_evaluator(ReplanVerdict.REPLAN),
        max_replan_count=max_replan_count,
    )
    plan = _plan(n_steps)
    outcome = _outcome(failure=StepFailure(step_id="s0", failure_reason="x"))

    replan_count = 0
    terminations = 0
    evaluations = 0
    result: ReplanResult | None = None
    while True:
        evaluations += 1
        result = agent.evaluate(plan, outcome, replan_count=replan_count)
        assert result.replan_count <= max_replan_count
        replan_count = result.replan_count
        if result.is_terminal:
            terminations += 1
            break
        plan = result.new_plan
        # 安全阀，避免潜在死循环。
        assert evaluations <= max_replan_count + 5

    # 恰好经过 max_replan_count 次评估才终止（每次自增 1，达上限即止）。
    assert evaluations == max_replan_count
    assert terminations == 1
    assert result is not None and result.replan_count == max_replan_count

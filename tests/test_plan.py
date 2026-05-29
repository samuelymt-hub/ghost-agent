"""Plan / Step / StepResult / StepFailure / ReplanVerdict 单元测试 (Req 11.2, 12.1, 13.1)。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ghost_agent.models import (
    Plan,
    ReplanVerdict,
    Step,
    StepFailure,
    StepResult,
    StepStatus,
)


def _step(order: int, *, tool: str = "query_cls_log", goal: str = "查日志") -> Step:
    return Step(order=order, tool_name=tool, goal=goal)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


def test_step_happy_path_defaults_to_pending() -> None:
    s = _step(0)
    assert s.order == 0
    assert s.status is StepStatus.PENDING
    assert isinstance(s.step_id, str) and s.step_id


def test_step_negative_order_rejected() -> None:
    with pytest.raises(ValidationError):
        Step(order=-1, tool_name="t", goal="g")


def test_step_empty_tool_name_rejected() -> None:
    with pytest.raises(ValidationError):
        Step(order=0, tool_name="", goal="g")


def test_step_empty_goal_rejected() -> None:
    with pytest.raises(ValidationError):
        Step(order=0, tool_name="t", goal="")


def test_step_invalid_status_rejected() -> None:
    with pytest.raises(ValidationError):
        Step(order=0, tool_name="t", goal="g", status="UNKNOWN")


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def test_plan_happy_path_with_continuous_orders() -> None:
    plan = Plan(
        grounded=True,
        steps=[_step(0), _step(1), _step(2)],
    )
    assert plan.grounded is True
    assert [s.order for s in plan.steps] == [0, 1, 2]
    assert isinstance(plan.plan_id, str) and plan.plan_id


def test_plan_requires_at_least_one_step() -> None:
    with pytest.raises(ValidationError):
        Plan(grounded=True, steps=[])


def test_plan_rejects_non_continuous_orders() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Plan(grounded=True, steps=[_step(0), _step(2)])
    assert "order" in str(exc_info.value).lower()


def test_plan_rejects_orders_not_starting_from_zero() -> None:
    with pytest.raises(ValidationError):
        Plan(grounded=True, steps=[_step(1), _step(2)])


def test_plan_rejects_duplicate_step_ids() -> None:
    s = _step(0)
    s2 = Step(
        step_id=s.step_id,
        order=1,
        tool_name="t",
        goal="g",
    )
    with pytest.raises(ValidationError) as exc_info:
        Plan(grounded=False, steps=[s, s2])
    assert "step_id" in str(exc_info.value)


def test_plan_grounded_field_required() -> None:
    with pytest.raises(ValidationError):
        Plan(steps=[_step(0)])  # type: ignore[call-arg]


def test_plan_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Plan(grounded=True, steps=[_step(0)], unknown="x")  # type: ignore[call-arg]


def test_plan_json_round_trip() -> None:
    original = Plan(grounded=False, steps=[_step(0), _step(1)])
    raw = original.model_dump_json()
    payload = json.loads(raw)
    assert payload["steps"][0]["status"] == "PENDING"

    restored = Plan.model_validate_json(raw)
    assert restored == original


# ---------------------------------------------------------------------------
# StepResult / StepFailure
# ---------------------------------------------------------------------------


def test_step_result_accepts_arbitrary_response() -> None:
    r = StepResult(step_id="s1", tool_response={"hits": [1, 2, 3]})
    assert r.tool_response == {"hits": [1, 2, 3]}


def test_step_result_missing_step_id_rejected() -> None:
    with pytest.raises(ValidationError):
        StepResult(tool_response="x")  # type: ignore[call-arg]


def test_step_failure_requires_non_empty_reason() -> None:
    with pytest.raises(ValidationError):
        StepFailure(step_id="s1", failure_reason="")


def test_step_failure_happy_path() -> None:
    f = StepFailure(step_id="s1", failure_reason="timeout")
    assert f.failure_reason == "timeout"


# ---------------------------------------------------------------------------
# ReplanVerdict
# ---------------------------------------------------------------------------


def test_replan_verdict_values() -> None:
    assert ReplanVerdict.COMPLETED.value == "COMPLETED"
    assert ReplanVerdict.CONTINUE.value == "CONTINUE"
    assert ReplanVerdict.REPLAN.value == "REPLAN"
    assert {v.value for v in ReplanVerdict} == {"COMPLETED", "CONTINUE", "REPLAN"}

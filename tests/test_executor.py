"""Executor_Agent 测试 (Req 12.1–12.5)。

包含：
- 任务 13.4 属性测试 Property 15（执行顺序与失败短路，Hypothesis，
  ``max_examples>=100``）。
- 单元测试：单步成功/失败/超时、计划全部成功、失败短路、乱序按 order 升序执行。

测试通过**可注入的假工具集**隔离外部依赖：每个步骤指向唯一工具名，假工具集
按工具名决定成功（返回值）、失败（抛错）或挂起（触发超时），从而在完全离线、
确定性的前提下覆盖 Executor 的执行/短路语义。
"""

from __future__ import annotations

import time
from typing import Any, Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.agents.executor import (
    TIMEOUT_REASON,
    ExecutionOutcome,
    ExecutorAgent,
)
from ghost_agent.core.tool_registry import ToolRegistry
from ghost_agent.models import (
    Plan,
    ParamDef,
    ParamType,
    Step,
    StepFailure,
    StepResult,
    StepStatus,
    ToolDefinition,
    ToolSource,
)


# --------------------------------------------------------------------------- #
# 测试辅助：可注入行为的假工具集                                                 #
# --------------------------------------------------------------------------- #
def _tool_def(name: str) -> ToolDefinition:
    """构造一个无必填参数的工具定义（便于以空参调用）。"""
    return ToolDefinition(
        name=name,
        description="测试工具",
        params=[ParamDef(name="query", type=ParamType.STRING, required=False)],
        source=ToolSource.BUILTIN,
    )


def _build_registry(handlers: dict[str, Callable[[dict[str, Any]], Any]]) -> ToolRegistry:
    """以 ``{tool_name: handler}`` 构造一个仅含这些工具的注册表。"""
    registry = ToolRegistry()
    for name, handler in handlers.items():
        registry.register(_tool_def(name), handler)
    return registry


def _success_handler(value: Any) -> Callable[[dict[str, Any]], Any]:
    def _h(params: dict[str, Any]) -> Any:
        return value

    return _h


def _error_handler(message: str = "boom") -> Callable[[dict[str, Any]], Any]:
    def _h(params: dict[str, Any]) -> Any:
        raise RuntimeError(message)

    return _h


def _sleep_handler(seconds: float, value: Any = "late") -> Callable[[dict[str, Any]], Any]:
    def _h(params: dict[str, Any]) -> Any:
        time.sleep(seconds)
        return value

    return _h


def _make_plan(tool_names: list[str], *, grounded: bool = True) -> Plan:
    """按给定工具名列表构造连续 order（0..n-1）的执行计划。"""
    steps = [
        Step(order=i, tool_name=name, goal=f"步骤 {i}")
        for i, name in enumerate(tool_names)
    ]
    return Plan(grounded=grounded, steps=steps)


# --------------------------------------------------------------------------- #
# 单元测试：单步原语 execute_step                                                #
# --------------------------------------------------------------------------- #
def test_execute_step_success_returns_step_result() -> None:
    """Req 12.3：工具成功返回时记录含步骤标识与工具响应的 StepResult。"""
    registry = _build_registry({"tool_ok": _success_handler({"data": 42})})
    agent = ExecutorAgent(tool_registry=registry, tool_call_timeout=5.0)
    step = Step(order=0, tool_name="tool_ok", goal="查询")

    result = agent.execute_step(step, {"query": "x"})

    assert isinstance(result, StepResult)
    assert result.step_id == step.step_id
    assert result.tool_response == {"data": 42}


def test_execute_step_tool_error_returns_step_failure_without_raising() -> None:
    """Req 12.4：工具返回错误时记录含步骤标识与原因的 StepFailure，且不抛出。"""
    registry = _build_registry({"tool_err": _error_handler("连接被拒绝")})
    agent = ExecutorAgent(tool_registry=registry, tool_call_timeout=5.0)
    step = Step(order=0, tool_name="tool_err", goal="查询")

    result = agent.execute_step(step)

    assert isinstance(result, StepFailure)
    assert result.step_id == step.step_id
    assert "连接被拒绝" in result.failure_reason


def test_execute_step_unknown_tool_returns_step_failure() -> None:
    """Req 12.4：工具名不存在时（ToolNotFoundError）归一化为 StepFailure，不抛出。"""
    registry = _build_registry({"tool_ok": _success_handler("ok")})
    agent = ExecutorAgent(tool_registry=registry, tool_call_timeout=5.0)
    step = Step(order=0, tool_name="missing_tool", goal="查询")

    result = agent.execute_step(step)

    assert isinstance(result, StepFailure)
    assert result.step_id == step.step_id
    assert result.failure_reason


def test_execute_step_timeout_judged_as_failure() -> None:
    """Req 12.5：工具调用超时被判定为该步骤的工具调用失败。"""
    registry = _build_registry({"tool_slow": _sleep_handler(2.0)})
    agent = ExecutorAgent(tool_registry=registry, tool_call_timeout=0.2)
    step = Step(order=0, tool_name="tool_slow", goal="慢查询")

    result = agent.execute_step(step)

    assert isinstance(result, StepFailure)
    assert result.step_id == step.step_id
    assert TIMEOUT_REASON in result.failure_reason


# --------------------------------------------------------------------------- #
# 单元测试：计划级编排 execute_plan                                              #
# --------------------------------------------------------------------------- #
def test_execute_plan_all_success_completes_in_order() -> None:
    """Req 12.1/12.3：全部成功时按顺序记录结果，步骤状态置为 SUCCESS。"""
    registry = _build_registry(
        {
            "tool_0": _success_handler("r0"),
            "tool_1": _success_handler("r1"),
            "tool_2": _success_handler("r2"),
        }
    )
    agent = ExecutorAgent(tool_registry=registry, tool_call_timeout=5.0)
    plan = _make_plan(["tool_0", "tool_1", "tool_2"])

    outcome = agent.execute_plan(plan)

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.completed is True
    assert outcome.failure is None
    assert outcome.executed_orders == [0, 1, 2]
    assert [r.tool_response for r in outcome.results] == ["r0", "r1", "r2"]
    assert [r.step_id for r in outcome.results] == [s.step_id for s in plan.steps]
    assert all(s.status is StepStatus.SUCCESS for s in plan.steps)


def test_execute_plan_failure_short_circuits_subsequent_steps() -> None:
    """Req 12.4：某步失败后暂停后续步骤；失败信息含该步骤标识，后续步骤未执行。"""
    executed: list[str] = []

    def _track(name: str, *, fail: bool = False) -> Callable[[dict[str, Any]], Any]:
        def _h(params: dict[str, Any]) -> Any:
            executed.append(name)
            if fail:
                raise RuntimeError("step failed")
            return name

        return _h

    registry = _build_registry(
        {
            "tool_0": _track("tool_0"),
            "tool_1": _track("tool_1", fail=True),
            "tool_2": _track("tool_2"),
        }
    )
    agent = ExecutorAgent(tool_registry=registry, tool_call_timeout=5.0)
    plan = _make_plan(["tool_0", "tool_1", "tool_2"])

    outcome = agent.execute_plan(plan)

    assert outcome.completed is False
    assert outcome.failure is not None
    # 失败步骤为 order=1 的步骤。
    failed_step = plan.steps[1]
    assert outcome.failure.step_id == failed_step.step_id
    # order=2 的工具从未被调用（短路）。
    assert executed == ["tool_0", "tool_1"]
    assert outcome.executed_orders == [0, 1]
    assert len(outcome.results) == 1
    # 步骤状态：成功步 SUCCESS、失败步 FAILED、后续步保持 PENDING。
    assert plan.steps[0].status is StepStatus.SUCCESS
    assert plan.steps[1].status is StepStatus.FAILED
    assert plan.steps[2].status is StepStatus.PENDING


def test_execute_plan_iterates_in_ascending_order_when_steps_shuffled() -> None:
    """Req 12.1：即使步骤列表乱序提供，也按 order 升序执行。

    Plan 模型校验要求 ``steps`` 列表本身按 order=0..n-1 排列，因此正常构造无法
    得到乱序列表；此处用 ``model_construct`` 绕过校验构造乱序列表，以验证
    ``execute_plan`` 内部按 ``order`` 升序排序的防御性逻辑确实生效。"""
    call_order: list[int] = []

    def _record(order: int) -> Callable[[dict[str, Any]], Any]:
        def _h(params: dict[str, Any]) -> Any:
            call_order.append(order)
            return order

        return _h

    registry = _build_registry(
        {f"tool_{i}": _record(i) for i in range(4)}
    )
    agent = ExecutorAgent(tool_registry=registry, tool_call_timeout=5.0)
    # 以乱序提供步骤列表（order 仍为 0..3），绕过 Plan 模型的连续性校验。
    steps = [
        Step(order=2, tool_name="tool_2", goal="s2"),
        Step(order=0, tool_name="tool_0", goal="s0"),
        Step(order=3, tool_name="tool_3", goal="s3"),
        Step(order=1, tool_name="tool_1", goal="s1"),
    ]
    plan = Plan.model_construct(plan_id="p-shuffled", grounded=True, steps=steps)

    outcome = agent.execute_plan(plan)

    assert outcome.completed is True
    assert call_order == [0, 1, 2, 3]
    assert outcome.executed_orders == [0, 1, 2, 3]


def test_execute_plan_params_for_supplies_per_step_params() -> None:
    """params_for 回调为每个步骤提供工具调用参数。"""
    seen: dict[str, dict[str, Any]] = {}

    def _capture(name: str) -> Callable[[dict[str, Any]], Any]:
        def _h(params: dict[str, Any]) -> Any:
            seen[name] = params
            return name

        return _h

    registry = _build_registry({"tool_0": _capture("tool_0"), "tool_1": _capture("tool_1")})
    agent = ExecutorAgent(tool_registry=registry, tool_call_timeout=5.0)
    plan = _make_plan(["tool_0", "tool_1"])

    outcome = agent.execute_plan(
        plan, params_for=lambda step: {"query": f"q-{step.order}"}
    )

    assert outcome.completed is True
    assert seen["tool_0"] == {"query": "q-0"}
    assert seen["tool_1"] == {"query": "q-1"}


# --------------------------------------------------------------------------- #
# 属性测试 Property 15：执行顺序与失败短路                                        #
# --------------------------------------------------------------------------- #
@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=15),
    fail_index=st.integers(min_value=0, max_value=30),
)
def test_property_15_execution_order_and_failure_short_circuit(
    n: int, fail_index: int
) -> None:
    """Feature: intelligent-oncall-agent, Property 15: 对任意执行计划及任一失败步骤位置，
    Executor_Agent 按步骤序号升序逐步执行；一旦某步骤工具调用失败，该步骤之后的步骤不再
    被执行，且失败信息包含所属步骤标识。

    **Validates: Requirements 12.1, 12.4**
    """
    # fail_index 取 [0, n] —— f == n 表示"无失败步骤"。
    f = min(fail_index, n)

    executed: list[int] = []

    def _handler(order: int, *, fail: bool) -> Callable[[dict[str, Any]], Any]:
        def _h(params: dict[str, Any]) -> Any:
            executed.append(order)
            if fail:
                raise RuntimeError(f"step {order} failed")
            return f"resp-{order}"

        return _h

    # 每个步骤指向唯一工具名 tool_{i}；仅 order==f 的工具会失败。
    handlers = {
        f"tool_{i}": _handler(i, fail=(i == f)) for i in range(n)
    }
    registry = _build_registry(handlers)
    agent = ExecutorAgent(tool_registry=registry, tool_call_timeout=5.0)
    plan = _make_plan([f"tool_{i}" for i in range(n)])

    outcome = agent.execute_plan(plan)

    # executed_orders 必为 [0..k] 形式的前缀（严格升序、从 0 开始、连续）。
    assert outcome.executed_orders == list(range(len(outcome.executed_orders)))
    # 实际工具调用顺序与 executed_orders 一致（升序逐步执行，Req 12.1）。
    assert executed == outcome.executed_orders

    if f < n:
        # 存在失败步骤：在第 f 步短路。
        assert outcome.completed is False
        assert outcome.failure is not None
        assert outcome.failure.step_id == plan.steps[f].step_id
        # 失败信息包含所属步骤标识（Property 15 末句）。
        assert outcome.failure.step_id  # 非空
        # 第 f 步之后的步骤不再被执行（Req 12.4 短路）。
        assert outcome.executed_orders == list(range(f + 1))
        assert max(executed) == f
        # 成功结果恰为失败步骤之前的 f 个步骤。
        assert len(outcome.results) == f
        assert [r.step_id for r in outcome.results] == [
            plan.steps[i].step_id for i in range(f)
        ]
    else:
        # 无失败步骤：全部步骤按升序执行完毕。
        assert outcome.completed is True
        assert outcome.failure is None
        assert outcome.executed_orders == list(range(n))
        assert len(outcome.results) == n

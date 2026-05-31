"""Planner_Agent 测试（任务 13.1–13.2，Req 11.1–11.6）。

覆盖：
- ``plan`` 单元测试（Req 11.1, 11.2, 11.3, 11.4, 11.5, 11.6）：
  - query_internal_docs 返回非空处理步骤 → grounded=True，并以告警 message 调用工具（11.1）。
  - query_internal_docs 返回空 → grounded=False、生成通用计划（11.4）。
  - query_internal_docs 抛错 → grounded=False、生成通用计划（11.5）。
  - query_internal_docs 超时 → grounded=False、生成通用计划（11.5）。
  - 步骤工具名均已注册、目标非空、序号连续、步骤数 ∈ [1, max_steps]（11.2/11.3）。
  - max_steps 钳制计划长度。
- 属性测试（Hypothesis, max_examples>=100, deadline=None）：
  - Property 14（13.2）：对任意告警、任意 max_steps∈[1,30] 与任意已注册工具子集，
    生成计划满足步骤数边界、序号连续、工具名属于已注册集合且目标非空。

测试以离线替身隔离全部外部依赖，无任何网络调用：默认确定性 plan_builder 不触达 LLM，
query_internal_docs 后端以可注入回调替身（返回 / 空 / 抛错 / 睡眠超时）提供。
"""

from __future__ import annotations

import time

from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.agents.planner import DOCS_TOOL_NAME, PlannerAgent
from ghost_agent.core.tool_registry import ToolRegistry, build_default_registry
from ghost_agent.models.plan import Plan
from ghost_agent.models.tool import ParamDef, ParamType, ToolDefinition, ToolSource
from ghost_agent.models.troubleshooting_task import AlarmInfo


# --------------------------------------------------------------------------- #
# 替身与辅助构造                                                                #
# --------------------------------------------------------------------------- #
class _DocsBackend:
    """query_internal_docs 后端替身：可配置返回值 / 抛错 / 睡眠，并记录调用参数。"""

    def __init__(self, *, result=None, error: Exception | None = None, sleep: float = 0.0) -> None:
        self._result = result
        self._error = error
        self._sleep = sleep
        self.calls: list[dict] = []

    def __call__(self, params: dict):
        self.calls.append(dict(params))
        if self._sleep:
            time.sleep(self._sleep)
        if self._error is not None:
            raise self._error
        return self._result


def _alarm(message: str = "磁盘使用率超过 90%") -> AlarmInfo:
    return AlarmInfo(source="prometheus", level="CRITICAL", message=message)


def _registry_from_names(tool_names: list[str]) -> ToolRegistry:
    """用给定工具名集合构造一个 ToolRegistry（每个工具带可选 query 参数与无副作用句柄）。"""
    registry = ToolRegistry()
    for name in tool_names:
        registry.register(
            ToolDefinition(
                name=name,
                description=f"工具 {name}",
                params=[ParamDef(name="query", type=ParamType.STRING, required=False)],
                source=ToolSource.BUILTIN,
            ),
            (lambda params, _n=name: {"status": "ok", "tool": _n}),
        )
    return registry


# =========================================================================== #
# 13.1 — plan 单元测试                                                          #
# =========================================================================== #
def test_plan_queries_internal_docs_with_alarm_message():
    """Req 11.1：plan 启动后以告警 message 调用 query_internal_docs。"""
    backend = _DocsBackend(result={"docs": ["先检查磁盘", "再清理日志"]})
    registry = build_default_registry(docs_query=backend)
    planner = PlannerAgent(tool_registry=registry)

    planner.plan(_alarm("磁盘使用率超过 90%"))

    assert backend.calls, "应调用 query_internal_docs"
    assert backend.calls[0] == {"query": "磁盘使用率超过 90%"}


def test_plan_grounded_true_when_docs_returned():
    """Req 11.1/11.2：query_internal_docs 返回非空处理步骤 → grounded=True。"""
    backend = _DocsBackend(result={"docs": ["步骤一", "步骤二"]})
    registry = build_default_registry(docs_query=backend)
    planner = PlannerAgent(tool_registry=registry)

    plan = planner.plan(_alarm())

    assert isinstance(plan, Plan)
    assert plan.grounded is True
    assert len(plan.steps) >= 1


def test_plan_generic_when_docs_empty():
    """Req 11.4：query_internal_docs 返回空 → grounded=False，生成通用计划。"""
    backend = _DocsBackend(result={"docs": []})
    registry = build_default_registry(docs_query=backend)
    planner = PlannerAgent(tool_registry=registry)

    plan = planner.plan(_alarm())

    assert plan.grounded is False
    assert len(plan.steps) >= 1


def test_plan_generic_when_docs_empty_list_result():
    """Req 11.4：返回空列表同样判定为无手册依据。"""
    backend = _DocsBackend(result=[])
    registry = build_default_registry(docs_query=backend)
    planner = PlannerAgent(tool_registry=registry)

    plan = planner.plan(_alarm())

    assert plan.grounded is False


def test_plan_generic_when_docs_query_raises():
    """Req 11.5：query_internal_docs 抛错 → grounded=False，生成通用计划。"""
    backend = _DocsBackend(error=RuntimeError("内部文档服务不可用"))
    registry = build_default_registry(docs_query=backend)
    planner = PlannerAgent(tool_registry=registry)

    plan = planner.plan(_alarm())

    assert plan.grounded is False
    assert len(plan.steps) >= 1


def test_plan_generic_when_docs_query_times_out():
    """Req 11.5：query_internal_docs 超时 → grounded=False，生成通用计划。"""
    backend = _DocsBackend(result={"docs": ["x"]}, sleep=0.3)
    registry = build_default_registry(docs_query=backend)
    planner = PlannerAgent(tool_registry=registry, query_timeout=0.05)

    plan = planner.plan(_alarm())

    assert plan.grounded is False
    assert len(plan.steps) >= 1


def test_plan_generic_when_docs_tool_absent():
    """Req 11.5：注册表无 query_internal_docs → grounded=False，仍生成合法计划。"""
    registry = _registry_from_names(["query_cls_log", "send_msg"])
    planner = PlannerAgent(tool_registry=registry)

    plan = planner.plan(_alarm())

    assert plan.grounded is False
    registered = {d.name for d in registry.list_definitions()}
    assert all(s.tool_name in registered for s in plan.steps)


def test_plan_steps_are_well_formed():
    """Req 11.2/11.3：步骤工具名均已注册、目标非空、序号连续、步骤数 ∈ [1, max_steps]。"""
    backend = _DocsBackend(result={"docs": ["步骤"]})
    registry = build_default_registry(docs_query=backend)
    planner = PlannerAgent(tool_registry=registry, max_steps=10)

    plan = planner.plan(_alarm())

    registered = {d.name for d in registry.list_definitions()}
    assert 1 <= len(plan.steps) <= 10
    assert [s.order for s in plan.steps] == list(range(len(plan.steps)))
    assert all(s.tool_name in registered for s in plan.steps)
    assert all(s.goal and s.goal.strip() for s in plan.steps)


def test_plan_default_registry_uses_generic_tool_order():
    """默认注册表（4 内置工具）下生成的计划按通用排查工具顺序标注工具。"""
    registry = build_default_registry()
    planner = PlannerAgent(tool_registry=registry, max_steps=20)

    plan = planner.plan(_alarm())

    assert [s.tool_name for s in plan.steps] == [
        "query_internal_docs",
        "query_cls_log",
        "query_prometheus_alarm",
        "send_msg",
    ]


def test_plan_max_steps_clamps_length():
    """Req 11.2：max_steps 钳制计划长度（默认注册表 4 工具，max_steps=2 → 2 步）。"""
    registry = build_default_registry()
    planner = PlannerAgent(tool_registry=registry, max_steps=2)

    plan = planner.plan(_alarm())

    assert len(plan.steps) == 2
    assert [s.order for s in plan.steps] == [0, 1]


def test_plan_max_steps_clamped_to_at_least_one():
    """步骤上限被钳制到 >= 1：即便传入 0 也至少生成 1 步。"""
    registry = build_default_registry()
    planner = PlannerAgent(tool_registry=registry, max_steps=0)

    assert planner.max_steps == 1
    plan = planner.plan(_alarm())
    assert len(plan.steps) == 1


def test_plan_goal_non_empty_even_for_whitespace_message():
    """目标由非空常量前缀拼装：即便告警 message 为纯空白，步骤目标仍非空（11.3）。"""
    registry = build_default_registry()
    # message 至少 1 个字符（模型约束），此处用纯空白字符。
    planner = PlannerAgent(tool_registry=registry)

    plan = planner.plan(_alarm("   "))

    assert all(s.goal and s.goal.strip() for s in plan.steps)


# =========================================================================== #
# 13.2 — 属性测试 Property 14                                                   #
# =========================================================================== #
# 候选工具池：含 4 个内置工具 + 2 个自定义工具，覆盖"含/不含 query_internal_docs"两种情形。
_TOOL_POOL = [
    "query_internal_docs",
    "query_cls_log",
    "query_prometheus_alarm",
    "send_msg",
    "custom_tool_a",
    "custom_tool_b",
]


# Feature: intelligent-oncall-agent, Property 14: 对任意由 Planner_Agent 生成的执行计划，步骤数量介于 1 与配置的最大步骤数上限之间，步骤序号连续，且每个步骤标注的工具名均属于 Tool_Registry 已注册工具名集合并带有非空目标。
# Validates: Requirements 11.2, 11.3
@settings(max_examples=150, deadline=None)
@given(
    message=st.text(min_size=1, max_size=80),
    max_steps=st.integers(min_value=1, max_value=30),
    tool_names=st.lists(
        st.sampled_from(_TOOL_POOL), min_size=1, max_size=len(_TOOL_POOL), unique=True
    ),
)
def test_property_14_plan_step_bounds_and_legality(
    message: str, max_steps: int, tool_names: list[str]
):
    """Property 14：步骤数边界、序号连续、工具名属于已注册集合且目标非空。"""
    registry = _registry_from_names(tool_names)
    planner = PlannerAgent(tool_registry=registry, max_steps=max_steps)

    plan = planner.plan(AlarmInfo(message=message))

    registered_names = {d.name for d in registry.list_definitions()}

    # 1) 步骤数量介于 1 与配置上限之间。
    assert 1 <= len(plan.steps) <= max_steps
    # 2) 步骤序号连续（0..n-1）。
    assert [s.order for s in plan.steps] == list(range(len(plan.steps)))
    # 3) 每个步骤工具名均属于已注册工具名集合。
    assert all(s.tool_name in registered_names for s in plan.steps)
    # 4) 每个步骤目标非空。
    assert all(s.goal and s.goal.strip() for s in plan.steps)


@settings(max_examples=100, deadline=None)
@given(
    message=st.text(min_size=1, max_size=40),
    max_steps=st.integers(min_value=1, max_value=30),
)
def test_property_14_default_registry_respects_bounds(message: str, max_steps: int):
    """Property 14（默认注册表）：含 query_internal_docs 调用路径下同样满足步骤合法性。"""
    registry = build_default_registry()
    planner = PlannerAgent(tool_registry=registry, max_steps=max_steps)

    plan = planner.plan(AlarmInfo(message=message))

    registered_names = {d.name for d in registry.list_definitions()}
    assert DOCS_TOOL_NAME in registered_names
    assert 1 <= len(plan.steps) <= max_steps
    assert [s.order for s in plan.steps] == list(range(len(plan.steps)))
    assert all(s.tool_name in registered_names for s in plan.steps)
    assert all(s.goal and s.goal.strip() for s in plan.steps)

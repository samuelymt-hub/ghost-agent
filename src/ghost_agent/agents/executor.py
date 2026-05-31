"""Executor_Agent（执行 Agent，Plan-Execute 阶段，Req 12.1–12.5）。

本模块实现运维 Agent 的**执行子 Agent**，负责按 Planner_Agent 生成的执行计划
逐步调用 Tool_Registry 中的工具并记录结果/失败信息：

* **逐步执行（Req 12.1）**：从执行计划的第一个步骤开始，按步骤的标注顺序
  （``Step.order`` 升序）每次执行一个步骤。
* **工具调用（Req 12.2）**：执行当前步骤时调用该步骤标注的工具
  （``Step.tool_name``），所调用工具来自 Tool_Registry（含 query_cls_log、
  query_prometheus_alarm、send_msg 与 MCP 工具）。
* **成功记录（Req 12.3）**：工具调用成功返回时记录含所属步骤标识与工具响应
  内容的 :class:`StepResult`，随后将其移交 Replanner_Agent。
* **失败短路（Req 12.4）**：工具调用失败时记录含所属步骤标识与失败原因的
  :class:`StepFailure`、**暂停执行后续步骤**，并将失败结果移交 Replanner_Agent。
* **超时判定（Req 12.5）**：工具调用等待响应达到配置的工具调用超时时间仍未
  返回时，终止该工具调用并将其判定为该步骤的工具调用失败。

设计抉择：design.md 将 Executor_Agent 的核心契约写作
``Executor_Agent.execute(step) -> StepResult | StepFailure``，即"执行单个步骤"
的原语。本实现同时提供两层 API：

* :meth:`ExecutorAgent.execute_step` —— 单步原语（对应 design 的 ``execute(step)``），
  调用工具并归一化为 :class:`StepResult` 或 :class:`StepFailure`，**永不抛出**
  工具错误/超时（统一返回 :class:`StepFailure`）。
* :meth:`ExecutorAgent.execute_plan` —— 计划级编排：按 ``order`` 升序逐步调用
  :meth:`execute_step`，一旦某步失败即短路停止（Req 12.1/12.4，即 Property 15
  所断言的执行顺序与失败短路语义）。

超时实现：工具调用经 :meth:`ExecutorAgent._call_with_timeout` 包裹（基于
``ThreadPoolExecutor`` 的硬超时）。``shutdown(wait=False)`` 不阻塞等待可能挂起
的线程，从而保证调用方视角的超时上界（Req 12.5）。所有协作者均以依赖注入方式
提供，默认实现惰性构造、构造期不触网，便于确定性单元/属性测试。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from ghost_agent.config import get_settings
from ghost_agent.core.tool_registry import ToolRegistry, build_default_registry
from ghost_agent.models.plan import (
    Plan,
    Step,
    StepFailure,
    StepResult,
    StepStatus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutorAgent",
    "ExecutionOutcome",
    "TIMEOUT_REASON",
]

#: 工具调用超时时归入 :class:`StepFailure.failure_reason` 的判定说明（Req 12.5）。
TIMEOUT_REASON = "调用超时"


# --------------------------------------------------------------------------- #
# 返回结构                                                                      #
# --------------------------------------------------------------------------- #
class ExecutionOutcome(BaseModel):
    """计划级执行的聚合结果（供 Ops_Agent / Replanner_Agent 消费）。

    Attributes:
        results: 已成功执行步骤的 :class:`StepResult`，按执行（order 升序）顺序。
        failure: 触发短路的 :class:`StepFailure`；无失败时为 ``None``（Req 12.4）。
        executed_orders: 实际被执行步骤的 ``order`` 列表，按执行顺序排列；用于
            断言"执行顺序与失败短路"（Property 15）。失败步骤的 order 也计入。
        completed: 是否所有步骤均成功执行完毕（即 ``failure is None``）。
    """

    model_config = ConfigDict(extra="forbid")

    results: list[StepResult] = Field(
        default_factory=list,
        description="已成功执行步骤的结果，按执行顺序。",
    )
    failure: StepFailure | None = Field(
        default=None,
        description="触发短路的失败信息；无失败时为 None。",
    )
    executed_orders: list[int] = Field(
        default_factory=list,
        description="实际被执行步骤的 order，按执行顺序（含失败步骤）。",
    )
    completed: bool = Field(
        default=False,
        description="是否所有步骤均成功执行完毕。",
    )


# --------------------------------------------------------------------------- #
# ExecutorAgent                                                                 #
# --------------------------------------------------------------------------- #
class ExecutorAgent:
    """运维 Agent 的执行子 Agent（Req 12.1–12.5）。

    Args:
        tool_registry: 工具集；为 ``None`` 时构造默认注册表
            （:func:`build_default_registry`，含四个内置工具）。
        tool_call_timeout: 单次工具调用超时（秒）；为 ``None`` 时取
            ``settings.tool_call_timeout_seconds``（默认 30，范围 1–300，Req 12.5）。
    """

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry | None = None,
        tool_call_timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._tool_registry = (
            tool_registry if tool_registry is not None else build_default_registry()
        )
        self._tool_call_timeout: float = (
            tool_call_timeout
            if tool_call_timeout is not None
            else settings.tool_call_timeout_seconds
        )

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def tool_call_timeout(self) -> float:
        """单次工具调用超时（秒）。"""
        return self._tool_call_timeout

    # ------------------------------------------------------------------ #
    # 单步原语：execute_step（对应 design 的 execute(step)）                #
    # ------------------------------------------------------------------ #
    def execute_step(
        self, step: Step, params: dict[str, Any] | None = None
    ) -> StepResult | StepFailure:
        """执行单个步骤：调用其标注工具并归一化为结果或失败信息（Req 12.2–12.5）。

        无论工具返回错误、参数校验失败还是调用超时，本方法都**不抛出异常**，而是
        统一返回携带所属步骤标识的 :class:`StepFailure`，便于上层短路与移交
        Replanner_Agent。

        Args:
            step: 待执行步骤；其 ``tool_name`` 必须为 Tool_Registry 中已注册工具。
            params: 工具调用参数；为 ``None`` 时以空字典调用。

        Returns:
            * 成功 → :class:`StepResult`（``step_id`` + ``tool_response``，Req 12.3）。
            * 失败/超时 → :class:`StepFailure`（``step_id`` + ``failure_reason``，
              Req 12.4/12.5）。
        """
        call_params = params or {}
        try:
            response = self._call_with_timeout(
                lambda: self._tool_registry.invoke(step.tool_name, call_params),
                self._tool_call_timeout,
            )
        except FuturesTimeoutError:
            # Req 12.5：工具调用超时 → 判定为该步骤的工具调用失败。
            logger.warning(
                "步骤工具调用超时（step_id=%s, tool=%s, timeout=%ss）",
                step.step_id,
                step.tool_name,
                self._tool_call_timeout,
            )
            return StepFailure(
                step_id=step.step_id,
                failure_reason=(
                    f"工具 {step.tool_name} {TIMEOUT_REASON}："
                    f"在 {self._tool_call_timeout} 秒内未返回响应。"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - 工具错误归一化为 StepFailure（Req 12.4）
            logger.warning(
                "步骤工具调用失败（step_id=%s, tool=%s）：%r",
                step.step_id,
                step.tool_name,
                exc,
            )
            return StepFailure(
                step_id=step.step_id,
                failure_reason=f"工具 {step.tool_name} 调用失败：{exc}",
            )

        # Req 12.3：成功记录含步骤标识与工具响应内容的执行结果。
        return StepResult(step_id=step.step_id, tool_response=response)

    # ------------------------------------------------------------------ #
    # 计划级编排：execute_plan（Property 15 的执行顺序与失败短路）           #
    # ------------------------------------------------------------------ #
    def execute_plan(
        self,
        plan: Plan,
        *,
        params_for: Callable[[Step], dict[str, Any]] | None = None,
    ) -> ExecutionOutcome:
        """按 ``order`` 升序逐步执行计划，一旦某步失败即短路停止（Req 12.1/12.4）。

        Args:
            plan: 待执行的执行计划。
            params_for: 可选回调 ``(step) -> params``，为每个步骤提供工具调用参数；
                为 ``None`` 时每步以空字典调用。便于为内置工具（如 query_cls_log
                需要 ``{"query": ...}``）提供通过校验的参数。

        Returns:
            :class:`ExecutionOutcome`，含成功结果序列、失败信息（如有）、实际执行
            的步骤 order 序列与是否全部完成。
        """
        outcome = ExecutionOutcome()

        # Req 12.1：从第一个步骤开始按标注顺序（order 升序）每次执行一个步骤。
        ordered_steps = sorted(plan.steps, key=lambda s: s.order)

        for step in ordered_steps:
            params = params_for(step) if params_for is not None else None
            outcome.executed_orders.append(step.order)
            result = self.execute_step(step, params)

            if isinstance(result, StepFailure):
                # Req 12.4：记录失败信息、标记步骤失败、暂停后续步骤（短路停止）。
                step.status = StepStatus.FAILED
                outcome.failure = result
                outcome.completed = False
                return outcome

            # Req 12.3：记录成功结果并继续下一步。
            step.status = StepStatus.SUCCESS
            outcome.results.append(result)

        outcome.completed = True
        return outcome

    # ------------------------------------------------------------------ #
    # 内部：硬超时执行                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _call_with_timeout(func: Callable[[], Any], timeout: float) -> Any:
        """在独立线程中执行 ``func`` 并施加硬超时（Req 12.5）。

        超时抛 :class:`concurrent.futures.TimeoutError`。``shutdown(wait=False)`` 不
        阻塞等待潜在挂起线程，从而保证调用方视角的超时上界（线程本身不被强杀）。
        """
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(func)
            return future.result(timeout=timeout)
        finally:
            executor.shutdown(wait=False)

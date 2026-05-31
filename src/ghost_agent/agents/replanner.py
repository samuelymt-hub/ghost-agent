"""Replanner_Agent（重规划智能体，Req 13.1–13.6）。

Ops_Agent 的 Plan-Execute-Replan 协作循环中的第三个子 Agent。它接收 *当前计划* 与
Executor_Agent 移交的 *执行态汇总*（复用
:class:`ghost_agent.agents.executor.ExecutionOutcome`），输出一个三态评估结果
（:class:`~ghost_agent.models.plan.ReplanVerdict`）并据此决定流程走向：

* **COMPLETED（任务已完成，Req 13.2）**：终止后续步骤、结束排查流程，并基于已执行
  步骤的工具结果生成分析结果总结。
* **CONTINUE（任务未完成且剩余计划仍适用，Req 13.3）**：不生成新计划，由调用方
  （Ops_Agent）指示 Executor_Agent 继续执行剩余计划的下一个步骤。
* **REPLAN（任务未完成且剩余计划不再适用，Req 13.4）**：生成修订后的新计划、将重规划
  次数加 1，并交由 Executor_Agent 从新计划的第一个步骤开始执行。

两个终止护栏：

* **重规划次数上限（Req 13.5）**：当 REPLAN 使重规划次数达到配置的最大重规划次数上限
  （默认 10，范围 1–50）时，转为终止流程并生成包含 :data:`REPLAN_LIMIT_NOTE`
  （"因达到最大重规划次数而未完成"）说明的分析结果总结。重规划次数恒不超过该上限。
* **模型错误（Req 13.6）**：评估或生成新计划过程中 Chat_Model 返回错误（如
  :class:`~ghost_agent.models.errors.GenerationError`）时，终止排查流程、保留已执行步骤
  的工具结果，并生成包含评估失败原因说明的分析结果总结。

设计要点（可测试性优先）：

* **三态封闭由构造保证（Property 16）**：:meth:`ReplannerAgent.evaluate` 始终把评估器
  输出强制归一为 :class:`ReplanVerdict` 三个枚举之一；非法返回值会被当作评估错误处理
  （转入 13.6 终止路径），因此 ``ReplanResult.verdict`` 恒为三个取值之一。
* **重规划计数上界由构造保证（Property 17）**：REPLAN 时计数自增并钳制到
  ``[0, max_replan_count]``，达上限即终止，计数恒不超过上限。
* **确定性默认实现 + 可注入 seam**：默认的评估器 / 重规划器 / 总结器均为确定性、离线、
  不触网的纯逻辑，便于属性化测试；同时全部以依赖注入暴露，生产环境可注入由 Chat_Model
  驱动的 LLM 版本（其抛出的 :class:`GenerationError` 会被 13.6 路径捕获）。

终止语义约定：``ReplanResult.summary`` 非 ``None`` 当且仅当流程终止
（COMPLETED / 达上限 / 模型错误）。CONTINUE 与未达上限的 REPLAN 均为非终止
（``summary is None``）。调用方据 :attr:`ReplanResult.is_terminal` 判定是否结束流程。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from ghost_agent.agents.executor import ExecutionOutcome
from ghost_agent.config import get_settings
from ghost_agent.models.plan import (
    Plan,
    ReplanVerdict,
    Step,
    StepResult,
)
from ghost_agent.models.troubleshooting_task import NO_CONTENT, AnalysisSummary

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutionOutcome",
    "ReplanResult",
    "ReplannerAgent",
    "REPLAN_LIMIT_NOTE",
    "EVALUATION_FAILED_PREFIX",
    "default_evaluator",
    "default_replanner",
    "default_summarizer",
]

# 重规划次数上限的取值范围（Req 13.5）。构造期对越界配置做防御性钳制。
_MIN_REPLAN = 1
_MAX_REPLAN = 50

#: 因达到最大重规划次数而终止时，写入分析结果总结的说明（必须原样包含此子串，Req 13.5 / Property 17）。
REPLAN_LIMIT_NOTE = "因达到最大重规划次数而未完成"

#: 因评估 / 生成新计划过程中模型返回错误而终止时，写入分析结果总结的失败说明前缀（Req 13.6）。
EVALUATION_FAILED_PREFIX = "因评估过程出错而未完成："


# --------------------------------------------------------------------------- #
# 输入：执行态汇总                                                              #
# --------------------------------------------------------------------------- #
# Replanner_Agent 的输入「执行态汇总」复用 Executor_Agent 的聚合结果
# :class:`ghost_agent.agents.executor.ExecutionOutcome`（含 ``completed`` /
# ``failure`` / ``results``），与 design.md 的 Plan-Execute-Replan 流程一致：
# Executor 产出 ExecutionOutcome 并移交 Replanner 评估。此处经 ``__all__`` 重导出，
# 便于上层与测试从本模块统一引用。


# --------------------------------------------------------------------------- #
# 输出：评估结果                                                                #
# --------------------------------------------------------------------------- #
class ReplanResult(BaseModel):
    """Replanner_Agent 的一次评估结果 (Req 13.1–13.6)。

    Attributes:
        verdict: 三态评估结果，恒为 :class:`ReplanVerdict` 三个取值之一（Property 16）。
        new_plan: 仅当未达上限的 REPLAN 时给出的修订后新计划（Req 13.4）；否则为 ``None``。
        replan_count: 评估后的重规划次数，恒满足 ``0 <= replan_count <= max_replan_count``
            （Property 17）。
        summary: 终止时生成的分析结果总结（COMPLETED / 达上限 / 模型错误）；非终止时为
            ``None``。
    """

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    verdict: ReplanVerdict = Field(..., description="三态评估结果。")
    new_plan: Plan | None = Field(default=None, description="修订后的新计划（仅 REPLAN 未达上限时）。")
    replan_count: int = Field(..., ge=0, description="评估后的重规划次数。")
    summary: AnalysisSummary | None = Field(
        default=None,
        description="终止时的分析结果总结；非终止时为 None。",
    )

    @property
    def is_terminal(self) -> bool:
        """是否应终止排查流程：当且仅当存在分析结果总结。"""
        return self.summary is not None


# --------------------------------------------------------------------------- #
# 默认（确定性、离线）实现                                                       #
# --------------------------------------------------------------------------- #
def default_evaluator(plan: Plan, outcome: ExecutionOutcome) -> ReplanVerdict:
    """确定性默认评估器（非 LLM，Req 13.1）。

    依据最近一次执行态汇总判定三态：

    * 所有步骤均成功完成 (``outcome.completed``) → ``COMPLETED``；
    * 存在失败步骤 (``outcome.failure is not None``) → ``REPLAN``（剩余计划不再适用）；
    * 其余情形 → ``CONTINUE``（剩余计划仍适用，继续下一步）。

    输出恒为三个枚举之一（Property 16 由构造成立）。
    """
    if outcome.completed:
        return ReplanVerdict.COMPLETED
    if outcome.failure is not None:
        return ReplanVerdict.REPLAN
    return ReplanVerdict.CONTINUE


def default_replanner(plan: Plan, outcome: ExecutionOutcome) -> Plan:
    """确定性默认重规划器（非 LLM，Req 13.4）。

    生成一个标注"无手册依据"的通用兜底计划：以一个基础诊断步骤重新开始排查。
    生产环境可注入由 Chat_Model 驱动的版本以生成更贴合上下文的新计划。
    """
    return Plan(
        grounded=False,
        steps=[
            Step(
                order=0,
                tool_name="query_cls_log",
                goal="重新排查：收集基础诊断信息以修订处理计划",
            )
        ],
    )


def _stringify(value: Any) -> str:
    """将工具响应归一化为可读文本（用于拼装"已执行操作记录"）。"""
    if isinstance(value, str):
        return value
    return str(value)


def default_summarizer(
    results: list[StepResult],
    note: str | None = None,
) -> AnalysisSummary:
    """确定性默认总结器（非 LLM，Req 13.2 / 13.5 / 13.6）。

    生成三段式分析结果总结（根因分析 / 处理建议 / 已执行操作记录）：

    * **已执行操作记录**：若存在已执行结果则逐条拼装步骤标识与工具响应；否则以
      :data:`~ghost_agent.models.troubleshooting_task.NO_CONTENT` 占位（保留已执行结果，
      Req 13.6）。
    * **根因分析**：终止说明（``note``，如达上限 / 评估失败原因）写入此处；无说明时以
      ``NO_CONTENT`` 占位。
    * **处理建议**：默认无可填充内容，以 ``NO_CONTENT`` 占位。

    Args:
        results: 已执行步骤的成功结果列表。
        note: 终止说明文本（非空）；达上限时须包含 :data:`REPLAN_LIMIT_NOTE`，
            模型错误时为评估失败原因说明。``None`` 表示无附加说明（如正常完成）。

    Returns:
        三段式 :class:`AnalysisSummary`。
    """
    if results:
        actions = "；".join(
            f"步骤 {r.step_id}: {_stringify(r.tool_response)}" for r in results
        )
    else:
        actions = NO_CONTENT

    root_cause = note if note else NO_CONTENT
    return AnalysisSummary(
        root_cause=root_cause,
        suggestions=NO_CONTENT,
        executed_actions=actions,
    )


# 可注入 seam 的类型别名。
EvaluatorFn = Callable[[Plan, ExecutionOutcome], ReplanVerdict]
ReplannerFn = Callable[[Plan, ExecutionOutcome], Plan]
SummarizerFn = Callable[[list[StepResult], str | None], AnalysisSummary]


# --------------------------------------------------------------------------- #
# ReplannerAgent                                                                #
# --------------------------------------------------------------------------- #
class ReplannerAgent:
    """重规划智能体（确定性默认实现，全部协作者可注入，Req 13.1–13.6）。

    Args:
        chat_model: 预留的对话模型 seam（默认评估器 / 重规划器为确定性离线实现，不使用
            它；注入 LLM 版评估器 / 重规划器时可经此传入模型）。
        evaluator: 评估器 seam ``(plan, outcome) -> ReplanVerdict``；为 ``None`` 时使用
            :func:`default_evaluator`。输出会被强制归一为 :class:`ReplanVerdict` 三个枚举
            之一（Property 16）。
        replanner: 重规划器 seam ``(plan, outcome) -> Plan``；为 ``None`` 时使用
            :func:`default_replanner`。
        summarizer: 总结器 seam ``(results, note?) -> AnalysisSummary``；为 ``None`` 时使用
            :func:`default_summarizer`。
        max_replan_count: 最大重规划次数；为 ``None`` 时取 ``settings.max_replan_count``。
            无论来源如何均被钳制到 ``[1, 50]``（Req 13.5 / Property 17）。
    """

    def __init__(
        self,
        *,
        chat_model: Any | None = None,
        evaluator: EvaluatorFn | None = None,
        replanner: ReplannerFn | None = None,
        summarizer: SummarizerFn | None = None,
        max_replan_count: int | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._evaluator: EvaluatorFn = evaluator if evaluator is not None else default_evaluator
        self._replanner: ReplannerFn = (
            replanner if replanner is not None else default_replanner
        )
        self._summarizer: SummarizerFn = (
            summarizer if summarizer is not None else default_summarizer
        )

        raw = (
            max_replan_count
            if max_replan_count is not None
            else get_settings().max_replan_count
        )
        # 防御性钳制：即便配置层失效，上限也不会超出 [1, 50]（Req 13.5 / Property 17）。
        self._max_replan_count: int = max(_MIN_REPLAN, min(_MAX_REPLAN, int(raw)))

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def max_replan_count(self) -> int:
        """最大重规划次数（已钳制到 [1, 50]）。"""
        return self._max_replan_count

    # ------------------------------------------------------------------ #
    # 公共 API：evaluate                                                  #
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        plan: Plan,
        latest: ExecutionOutcome,
        *,
        replan_count: int = 0,
    ) -> ReplanResult:
        """评估当前计划与执行态汇总，输出三态评估并决定流程走向（Req 13.1–13.6）。

        Args:
            plan: 当前执行计划。
            latest: Executor_Agent 移交的执行态汇总（含最近结果 / 失败信息与已执行结果）。
            replan_count: 进入本次评估前已累计的重规划次数（由调用方在循环中透传）。
                负值会被钳制为 0。

        Returns:
            :class:`ReplanResult`，其 ``verdict`` 恒为三态之一（Property 16），
            ``replan_count`` 恒不超过配置上限（Property 17），``summary`` 非 ``None``
            表示流程终止。
        """
        current_count = max(0, int(replan_count))

        # --- 13.1：评估，强制三态归一；评估出错按 13.6 终止 ---------------- #
        try:
            verdict = ReplanVerdict(self._evaluator(plan, latest))
        except Exception as exc:  # noqa: BLE001 - 评估失败（含 LLM GenerationError / 非法返回值）按 13.6 处理
            logger.warning("Replanner 评估失败，按 Req 13.6 终止流程：%r", exc)
            return self._terminate_on_error(latest, current_count, exc)

        # --- 13.2：任务已完成 → 终止并基于已执行结果生成总结 --------------- #
        if verdict is ReplanVerdict.COMPLETED:
            return ReplanResult(
                verdict=ReplanVerdict.COMPLETED,
                replan_count=current_count,
                summary=self._summarize(latest, note=None),
            )

        # --- 13.3：剩余计划仍适用 → 指示执行下一步（不生成新计划） --------- #
        if verdict is ReplanVerdict.CONTINUE:
            return ReplanResult(
                verdict=ReplanVerdict.CONTINUE,
                replan_count=current_count,
                new_plan=None,
                summary=None,
            )

        # --- REPLAN：剩余计划不再适用 ------------------------------------- #
        # 13.4：重规划次数加 1。
        new_count = current_count + 1

        # 13.5：达到（或越过）上限 → 转为终止并生成含达上限说明的总结。
        if new_count >= self._max_replan_count:
            capped = min(new_count, self._max_replan_count)
            logger.info(
                "Replanner 重规划次数达上限 %d，按 Req 13.5 终止流程", self._max_replan_count
            )
            return ReplanResult(
                verdict=ReplanVerdict.REPLAN,
                replan_count=capped,
                new_plan=None,
                summary=self._summarize(latest, note=REPLAN_LIMIT_NOTE),
            )

        # 13.4：未达上限 → 生成修订后的新计划，交由 Executor 从第一步执行。
        try:
            new_plan = self._replanner(plan, latest)
        except Exception as exc:  # noqa: BLE001 - 生成新计划出错按 13.6 处理（保留已执行结果，计数不前进）
            logger.warning("Replanner 生成新计划失败，按 Req 13.6 终止流程：%r", exc)
            return self._terminate_on_error(latest, current_count, exc)

        return ReplanResult(
            verdict=ReplanVerdict.REPLAN,
            replan_count=new_count,
            new_plan=new_plan,
            summary=None,
        )

    # ------------------------------------------------------------------ #
    # 内部辅助                                                            #
    # ------------------------------------------------------------------ #
    def _summarize(
        self, latest: ExecutionOutcome, *, note: str | None
    ) -> AnalysisSummary:
        """基于已执行结果生成分析结果总结（Req 13.2 / 13.5 / 13.6）。"""
        results = list(getattr(latest, "results", None) or [])
        return self._summarizer(results, note)

    def _terminate_on_error(
        self, latest: ExecutionOutcome, replan_count: int, exc: BaseException
    ) -> ReplanResult:
        """模型错误终止路径（Req 13.6）：保留已执行结果，生成含失败原因说明的总结。

        重规划次数保持不变（错误发生时本次重规划未成功推进），评估结果以 ``COMPLETED``
        表示终止（终止与否由 ``summary`` 是否存在判定，三态保持封闭，Property 16）。
        """
        note = f"{EVALUATION_FAILED_PREFIX}{exc}"
        return ReplanResult(
            verdict=ReplanVerdict.COMPLETED,
            replan_count=replan_count,
            summary=self._summarize(latest, note=note),
        )

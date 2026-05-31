"""Planner_Agent（规划智能体，Req 11.1–11.6）。

本模块实现运维 Agent（Plan-Execute-Replan）规划阶段的子智能体 Planner_Agent，
对外提供单一入口 :meth:`PlannerAgent.plan`，依据告警信息制定一份由有序步骤组成的
执行计划（:class:`~ghost_agent.models.plan.Plan`）：

* **查询处理手册（Req 11.1）**：``plan`` 启动后先经 Tool_Registry 调用内置工具
  ``query_internal_docs`` 查询与告警相关的处理步骤。
* **生成有序计划（Req 11.2, 11.3）**：基于查询结果与告警信息生成有序步骤计划，步骤
  数量介于 1 与配置的最大步骤数上限之间；每个步骤标注待调用的工具（取自
  Tool_Registry 已注册工具）与该步骤目标。
* **降级为通用计划（Req 11.4, 11.5）**：当 ``query_internal_docs`` 未返回相关处理步骤
  （结果为空），或调用失败 / 在配置的查询超时时间内未返回时，基于告警信息生成通用排查
  计划并标注 ``grounded=False``（无手册依据）。
* **移交 Executor（Req 11.6）**：完成计划生成即返回 :class:`Plan`，由上层
  Ops_Agent 编排移交给 Executor_Agent；返回 ``Plan`` 即为该"移交"动作。

设计抉择：**默认采用确定性（deterministic）的离线计划生成器，而非 LLM。**
设计文档将 LLM（``ops_planner`` 提示词 + Chat_Model）列为生成计划的便利项，但
**Property 14**（步骤数边界与步骤合法性）要求"步骤数量 1..maxSteps、序号连续、工具名
均属于 Tool_Registry 已注册集合且目标非空"这些约束"由构造保证"，并可在完全离线、无
网络环境下被属性化测试覆盖。因此默认的 ``plan_builder`` 是一个确定性生成器：它只从
注册表"已注册工具名集合"中按既定排查顺序挑选工具、为每个步骤生成非空目标、并把步骤数
钳制到 ``[1, max_steps]``——从而 Property 14 由构造成立。LLM 驱动的 builder 可在装配期
（任务 13.8 / 16）通过 ``plan_builder`` 注入，但不作为默认，以保证测试确定性与离线性。

所有协作者（Tool_Registry / Chat_Model / Prompt_Module / 计划生成器）均以依赖注入方式
提供，默认实现惰性构造、构造期不触网，从而支持确定性单元 / 属性测试。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable

from ghost_agent.config import get_settings
from ghost_agent.core.chat_model import ChatModel
from ghost_agent.core.prompt_module import PromptModule
from ghost_agent.core.tool_registry import ToolRegistry, build_default_registry
from ghost_agent.models.plan import Plan, Step
from ghost_agent.models.troubleshooting_task import AlarmInfo

logger = logging.getLogger(__name__)

__all__ = [
    "PlannerAgent",
    "PlanBuilder",
    "default_plan_builder",
    "DOCS_TOOL_NAME",
    "GENERIC_TOOL_ORDER",
]

#: 内部文档查询工具名（Req 11.1）。
DOCS_TOOL_NAME = "query_internal_docs"

#: 通用排查计划的工具优先顺序：先查手册、再查日志与告警指标、最后上报结论。
#: 实际生成时与注册表"已注册工具名集合"取交集并保持本顺序（Req 11.3 / Property 14）。
GENERIC_TOOL_ORDER: tuple[str, ...] = (
    "query_internal_docs",
    "query_cls_log",
    "query_prometheus_alarm",
    "send_msg",
)

#: 各工具在排查计划中的目标提示（用于生成非空 goal，Req 11.3）。
_TOOL_GOAL_HINTS: dict[str, str] = {
    "query_internal_docs": "检索内部处理手册，定位与告警相关的标准处理步骤",
    "query_cls_log": "查询 CLS 日志，定位异常时间段的错误与上下文",
    "query_prometheus_alarm": "查询 Prometheus 告警与指标，确认受影响的服务与范围",
    "send_msg": "向值班群发送排查结论与处理建议",
}

#: 判定 query_internal_docs 返回结果中"处理步骤/文档"集合的候选键（Req 11.4）。
_DOC_RESULT_KEYS: tuple[str, ...] = (
    "docs",
    "documents",
    "results",
    "steps",
    "hits",
    "items",
    "matches",
    "content",
)

#: 目标文本中嵌入告警描述时的最大长度（仅为可读性截断，不影响非空保证）。
_GOAL_MESSAGE_MAX = 160

#: 计划生成器签名：依据告警、手册查询结果、可用工具名、步骤上限与是否有手册依据，
#: 产出有序步骤列表。注入自定义（如 LLM 驱动）实现时需遵循此签名。
PlanBuilder = Callable[..., list[Step]]


def _truncate(text: str, limit: int = _GOAL_MESSAGE_MAX) -> str:
    """对过长文本做可读性截断（不改变非空性）。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _goal_for(tool_name: str, message: str, *, grounded: bool) -> str:
    """为某个工具步骤构造**非空**目标文本（Req 11.3 / Property 14）。

    目标由固定的中文前缀 + 工具语义提示 + 告警摘要拼装而成；前缀与提示均为非空常量，
    因此无论 ``message`` 内容如何（含纯空白 / 特殊字符），返回的目标始终非空。
    """
    hint = _TOOL_GOAL_HINTS.get(tool_name, f"调用工具 {tool_name} 推进排查")
    plan_kind = "依据处理手册" if grounded else "通用排查"
    snippet = _truncate(message)
    return f"{plan_kind}：{hint}（告警：{snippet}）"


def default_plan_builder(
    *,
    alarm: AlarmInfo,
    docs: Any,
    available_tools: list[str],
    max_steps: int,
    grounded: bool,
) -> list[Step]:
    """默认的确定性计划生成器（离线、无 LLM，Property 14 由构造成立）。

    生成规则：

    1. 从 :data:`GENERIC_TOOL_ORDER` 中取与 ``available_tools`` 的交集并保持该顺序；
       若交集为空（注册表中没有任何通用排查工具），则回退为按注册顺序使用全部可用工具。
    2. 将所选工具序列截断到至多 ``max_steps`` 个，保证 ``len(steps) <= max_steps``。
    3. 为每个步骤分配连续序号（``order == index``，0..n-1）与**非空**目标。

    由此产出的步骤恒满足 Property 14：

    * ``1 <= len(steps) <= max_steps``（前提：``available_tools`` 非空）；
    * 序号连续（0..n-1）；
    * 每个 ``tool_name`` 均属于 ``available_tools``（即注册表已注册工具名集合）；
    * 每个 ``goal`` 非空。

    Args:
        alarm: 告警信息（其 ``message`` 摘要嵌入各步骤目标）。
        docs: ``query_internal_docs`` 的查询结果（仅作上下文，默认生成器不深度解析）。
        available_tools: 注册表已注册工具名列表（保持注册顺序）。
        max_steps: 步骤数上限（调用方已保证 ``>= 1``）。
        grounded: 是否有手册依据（影响目标文案，不影响结构合法性）。

    Returns:
        有序 :class:`Step` 列表。

    Raises:
        ValueError: 当 ``available_tools`` 为空（无任何已注册工具可供标注）时；此时
            无法产出"工具名属于已注册集合"的合法步骤。默认注册表始终含四个内置工具，
            因此正常装配下不会触发。
    """
    available_set = set(available_tools)
    ordered = [name for name in GENERIC_TOOL_ORDER if name in available_set]
    if not ordered:
        # 回退：注册表不含任何通用排查工具时，按注册顺序使用全部可用工具。
        ordered = list(available_tools)
    if not ordered:
        raise ValueError("无法生成执行计划：Tool_Registry 中没有任何已注册工具")

    # 截断到步骤上限，保证 len(steps) <= max_steps。
    ordered = ordered[: max(1, int(max_steps))]

    steps: list[Step] = []
    for index, tool_name in enumerate(ordered):
        steps.append(
            Step(
                order=index,
                tool_name=tool_name,
                goal=_goal_for(tool_name, alarm.message, grounded=grounded),
            )
        )
    return steps


class PlannerAgent:
    """运维 Agent 规划阶段子智能体（Req 11.1–11.6）。

    Args:
        tool_registry: 工具集；为 ``None`` 时构造默认注册表
            （:func:`build_default_registry`，含四个内置工具）。
        chat_model: 对话模型；为 ``None`` 时构造默认 :class:`ChatModel`（惰性连接）。
            默认确定性 ``plan_builder`` 不使用本模型；其仅供注入的 LLM 驱动 builder 使用。
        prompt_module: 提示词模块（``ops_planner`` 模板）；为 ``None`` 时构造默认
            :class:`PromptModule`。同样仅供 LLM 驱动 builder 使用。
        max_steps: 单个执行计划的最大步骤数上限；为 ``None`` 时取
            ``settings.max_plan_steps``。无论来源如何均被钳制到 ``>= 1``（Req 11.2）。
        plan_builder: 计划生成器注入点；为 ``None`` 时使用 :func:`default_plan_builder`
            （确定性、离线，保证 Property 14）。
        query_timeout: ``query_internal_docs`` 查询超时（秒）；为 ``None`` 时取
            ``settings.tool_call_timeout_seconds``（Req 11.5）。
    """

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry | None = None,
        chat_model: ChatModel | None = None,
        prompt_module: PromptModule | None = None,
        max_steps: int | None = None,
        plan_builder: PlanBuilder | None = None,
        query_timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._tool_registry = (
            tool_registry if tool_registry is not None else build_default_registry()
        )
        self._chat_model = chat_model if chat_model is not None else ChatModel()
        self._prompt_module = (
            prompt_module if prompt_module is not None else PromptModule()
        )
        raw_max_steps = max_steps if max_steps is not None else settings.max_plan_steps
        # 防御性钳制：步骤上限至少为 1（Req 11.2 / Property 14）。
        self._max_steps: int = max(1, int(raw_max_steps))
        self._plan_builder: PlanBuilder = (
            plan_builder if plan_builder is not None else default_plan_builder
        )
        self._query_timeout: float = (
            query_timeout
            if query_timeout is not None
            else settings.tool_call_timeout_seconds
        )

    # ------------------------------------------------------------------ #
    # 只读属性                                                            #
    # ------------------------------------------------------------------ #
    @property
    def max_steps(self) -> int:
        """单个执行计划的最大步骤数上限（已钳制到 >= 1）。"""
        return self._max_steps

    # ------------------------------------------------------------------ #
    # 公共 API：plan                                                      #
    # ------------------------------------------------------------------ #
    def plan(self, alarm: AlarmInfo) -> Plan:
        """依据告警信息制定执行计划（Req 11.1–11.6）。

        流程：
            1. 经 Tool_Registry 调用 ``query_internal_docs`` 查询处理手册（Req 11.1），
               调用受 ``query_timeout`` 硬超时保护。
            2. 依据查询结果判定是否有手册依据（``grounded``）：返回非空处理步骤 →
               ``grounded=True``；返回为空（Req 11.4）/ 调用失败 / 超时（Req 11.5）→
               ``grounded=False`` 并生成通用计划。
            3. 经 ``plan_builder`` 生成有序步骤；默认生成器保证步骤数 ``[1, max_steps]``、
               序号连续、工具名属于已注册集合且目标非空（Req 11.2, 11.3 / Property 14）。
            4. 返回 :class:`Plan`，即移交 Executor_Agent（Req 11.6）。

        Args:
            alarm: 告警信息。

        Returns:
            生成的执行计划 :class:`Plan`。
        """
        docs, grounded = self._query_internal_docs(alarm)

        available_tools = [d.name for d in self._tool_registry.list_definitions()]
        steps = self._plan_builder(
            alarm=alarm,
            docs=docs,
            available_tools=available_tools,
            max_steps=self._max_steps,
            grounded=grounded,
        )

        # Plan 模型校验：>=1 步、序号连续、step_id 唯一（与默认生成器的构造一致）。
        return Plan(grounded=grounded, steps=steps)

    # ------------------------------------------------------------------ #
    # 内部：查询处理手册                                                    #
    # ------------------------------------------------------------------ #
    def _query_internal_docs(self, alarm: AlarmInfo) -> tuple[Any, bool]:
        """调用 ``query_internal_docs`` 查询处理步骤（Req 11.1, 11.4, 11.5）。

        以 try/except 包裹并施加硬超时：

        * 工具未注册 / 调用失败 / 超时（Req 11.5）→ 返回 ``(None, False)``。
        * 调用成功但结果为空（Req 11.4）→ 返回 ``(result, False)``。
        * 调用成功且结果含非空处理步骤 → 返回 ``(result, True)``。

        Returns:
            ``(docs, grounded)`` 二元组。
        """
        if not self._tool_registry.has(DOCS_TOOL_NAME):
            # 注册表未提供内部文档查询工具，无法查询手册 → 通用计划。
            logger.warning(
                "Tool_Registry 未注册 %s，按无手册依据生成通用计划", DOCS_TOOL_NAME
            )
            return None, False

        try:
            result = self._call_with_timeout(
                lambda: self._tool_registry.invoke(
                    DOCS_TOOL_NAME, {"query": alarm.message}
                ),
                self._query_timeout,
            )
        except FuturesTimeoutError:
            # Req 11.5：查询超时 → 降级为通用计划。
            logger.warning(
                "%s 查询超时（%ss），降级为通用计划",
                DOCS_TOOL_NAME,
                self._query_timeout,
            )
            return None, False
        except Exception as exc:  # noqa: BLE001 - 查询失败统一降级（Req 11.5）
            logger.warning(
                "%s 查询失败，降级为通用计划：%r", DOCS_TOOL_NAME, exc
            )
            return None, False

        grounded = self._has_docs(result)
        if not grounded:
            # Req 11.4：未返回相关处理步骤 → 通用计划、无手册依据。
            logger.info("%s 未返回相关处理步骤，生成通用计划", DOCS_TOOL_NAME)
        return result, grounded

    # ------------------------------------------------------------------ #
    # 内部：结果判定与超时执行                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _has_docs(result: Any) -> bool:
        """判定 ``query_internal_docs`` 的结果是否包含非空处理步骤（Req 11.4）。

        判定规则（宽松且确定）：
            * ``None`` / 空集合 / 空串 → 无文档。
            * 列表 / 元组 / 集合 → 以其非空性判定。
            * 字典 → 在 :data:`_DOC_RESULT_KEYS` 中寻找"文档/步骤"候选键，命中则以该键
              对应值的非空性判定；若不含任何候选键，则视为元数据（非文档）→ 无文档。
            * 非空字符串 → 视为有文档。
            * 其他真值 → 视为有文档。
        """
        if result is None:
            return False
        if isinstance(result, (list, tuple, set)):
            return len(result) > 0
        if isinstance(result, dict):
            for key in _DOC_RESULT_KEYS:
                if key in result:
                    return bool(result[key])
            # 不含任何"文档"候选键：视为元数据（如占位 stub），判定为无文档。
            return False
        if isinstance(result, str):
            return bool(result.strip())
        return bool(result)

    @staticmethod
    def _call_with_timeout(func: Callable[[], Any], timeout: float) -> Any:
        """在独立线程中执行 ``func`` 并施加硬超时（Req 11.5）。

        超时抛 :class:`concurrent.futures.TimeoutError`。``shutdown(wait=False)`` 不阻塞
        等待潜在挂起线程，从而保证调用方视角的查询超时上界（线程本身不被强杀）。
        """
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(func)
            return future.result(timeout=timeout)
        finally:
            executor.shutdown(wait=False)

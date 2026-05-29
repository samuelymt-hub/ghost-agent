"""执行计划 Plan 与步骤 Step 数据模型 (Req 11.2, 12.1, 13.1)。

包含:

* :class:`StepStatus`     —— 步骤状态枚举。
* :class:`ReplanVerdict`  —— Replanner_Agent 三态评估结果 (Req 13.1)。
* :class:`Step`           —— 单个执行步骤。
* :class:`Plan`           —— 由有序步骤组成的执行计划。
* :class:`StepResult`     —— 步骤成功执行结果 (Req 12.3)。
* :class:`StepFailure`    —— 步骤失败信息 (Req 12.4)。

步骤数与配置 ``maxSteps`` 的上限对比由 :mod:`ghost_agent.agents.planner` 在
运行时完成 (Req 11.2)；本模型只强制结构性约束：步骤序号连续、step_id 唯一。
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _new_id() -> str:
    """生成 plan/step 的唯一标识。"""
    return str(uuid.uuid4())


class StepStatus(str, Enum):
    """步骤执行状态。"""

    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ReplanVerdict(str, Enum):
    """Replanner_Agent 三态评估结果 (Req 13.1)。

    * ``COMPLETED`` —— 任务已完成。
    * ``CONTINUE``  —— 任务未完成且剩余计划仍适用。
    * ``REPLAN``    —— 任务未完成且剩余计划不再适用。
    """

    COMPLETED = "COMPLETED"
    CONTINUE = "CONTINUE"
    REPLAN = "REPLAN"


class Step(BaseModel):
    """单个执行步骤 (Req 11.2, 11.3, 12.1)。"""

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    step_id: str = Field(
        default_factory=_new_id,
        description="唯一步骤标识。",
    )
    order: int = Field(
        ...,
        ge=0,
        description="步骤序号 (Req 11.2, 12.1)。",
    )
    tool_name: str = Field(
        ...,
        min_length=1,
        description="待调用工具名，取自 Tool_Registry (Req 11.3, 12.2)。",
    )
    goal: str = Field(
        ...,
        min_length=1,
        description="本步骤的目标 (Req 11.3)。",
    )
    status: StepStatus = Field(
        default=StepStatus.PENDING,
        description="步骤执行状态。",
    )


class Plan(BaseModel):
    """执行计划 (Req 11.2)。

    本模型强制：

    * ``steps`` 至少包含一个步骤；
    * ``step_id`` 在计划内唯一；
    * ``order`` 为从 0 开始的连续递增序列（即按 design.md "Property 14" 中
      "步骤序号连续"的语义）。
    """

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    plan_id: str = Field(
        default_factory=_new_id,
        description="唯一计划标识。",
    )
    grounded: bool = Field(
        ...,
        description="是否有手册依据 (Req 11.4, 11.5)。",
    )
    steps: list[Step] = Field(
        ...,
        min_length=1,
        description="按 order 升序的执行步骤序列。",
    )

    @model_validator(mode="after")
    def _check_steps(self) -> "Plan":
        # step_id 在计划内唯一
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan.steps 中存在重复的 step_id")

        # order 必须是从 0 开始的连续递增序列：[0, 1, 2, ..., n-1]
        expected = list(range(len(self.steps)))
        actual = [s.order for s in self.steps]
        if actual != expected:
            raise ValueError(
                "Plan.steps 的 order 必须是从 0 开始连续递增的序列；"
                f"期望 {expected}，实际 {actual}"
            )
        return self


class StepResult(BaseModel):
    """步骤成功执行结果 (Req 12.3)。"""

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    step_id: str = Field(
        ...,
        min_length=1,
        description="所属步骤标识。",
    )
    tool_response: Any = Field(
        ...,
        description="工具响应内容 (Req 12.3)。",
    )


class StepFailure(BaseModel):
    """步骤失败信息 (Req 12.4)。"""

    model_config = ConfigDict(
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    step_id: str = Field(
        ...,
        min_length=1,
        description="所属步骤标识。",
    )
    failure_reason: str = Field(
        ...,
        min_length=1,
        description="失败原因。",
    )


__all__ = [
    "StepStatus",
    "ReplanVerdict",
    "Step",
    "Plan",
    "StepResult",
    "StepFailure",
]

"""Prompt_Module（提示词模块，Req 9.1, 20.1–20.5）。

本模块以**唯一名称**管理提示词模板，并将模板渲染为最终提示词文本，对上层 Agent
（Knowledge_Base_Agent / Conversation_Agent / Ops_Agent）提供与具体调用代码解耦的
提示词构造能力：

* :class:`PromptTemplate` —— 具名提示词模板，含角色定义、任务目标、输出结构/格式
  约束说明、是否需要分步推理标注，以及可含占位符的主体模板。
* :class:`Prompt`         —— 渲染产物，承载最终文本与结构化分节（便于断言/调试）。
* :class:`PromptModule`   —— 模板注册表与渲染器：
  - :meth:`PromptModule.register` 以唯一名称新增或**热替换**同名模板（Req 20.4），
    调用方无需改动代码即可更新模板内容。
  - :meth:`PromptModule.get` 按名称取模板，缺失时抛 :class:`TemplateNotFoundError`
    （Req 20.5）。
  - :meth:`PromptModule.build` 渲染具名模板：始终包含角色定义、任务目标与输出约束
    （Req 20.1, 20.3），模板标注需分步推理时额外加入分步思考指令（Req 20.2）；引用
    模板不存在时停止构造并返回缺失模板名错误（Req 20.5）。
  - :meth:`PromptModule.build_rag_prompt` 构造 RAG 增强提示词：将用户查询与召回集合
    中**全部** Chunk 拼为单条提示词，并标注每个 Chunk 的来源文件标识（Req 9.1）。

设计要点：
- **变量替换的 brace 安全（Brace-safe）**：仅对模板作者可控的 ``body_template`` 执行
  ``str.format_map`` 替换，且使用"宽容字典"——缺失占位符原样保留、额外变量被忽略，
  绝不因缺失/多余变量而崩溃。用户查询与召回 Chunk 文本**一律以拼接方式**嵌入，绝不
  对其执行 ``str.format``，从而杜绝任意 Unicode / 花括号内容引发的 ``KeyError`` /
  格式注入（对 Hypothesis 生成的特殊输入尤为关键）。
- **结构化分节标记**：渲染文本使用固定中文小节标记（``## 角色`` / ``## 任务目标`` /
  ``## 输出要求`` / ``## 分步思考`` / ``## 参考资料`` / ``## 用户问题``），使必含字段
  与分步指令"按构造成立"，便于属性测试断言（Property 10 / Property 12）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ghost_agent.models.errors import TemplateNotFoundError
from ghost_agent.vector_db.vector_store import SearchHit

__all__ = [
    "PromptTemplate",
    "Prompt",
    "PromptModule",
    "STEP_REASONING_INSTRUCTION",
]


# --------------------------------------------------------------------------- #
# 固定小节标记与分步推理指令                                                     #
# --------------------------------------------------------------------------- #
_SECTION_ROLE = "## 角色"
_SECTION_TASK_GOAL = "## 任务目标"
_SECTION_OUTPUT = "## 输出要求"
_SECTION_STEP = "## 分步思考"
_SECTION_BODY = "## 说明"
_SECTION_CONTEXT = "## 参考资料"
_SECTION_QUESTION = "## 用户问题"

#: 模板标注需要分步推理时注入的分步思考指令（Req 20.2）。
#: 取一段足够独特的中文长句，确保其不会与普通模板字段内容偶然碰撞。
STEP_REASONING_INSTRUCTION = (
    "请按步骤逐步推理（Step-by-Step）：先逐条列出分析过程与依据，"
    "再据此给出最终结论，避免跳步或臆断。"
)


# --------------------------------------------------------------------------- #
# 宽容字典：缺失占位符原样保留，额外变量忽略                                       #
# --------------------------------------------------------------------------- #
class _ForgivingDict(dict):
    """供 ``str.format_map`` 使用的宽容映射。

    当模板中引用的占位符在变量字典中缺失时，``__missing__`` 返回原样的
    ``{key}`` 文本而非抛出 ``KeyError``，保证渲染不因缺失变量崩溃（Req 20 工程规范）。
    """

    def __missing__(self, key: str) -> str:  # noqa: D401 - 简单委托
        return "{" + key + "}"


# --------------------------------------------------------------------------- #
# 模型                                                                          #
# --------------------------------------------------------------------------- #
class PromptTemplate(BaseModel):
    """具名提示词模板（Req 20.1–20.4）。

    Attributes:
        name: 模板唯一名称（Req 20.4，注册/热替换的键）。
        role: 角色定义（Req 20.1）。
        task_goal: 任务目标（Req 20.1）。
        output_constraints: 输出结构/格式约束说明（Req 20.3）。
        requires_step_reasoning: 是否标注为需要分步推理；为 ``True`` 时渲染会额外
            加入分步思考指令（Req 20.2）。
        body_template: 可选主体模板，可含 ``{var}`` 命名占位符；渲染时以宽容方式
            用调用方提供的变量替换（缺失占位符原样保留）。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="模板唯一名称 (Req 20.4)。")
    role: str = Field(..., description="角色定义 (Req 20.1)。")
    task_goal: str = Field(..., description="任务目标 (Req 20.1)。")
    output_constraints: str = Field(..., description="输出结构/格式约束说明 (Req 20.3)。")
    requires_step_reasoning: bool = Field(
        default=False, description="是否需要分步推理 (Req 20.2)。"
    )
    body_template: str = Field(
        default="", description="可含 {var} 命名占位符的主体模板。"
    )


class Prompt(BaseModel):
    """提示词渲染产物。

    Attributes:
        text: 最终拼装的完整提示词文本。
        template_name: 来源模板名称。
        sections: 结构化分节（``role`` / ``task_goal`` / ``output_constraints`` 及可选
            的 ``step_reasoning`` / ``body`` / ``context`` / ``query``），便于断言与调试。
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    template_name: str
    sections: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# PromptModule                                                                  #
# --------------------------------------------------------------------------- #
class PromptModule:
    """提示词模板注册表与渲染器（Req 9.1, 20.1–20.5）。

    构造时注册一组系统内置具名模板（``rag_answer`` / ``react_agent`` /
    ``ops_planner`` / ``ops_replanner`` / ``memory_summary``）；其中规划与重规划模板
    标注为需要分步推理（``requires_step_reasoning=True``）。
    """

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        for template in _builtin_templates():
            self._templates[template.name] = template

    # ------------------------------------------------------------------ #
    # 注册 / 查找                                                          #
    # ------------------------------------------------------------------ #
    def register(self, template: PromptTemplate) -> None:
        """按唯一名称新增或**热替换**模板（Req 20.4）。

        若已存在同名模板，则以新模板覆盖之；调用方（各 Agent）无需改动调用代码
        即可获得更新后的模板内容。
        """
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        """按名称返回模板；不存在时抛 :class:`TemplateNotFoundError`（Req 20.5）。"""
        template = self._templates.get(name)
        if template is None:
            raise TemplateNotFoundError(
                f"提示词模板不存在：{name}",
                details={"template_name": name},
            )
        return template

    def has(self, name: str) -> bool:
        """判断指定名称的模板是否已注册。"""
        return name in self._templates

    @property
    def template_names(self) -> list[str]:
        """当前已注册的全部模板名称（无序）。"""
        return list(self._templates.keys())

    # ------------------------------------------------------------------ #
    # 渲染                                                                #
    # ------------------------------------------------------------------ #
    def build(
        self, template_name: str, variables: dict[str, Any] | None = None
    ) -> Prompt:
        """渲染具名模板为最终提示词（Req 20.1, 20.2, 20.3, 20.5）。

        渲染结果始终包含角色定义、任务目标与输出结构/格式约束说明；当模板标注
        ``requires_step_reasoning=True`` 时额外包含分步思考指令。``body_template``
        中的命名占位符以宽容方式用 ``variables`` 替换（缺失占位符原样保留，额外
        变量忽略，绝不崩溃）。

        Args:
            template_name: 目标模板名称。
            variables: 注入 ``body_template`` 的变量字典；为 ``None`` 时视为空。

        Returns:
            渲染后的 :class:`Prompt`。

        Raises:
            TemplateNotFoundError: 引用的具名模板不存在（Req 20.5）。
        """
        template = self.get(template_name)
        body_text = self._render_body(template.body_template, variables or {})
        text, sections = self._render(template, body_text=body_text)
        return Prompt(text=text, template_name=template.name, sections=sections)

    def build_rag_prompt(
        self,
        query: str,
        recalled: list[SearchHit],
        template_name: str = "rag_answer",
    ) -> Prompt:
        """构造 RAG 增强提示词（Req 9.1）。

        将用户查询与召回集合中**全部** Chunk 拼装为单条提示词，并为每个 Chunk 标注
        其来源文件标识（``SearchHit.source_id``）。Chunk 文本与查询均以拼接方式嵌入，
        不执行任何 ``str.format``，因此对包含花括号 / 任意 Unicode 的内容亦安全。

        最终 ``Prompt.text`` 必然包含用户查询，且召回集合中每个 Chunk 的来源文件
        标识均出现在文本中。

        Args:
            query: 用户查询文本。
            recalled: 召回的 Chunk 命中集合（顺序保留）。
            template_name: 基础模板名称（提供角色/目标/输出约束），默认 ``rag_answer``。

        Returns:
            渲染后的 :class:`Prompt`。

        Raises:
            TemplateNotFoundError: 基础模板不存在（Req 20.5）。
        """
        template = self.get(template_name)
        context_block = self._assemble_context(recalled)
        text, sections = self._render(
            template, context_block=context_block, query=query
        )
        return Prompt(text=text, template_name=template.name, sections=sections)

    # ------------------------------------------------------------------ #
    # 内部：渲染装配                                                       #
    # ------------------------------------------------------------------ #
    def _render(
        self,
        template: PromptTemplate,
        *,
        body_text: str | None = None,
        context_block: str | None = None,
        query: str | None = None,
    ) -> tuple[str, dict[str, str]]:
        """将模板与可选的主体/参考资料/查询装配为文本与结构化分节。

        始终输出角色 / 任务目标 / 输出约束三节（Req 20.1, 20.3）；需要分步推理时
        追加分步思考节（Req 20.2）。参考资料与查询通过纯拼接嵌入（brace 安全）。
        """
        sections: dict[str, str] = {
            "role": template.role,
            "task_goal": template.task_goal,
            "output_constraints": template.output_constraints,
        }
        parts: list[str] = [
            f"{_SECTION_ROLE}\n{template.role}",
            f"{_SECTION_TASK_GOAL}\n{template.task_goal}",
            f"{_SECTION_OUTPUT}\n{template.output_constraints}",
        ]

        if template.requires_step_reasoning:
            sections["step_reasoning"] = STEP_REASONING_INSTRUCTION
            parts.append(f"{_SECTION_STEP}\n{STEP_REASONING_INSTRUCTION}")

        if body_text:
            sections["body"] = body_text
            parts.append(f"{_SECTION_BODY}\n{body_text}")

        if context_block is not None:
            sections["context"] = context_block
            parts.append(f"{_SECTION_CONTEXT}\n{context_block}")

        if query is not None:
            sections["query"] = query
            parts.append(f"{_SECTION_QUESTION}\n{query}")

        return "\n\n".join(parts), sections

    @staticmethod
    def _render_body(body_template: str, variables: dict[str, Any]) -> str:
        """以宽容方式将变量替换进 ``body_template``（brace 安全）。

        仅对模板作者可控的 ``body_template`` 执行 ``str.format_map``；缺失占位符
        由 :class:`_ForgivingDict` 原样保留，额外变量被忽略。若模板字符串本身存在
        无法解析的格式（作者笔误），降级为返回原始模板文本，绝不向上抛出格式异常。
        """
        if not body_template:
            return ""
        try:
            return body_template.format_map(_ForgivingDict(variables))
        except (KeyError, IndexError, ValueError):
            # 防御性降级：模板格式异常时返回原文，保证构造不崩溃。
            return body_template

    @staticmethod
    def _assemble_context(recalled: list[SearchHit]) -> str:
        """将召回 Chunk 拼装为带来源标注的参考资料块（纯拼接，brace 安全，Req 9.1）。

        每个 Chunk 渲染为 ``[来源: <source_id>]\\n<chunk text>``，块间以空行分隔。
        来源标识与文本均原样嵌入，不执行任何格式化。
        """
        blocks: list[str] = []
        for index, hit in enumerate(recalled, start=1):
            blocks.append(f"[{index}] [来源: {hit.source_id}]\n{hit.text}")
        return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 内置模板                                                                      #
# --------------------------------------------------------------------------- #
def _builtin_templates() -> list[PromptTemplate]:
    """构造系统内置具名模板集合。

    规划（``ops_planner``）与重规划（``ops_replanner``）模板标注为需要分步推理。
    """
    return [
        PromptTemplate(
            name="rag_answer",
            role="你是企业级运维知识库问答助手，熟悉业务接入手册、告警处理手册与历史工单。",
            task_goal=(
                "依据“参考资料”中提供的知识库片段回答“用户问题”，"
                "仅使用片段中的信息作答，不臆造知识库以外的内容。"
            ),
            output_constraints=(
                "输出要求：使用简体中文；先给出直接答案，再补充必要依据；"
                "若参考资料不足以回答，请明确说明“未在知识库中检索到相关内容”；"
                "在答案末尾以列表形式给出所引用片段的来源文件标识。"
            ),
            requires_step_reasoning=False,
        ),
        PromptTemplate(
            name="react_agent",
            role="你是采用 ReAct 模式的智能对话运维助手，可在思考与行动之间交替并调用工具。",
            task_goal=(
                "理解用户问题，必要时调用工具或检索知识库获取信息，"
                "在获得足够信息后给出准确、可执行的回答。"
            ),
            output_constraints=(
                "输出要求：使用简体中文；需要调用工具时输出结构化的工具调用请求，"
                "无需调用工具时直接给出最终答案；保持回答简洁、聚焦问题。"
            ),
            requires_step_reasoning=False,
        ),
        PromptTemplate(
            name="ops_planner",
            role="你是资深运维规划工程师，负责将告警排查拆解为有序、可执行的步骤计划。",
            task_goal=(
                "依据告警信息与处理手册，制定一份由有序步骤组成的排查计划，"
                "为每个步骤标注待调用的工具与该步骤目标。"
            ),
            output_constraints=(
                "输出要求：以有序列表给出步骤；每个步骤包含 tool_name 与 goal 两个字段；"
                "步骤数量不超过配置的最大步骤数上限；无手册依据时明确标注计划为通用计划。"
            ),
            requires_step_reasoning=True,
        ),
        PromptTemplate(
            name="ops_replanner",
            role="你是资深运维重规划工程师，负责依据执行结果评估排查进展并决定后续动作。",
            task_goal=(
                "结合当前计划与最新执行结果，评估排查完成情况，"
                "在“任务已完成 / 未完成且剩余计划仍适用 / 未完成且剩余计划不再适用”中择一，"
                "必要时生成修订后的新计划。"
            ),
            output_constraints=(
                "输出要求：先给出三态评估结论之一；若需重规划则给出修订后的有序步骤计划；"
                "评估与计划均使用简体中文且结构清晰。"
            ),
            requires_step_reasoning=True,
        ),
        PromptTemplate(
            name="memory_summary",
            role="你是对话记忆总结助手，负责将较早的多轮对话压缩为简洁摘要。",
            task_goal="在不丢失关键事实、决策与未决事项的前提下，将给定的历史消息总结为简短摘要。",
            output_constraints=(
                "输出要求：使用简体中文；以要点形式给出摘要；"
                "保留实体、结论与待办，省略寒暄与冗余内容。"
            ),
            requires_step_reasoning=False,
        ),
    ]

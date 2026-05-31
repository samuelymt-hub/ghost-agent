"""Prompt_Module 测试（任务 7.2 / 7.3 / 7.4 / 7.5）。

包含：
- 任务 7.3 属性测试 Property 10（增强提示词包含查询与全部来源标识，Hypothesis,
  max_examples>=100）。
- 任务 7.4 属性测试 Property 12（Prompt 必含字段，max_examples>=100）。
- 任务 7.5 单元测试：同名模板热替换（Req 20.4）、引用缺失模板返回错误（Req 20.5）、
  花括号/Unicode 内容不崩溃、内置模板存在且 planner/replanner 需分步推理、
  body_template 变量替换（提供变量生效、缺失变量宽容不崩溃）。

外部依赖无（Prompt_Module 为纯逻辑组件）；召回集合以 :class:`SearchHit` 直接构造。
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ghost_agent.core.prompt_module import (
    STEP_REASONING_INSTRUCTION,
    Prompt,
    PromptModule,
    PromptTemplate,
)
from ghost_agent.models.errors import TemplateNotFoundError
from ghost_agent.models.vector_record import VectorType
from ghost_agent.vector_db.vector_store import SearchHit


# --------------------------------------------------------------------------- #
# 工厂                                                                          #
# --------------------------------------------------------------------------- #
def _hit(*, source_id: str, text: str, id: str = "h", score: float = 0.9) -> SearchHit:  # noqa: A002
    return SearchHit(
        id=id,
        text=text,
        source_id=source_id,
        vector_type=VectorType.DOC_CHUNK,
        score=score,
        metadata={},
    )


# =========================================================================== #
# 任务 7.3 — 属性测试 Property 10                                               #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 10: 对任意非空召回 Chunk 集合与用户查询，构造出的增强提示词包含该用户查询，且召回集合中每个 Chunk 的来源文件标识均出现在提示词中。
# Validates: Requirements 9.1


# 含花括号 / Unicode / 空白的文本，确保拼接（非 str.format）下不崩溃。
_text_strategy = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x4FFF),
    min_size=0,
    max_size=40,
)
# 来源标识非空（避免空串作为子串平凡命中），含花括号/Unicode。
_source_id_strategy = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x4FFF),
    min_size=1,
    max_size=24,
)
# 用户查询非空（去空白后非空），含花括号/Unicode。
_query_strategy = st.text(min_size=1, max_size=60).filter(lambda s: s.strip() != "")


@settings(max_examples=200, deadline=None)
@given(
    chunks=st.lists(
        st.tuples(_source_id_strategy, _text_strategy),
        min_size=1,  # 非空召回集合
        max_size=8,
    ),
    query=_query_strategy,
)
def test_property_10_rag_prompt_contains_query_and_all_sources(
    chunks: list[tuple[str, str]], query: str
):
    """Property 10：增强提示词包含用户查询，且每个 Chunk 来源标识均出现在提示词中。"""
    recalled = [
        _hit(source_id=source_id, text=text, id=f"c-{i}")
        for i, (source_id, text) in enumerate(chunks)
    ]
    module = PromptModule()

    prompt = module.build_rag_prompt(query, recalled)

    assert isinstance(prompt, Prompt)
    # 包含用户查询。
    assert query in prompt.text
    # 每个 Chunk 的来源文件标识均出现在提示词中。
    for source_id, _ in chunks:
        assert source_id in prompt.text


# =========================================================================== #
# 任务 7.4 — 属性测试 Property 12                                               #
# =========================================================================== #
# Feature: intelligent-oncall-agent, Property 12: 对任意提示词构造请求，生成的提示词均包含角色定义、任务目标与输出结构/格式约束说明；当模板被标注为需要分步推理时，提示词额外包含分步思考指令。
# Validates: Requirements 20.1, 20.2, 20.3


# 角色/目标/约束为非空字符串（min_size=1），含 Unicode；避免与小节标记/分步指令碰撞。
_field_strategy = st.text(min_size=1, max_size=50).filter(
    lambda s: s.strip() != "" and STEP_REASONING_INSTRUCTION not in s
)
_name_strategy = st.text(
    alphabet=st.characters(min_codepoint=0x41, max_codepoint=0x7A),
    min_size=1,
    max_size=16,
)


@settings(max_examples=200, deadline=None)
@given(
    role=_field_strategy,
    task_goal=_field_strategy,
    output_constraints=_field_strategy,
    requires_step_reasoning=st.booleans(),
    name=_name_strategy,
)
def test_property_12_prompt_contains_required_fields(
    role: str,
    task_goal: str,
    output_constraints: str,
    requires_step_reasoning: bool,
    name: str,
):
    """Property 12：提示词必含角色/目标/输出约束，需分步推理时额外含分步指令。"""
    module = PromptModule()
    module.register(
        PromptTemplate(
            name=name,
            role=role,
            task_goal=task_goal,
            output_constraints=output_constraints,
            requires_step_reasoning=requires_step_reasoning,
        )
    )

    prompt = module.build(name)

    # 始终包含角色定义、任务目标与输出结构/格式约束说明。
    assert role in prompt.text
    assert task_goal in prompt.text
    assert output_constraints in prompt.text
    # 结构化分节亦反映必含字段。
    assert prompt.sections["role"] == role
    assert prompt.sections["task_goal"] == task_goal
    assert prompt.sections["output_constraints"] == output_constraints
    # 分步思考指令当且仅当标注需要分步推理时出现。
    if requires_step_reasoning:
        assert STEP_REASONING_INSTRUCTION in prompt.text
        assert "step_reasoning" in prompt.sections
    else:
        assert STEP_REASONING_INSTRUCTION not in prompt.text
        assert "step_reasoning" not in prompt.sections


@settings(max_examples=100, deadline=None)
@given(name=st.sampled_from(PromptModule().template_names))
def test_property_12_holds_for_builtin_templates(name: str):
    """Property 12（内置模板分支）：每个内置模板渲染均含必含字段及（如需）分步指令。"""
    module = PromptModule()
    template = module.get(name)

    prompt = module.build(name)

    assert template.role in prompt.text
    assert template.task_goal in prompt.text
    assert template.output_constraints in prompt.text
    if template.requires_step_reasoning:
        assert STEP_REASONING_INSTRUCTION in prompt.text
    else:
        assert STEP_REASONING_INSTRUCTION not in prompt.text


# =========================================================================== #
# 任务 7.5 — 单元测试                                                           #
# =========================================================================== #

# --- 同名模板热替换（Req 20.4） --------------------------------------------- #
def test_register_hot_swaps_template_by_name():
    """同名模板热替换：register v2 覆盖 v1，构造无需改调用代码即反映新内容。"""
    module = PromptModule()
    module.register(
        PromptTemplate(
            name="x",
            role="角色 V1",
            task_goal="目标 V1",
            output_constraints="约束 V1",
        )
    )
    first = module.build("x")
    assert "角色 V1" in first.text

    # 以不同 role 热替换同名模板。
    module.register(
        PromptTemplate(
            name="x",
            role="角色 V2",
            task_goal="目标 V2",
            output_constraints="约束 V2",
        )
    )
    second = module.build("x")  # 调用方代码完全相同
    assert "角色 V2" in second.text
    assert "角色 V1" not in second.text
    assert "目标 V2" in second.text


def test_register_does_not_grow_when_replacing_same_name():
    """热替换同名模板不应增加模板数量。"""
    module = PromptModule()
    before = len(module.template_names)
    module.register(
        PromptTemplate(name="dup", role="r", task_goal="g", output_constraints="o")
    )
    after_add = len(module.template_names)
    module.register(
        PromptTemplate(name="dup", role="r2", task_goal="g2", output_constraints="o2")
    )
    after_replace = len(module.template_names)
    assert after_add == before + 1
    assert after_replace == after_add  # 替换不增长


# --- 引用缺失模板返回错误（Req 20.5） --------------------------------------- #
def test_build_unknown_template_raises_template_not_found():
    module = PromptModule()
    with pytest.raises(TemplateNotFoundError) as exc_info:
        module.build("does-not-exist")
    # 错误信息/details 指明缺失模板名（Req 20.5）。
    assert "does-not-exist" in exc_info.value.message
    assert exc_info.value.details == {"template_name": "does-not-exist"}


def test_get_unknown_template_raises_template_not_found():
    module = PromptModule()
    with pytest.raises(TemplateNotFoundError):
        module.get("missing")


def test_build_rag_prompt_unknown_template_raises():
    module = PromptModule()
    with pytest.raises(TemplateNotFoundError):
        module.build_rag_prompt("查询", [_hit(source_id="s", text="t")], template_name="nope")


# --- 花括号 / Unicode 内容不崩溃（brace 安全） ------------------------------ #
def test_build_rag_prompt_with_braces_and_unicode_does_not_crash():
    recalled = [
        _hit(source_id="src-{a}", text="包含花括号 {not_a_var} 与 emoji 🚀 的内容", id="c1"),
        _hit(source_id="文件-2", text="另一段 {0} {} {{}} 文本", id="c2"),
    ]
    module = PromptModule()
    query = "如何处理 {alarm} 告警？包含 } 与 { 字符"

    prompt = module.build_rag_prompt(query, recalled)

    # 查询与全部来源标识与文本内容均原样出现。
    assert query in prompt.text
    assert "src-{a}" in prompt.text
    assert "文件-2" in prompt.text
    assert "包含花括号 {not_a_var} 与 emoji 🚀 的内容" in prompt.text
    assert "另一段 {0} {} {{}} 文本" in prompt.text


# --- 内置模板存在且 planner/replanner 需分步推理 ---------------------------- #
def test_builtin_templates_registered():
    module = PromptModule()
    names = set(module.template_names)
    expected = {"rag_answer", "react_agent", "ops_planner", "ops_replanner", "memory_summary"}
    assert expected.issubset(names)


def test_planner_and_replanner_require_step_reasoning():
    module = PromptModule()
    assert module.get("ops_planner").requires_step_reasoning is True
    assert module.get("ops_replanner").requires_step_reasoning is True
    # RAG / 对话 / 总结模板默认不需要分步推理。
    assert module.get("rag_answer").requires_step_reasoning is False
    assert module.get("react_agent").requires_step_reasoning is False
    assert module.get("memory_summary").requires_step_reasoning is False


# --- body_template 变量替换：提供变量生效、缺失变量宽容不崩溃 ----------------- #
def test_build_substitutes_provided_variables_into_body():
    module = PromptModule()
    module.register(
        PromptTemplate(
            name="with_body",
            role="r",
            task_goal="g",
            output_constraints="o",
            body_template="告警类型={alarm_type}，目标={target}",
        )
    )
    prompt = module.build("with_body", {"alarm_type": "CPU 高", "target": "svc-a"})
    assert "告警类型=CPU 高，目标=svc-a" in prompt.text


def test_build_missing_variable_is_forgiving_and_preserves_placeholder():
    module = PromptModule()
    module.register(
        PromptTemplate(
            name="partial_body",
            role="r",
            task_goal="g",
            output_constraints="o",
            body_template="已知={known}，未知={unknown}",
        )
    )
    # 仅提供 known，缺失 unknown 不应崩溃，占位符原样保留。
    prompt = module.build("partial_body", {"known": "值A"})
    assert "已知=值A" in prompt.text
    assert "{unknown}" in prompt.text


def test_build_ignores_extra_variables():
    module = PromptModule()
    module.register(
        PromptTemplate(
            name="extra_body",
            role="r",
            task_goal="g",
            output_constraints="o",
            body_template="仅={used}",
        )
    )
    prompt = module.build("extra_body", {"used": "X", "unused": "Y"})
    assert "仅=X" in prompt.text
    # 额外变量被忽略，不出现其值。
    assert "Y" not in prompt.sections.get("body", "")


def test_build_no_variables_when_body_empty():
    """无 body_template 的模板渲染不包含 body 分节。"""
    module = PromptModule()
    module.register(
        PromptTemplate(name="nobody", role="r", task_goal="g", output_constraints="o")
    )
    prompt = module.build("nobody")
    assert "body" not in prompt.sections

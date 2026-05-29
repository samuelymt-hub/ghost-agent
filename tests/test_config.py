"""配置管理模块单元测试 (任务 1.3)。

覆盖范围：
- 默认值加载（Req 23.1）。
- 参数范围越界拒绝（Req 1.5/2.8/7.3/10.5/10.6/10.7/13.5/8.2/18.2/6.1）。
- 技术栈 guard 合法/非法值（Req 23.6, 23.7）。
- ``get_settings`` 单例语义。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ghost_agent.config import (
    Settings,
    TechStack,
    get_settings,
    validate_tech_stack,
)


# --------------------------------------------------------------------------- #
# 默认值                                                                        #
# --------------------------------------------------------------------------- #
def test_defaults_match_design_table():
    """默认值应与 design.md "超时与重试参数（可配置）" 表保持一致。"""
    s = Settings()

    assert s.tech_stack is TechStack.PYTHON
    assert s.chat_timeout_seconds == 60.0
    assert s.sse_idle_timeout_seconds == 30.0
    assert s.embedding_max_retries == 3
    assert s.react_max_iterations == 10
    assert s.tool_call_timeout_seconds == 30.0
    assert s.model_call_timeout_seconds == 60.0
    assert s.max_replan_count == 10
    assert s.history_message_top_k == 5
    assert s.retrieval_top_k == 5
    assert 0.0 <= s.min_similarity_threshold <= 1.0
    assert s.short_term_memory_limit >= 1
    assert s.chunk_min_length < s.chunk_max_length
    assert s.chunk_max_length <= s.embedding_max_input_length
    assert {"txt", "md", "pdf", "docx", "html"}.issubset(
        s.supported_document_types
    )
    assert s.max_file_size_bytes > 0
    assert s.max_plan_steps >= 1


# --------------------------------------------------------------------------- #
# 参数范围越界                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        # 嵌入重试 0–5
        ("embedding_max_retries", -1),
        ("embedding_max_retries", 6),
        # ReAct 迭代 1–50
        ("react_max_iterations", 0),
        ("react_max_iterations", 51),
        # 模型调用超时 1–120s
        ("model_call_timeout_seconds", 0),
        ("model_call_timeout_seconds", 121),
        # 工具调用超时 1–300s
        ("tool_call_timeout_seconds", 0),
        ("tool_call_timeout_seconds", 301),
        # 重规划 1–50
        ("max_replan_count", 0),
        ("max_replan_count", 51),
        # 检索 Top-K 1–100
        ("retrieval_top_k", 0),
        ("retrieval_top_k", 101),
        # 历史消息 Top-K 1–50
        ("history_message_top_k", 0),
        ("history_message_top_k", 51),
        # 短期记忆容量 ≥1
        ("short_term_memory_limit", 0),
        # 相似度阈值 [0,1]
        ("min_similarity_threshold", -0.01),
        ("min_similarity_threshold", 1.01),
        # 文件大小上限必须 > 0
        ("max_file_size_bytes", 0),
        # 计划步骤上限 1–100
        ("max_plan_steps", 0),
        ("max_plan_steps", 101),
    ],
)
def test_out_of_range_values_rejected(field: str, bad_value: object):
    """越界参数应直接被 Pydantic 校验拒绝。"""
    with pytest.raises(ValidationError):
        Settings(**{field: bad_value})


def test_chunk_length_cross_field_rejected():
    """chunk_min_length 必须严格小于 chunk_max_length。"""
    with pytest.raises(ValidationError):
        Settings(chunk_min_length=500, chunk_max_length=500)
    with pytest.raises(ValidationError):
        Settings(chunk_min_length=1000, chunk_max_length=500)


def test_chunk_max_length_cannot_exceed_embedding_input_length():
    """chunk_max_length 不得超过 embedding_max_input_length。"""
    with pytest.raises(ValidationError):
        Settings(
            chunk_min_length=100,
            chunk_max_length=8192,
            embedding_max_input_length=4096,
        )


# --------------------------------------------------------------------------- #
# 技术栈校验                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("python", TechStack.PYTHON),
        ("Python", TechStack.PYTHON),
        ("PYTHON", TechStack.PYTHON),
        ("go", TechStack.GO),
        ("Go", TechStack.GO),
        ("java", TechStack.JAVA),
        ("  java  ", TechStack.JAVA),
    ],
)
def test_validate_tech_stack_accepts_supported_values(
    raw: str, expected: TechStack
):
    assert validate_tech_stack(raw) is expected


@pytest.mark.parametrize(
    "raw",
    ["rust", "node", "kotlin", "", "  ", "py thon", "javascript"],
)
def test_validate_tech_stack_rejects_unsupported_values(raw: str):
    with pytest.raises(ValueError) as exc_info:
        validate_tech_stack(raw)
    # 错误信息须含中文提示，便于运维快速定位（Req 23.7）
    assert "不支持该技术栈" in str(exc_info.value) or "技术栈不能为空" in str(
        exc_info.value
    )


def test_validate_tech_stack_rejects_none():
    with pytest.raises(ValueError):
        validate_tech_stack(None)  # type: ignore[arg-type]


def test_settings_rejects_invalid_tech_stack_via_field():
    """通过 Settings 字段层面同样应拒绝非法技术栈。"""
    with pytest.raises(ValidationError):
        Settings(tech_stack="rust")  # type: ignore[arg-type]


def test_settings_accepts_tech_stack_string_case_insensitive():
    s = Settings(tech_stack="JAVA")  # type: ignore[arg-type]
    assert s.tech_stack is TechStack.JAVA


# --------------------------------------------------------------------------- #
# 环境变量加载                                                                  #
# --------------------------------------------------------------------------- #
def test_env_var_loading(monkeypatch: pytest.MonkeyPatch):
    """通过环境变量覆盖默认值。"""
    monkeypatch.setenv("TECH_STACK", "go")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "12")
    monkeypatch.setenv("EMBEDDING_MAX_RETRIES", "5")

    s = Settings()
    assert s.tech_stack is TechStack.GO
    assert s.retrieval_top_k == 12
    assert s.embedding_max_retries == 5


def test_env_var_invalid_tech_stack_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TECH_STACK", "rust")
    with pytest.raises(ValidationError):
        Settings()


# --------------------------------------------------------------------------- #
# get_settings 单例                                                            #
# --------------------------------------------------------------------------- #
def test_get_settings_returns_cached_singleton():
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second


def test_get_settings_cache_clear_returns_fresh_instance():
    get_settings.cache_clear()
    first = get_settings()
    get_settings.cache_clear()
    second = get_settings()
    assert first is not second

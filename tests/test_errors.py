"""统一错误模型与领域异常类的单元测试（example-based）。

覆盖：
* :class:`ErrorResponse` 的字段校验、序列化与 enum/字符串兼容。
* :class:`GhostAgentError` 基类的 ``to_response`` 转换与 ``details`` 透传。
* 每个领域异常类与 :class:`ErrorCode` 的一一对应关系。
* ``ErrorCode`` 取值唯一性。
"""

from __future__ import annotations

import inspect
import json

import pytest

from ghost_agent.models import errors as errors_module
from ghost_agent.models.errors import (
    DimensionMismatchError,
    EmptyMessageError,
    ErrorCode,
    ErrorResponse,
    FileTooLargeError,
    GhostAgentError,
    ToolNotFoundError,
    ToolValidationError,
)


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------


def test_error_response_with_details_round_trips() -> None:
    resp = ErrorResponse(
        error_code=ErrorCode.FILE_TOO_LARGE,
        message="文件超过单文件大小上限",
        details={"max_size_mb": 50, "actual_size_mb": 80},
    )

    assert resp.error_code == "FILE_TOO_LARGE"
    assert resp.message == "文件超过单文件大小上限"
    assert resp.details == {"max_size_mb": 50, "actual_size_mb": 80}

    payload = json.loads(resp.model_dump_json())
    assert payload == {
        "error_code": "FILE_TOO_LARGE",
        "message": "文件超过单文件大小上限",
        "details": {"max_size_mb": 50, "actual_size_mb": 80},
    }


def test_error_response_without_details_defaults_to_none() -> None:
    resp = ErrorResponse(error_code=ErrorCode.EMPTY_MESSAGE, message="消息为空")

    assert resp.details is None
    payload = resp.model_dump()
    assert payload["details"] is None
    assert payload["error_code"] == "EMPTY_MESSAGE"


def test_error_response_accepts_bare_string_code() -> None:
    """允许调用方使用裸字符串作为 error_code（跨语言契约场景）。"""
    resp = ErrorResponse(error_code="EMPTY_MESSAGE", message="x")
    assert resp.error_code == "EMPTY_MESSAGE"


def test_error_response_serializes_enum_as_string_not_repr() -> None:
    """JSON 输出必须是 ``"EMPTY_MESSAGE"``，而非 ``"ErrorCode.EMPTY_MESSAGE"``。"""
    resp = ErrorResponse(error_code=ErrorCode.EMPTY_MESSAGE, message="x")
    raw = resp.model_dump_json()
    assert '"error_code":"EMPTY_MESSAGE"' in raw
    assert "ErrorCode." not in raw


def test_error_response_rejects_invalid_code_type() -> None:
    with pytest.raises(Exception):  # ValidationError，包装了上面的 TypeError
        ErrorResponse(error_code=123, message="x")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GhostAgentError 基类
# ---------------------------------------------------------------------------


def test_base_exception_to_response_carries_message_and_details() -> None:
    exc = EmptyMessageError("消息为空", details={"hint": "trim"})
    resp = exc.to_response()

    assert isinstance(resp, ErrorResponse)
    assert resp.error_code == ErrorCode.EMPTY_MESSAGE.value
    assert resp.message == "消息为空"
    assert resp.details == {"hint": "trim"}


def test_base_exception_to_response_without_details_is_none() -> None:
    exc = ToolNotFoundError("工具不存在")
    resp = exc.to_response()

    assert resp.error_code == "TOOL_NOT_FOUND"
    assert resp.details is None


def test_base_exception_str_returns_message() -> None:
    exc = DimensionMismatchError("维度不一致")
    assert str(exc) == "维度不一致"


def test_domain_exception_is_subclass_of_base_and_exception() -> None:
    exc = ToolValidationError("缺少必填参数 foo")
    assert isinstance(exc, GhostAgentError)
    assert isinstance(exc, Exception)

    # 可被 GhostAgentError 一并捕获
    with pytest.raises(GhostAgentError):
        raise FileTooLargeError("too large")


# ---------------------------------------------------------------------------
# ErrorCode 唯一性 & 异常映射完备性
# ---------------------------------------------------------------------------


def test_error_code_values_are_unique() -> None:
    values = [member.value for member in ErrorCode]
    assert len(values) == len(set(values)), "ErrorCode 取值出现重复"


def _domain_exception_classes() -> list[type[GhostAgentError]]:
    classes: list[type[GhostAgentError]] = []
    for _, obj in inspect.getmembers(errors_module, inspect.isclass):
        if obj is GhostAgentError:
            continue
        if issubclass(obj, GhostAgentError) and obj.__module__ == errors_module.__name__:
            classes.append(obj)
    return classes


def test_every_domain_exception_has_valid_error_code() -> None:
    """每个领域异常类必须绑定一个 ``ErrorCode`` 成员。"""
    for cls in _domain_exception_classes():
        code = cls.error_code
        assert isinstance(code, ErrorCode), f"{cls.__name__} 的 error_code 不是 ErrorCode"


def test_domain_exceptions_cover_all_error_codes() -> None:
    """每个 ``ErrorCode`` 至少有一个对应领域异常类（保证调用方可按类型捕获）。"""
    used = {cls.error_code for cls in _domain_exception_classes()}
    missing = set(ErrorCode) - used
    assert not missing, f"以下 ErrorCode 缺少对应领域异常类: {sorted(c.value for c in missing)}"


def test_domain_exception_codes_are_unique() -> None:
    """避免两个异常类映射到同一个 ErrorCode（保证 1:1 关系）。"""
    codes = [cls.error_code for cls in _domain_exception_classes()]
    assert len(codes) == len(set(codes)), "存在多个领域异常类绑定到同一个 ErrorCode"


# ---------------------------------------------------------------------------
# 显式覆盖几个高频异常的 error_code 绑定（防止重命名时静默漂移）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_cls, expected_code",
    [
        (EmptyMessageError, ErrorCode.EMPTY_MESSAGE),
        (FileTooLargeError, ErrorCode.FILE_TOO_LARGE),
        (DimensionMismatchError, ErrorCode.DIMENSION_MISMATCH),
        (ToolValidationError, ErrorCode.TOOL_VALIDATION_ERROR),
        (ToolNotFoundError, ErrorCode.TOOL_NOT_FOUND),
    ],
)
def test_specific_exception_code_bindings(
    exc_cls: type[GhostAgentError], expected_code: ErrorCode
) -> None:
    assert exc_cls.error_code is expected_code
    instance = exc_cls("msg")
    assert instance.to_response().error_code == expected_code.value

"""
title: Tests for AssertStmt lowering and runtime failure reporting.
"""

from __future__ import annotations

import pytest

from irx import astx
from irx.analysis import SemanticError
from irx.builder import Builder as LLVMBuilder
from irx.builder.runtime.assertions import (
    ASSERT_FAILURE_SYMBOL_NAME,
    ASSERT_RUNTIME_FEATURE_NAME,
    AssertionFailureReport,
    parse_assert_failure_line,
    parse_assert_failure_output,
)
from irx.system import AssertStmt

from .conftest import assert_ir_parses, build_and_run


def _module_with_assert(
    assert_stmt: AssertStmt,
    *,
    module_name: str = "tests/main.x",
) -> astx.Module:
    """
    title: Build one small module whose main body contains one assertion.
    parameters:
      assert_stmt:
        type: AssertStmt
      module_name:
        type: str
    returns:
      type: astx.Module
    """
    module = astx.Module(name=module_name)
    main_proto = astx.FunctionPrototype(
        "main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    body = astx.Block()
    body.append(assert_stmt)
    body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))
    return module


def test_assert_stmt_translation_activates_runtime_feature() -> None:
    """
    title: AssertStmt lowering should declare the assertion runtime helper.
    """
    builder = LLVMBuilder()
    module = _module_with_assert(
        AssertStmt(
            astx.LiteralBoolean(value=False),
            astx.LiteralUTF8String("boom"),
            loc=astx.SourceLocation(line=12, col=3),
        )
    )

    ir_text = builder.translate(module)
    active_features = (
        builder.translator.runtime_features.active_feature_names()
    )
    native_artifacts = builder.translator.runtime_features.native_artifacts()

    assert ASSERT_RUNTIME_FEATURE_NAME in active_features
    assert ASSERT_FAILURE_SYMBOL_NAME in ir_text
    assert 'call void @"__arx_assert_fail"' in ir_text
    assert any(
        artifact.path.name == "irx_assert_runtime.c"
        for artifact in native_artifacts
    )
    assert_ir_parses(ir_text)


def test_assert_stmt_passes_without_failure_output() -> None:
    """
    title: A passing AssertStmt should leave the process exit code at zero.
    """
    builder = LLVMBuilder()
    module = _module_with_assert(AssertStmt(astx.LiteralBoolean(value=True)))

    result = build_and_run(builder, module)

    assert result.returncode == 0
    assert parse_assert_failure_output(result.stderr) is None


def test_assert_stmt_failure_uses_default_message() -> None:
    """
    title: >-
      A failing AssertStmt should report source, location, and default text.
    """
    builder = LLVMBuilder()
    expected_exit_code = 1
    expected_failure = AssertionFailureReport(
        source="tests/main.x",
        line=12,
        col=3,
        message="assertion failed",
    )
    module = _module_with_assert(
        AssertStmt(
            astx.LiteralBoolean(value=False),
            loc=astx.SourceLocation(
                line=expected_failure.line, col=expected_failure.col
            ),
        )
    )

    result = build_and_run(builder, module)

    assert result.returncode == expected_exit_code
    assert parse_assert_failure_output(result.stderr) == expected_failure


def test_assert_stmt_failure_uses_custom_message() -> None:
    """
    title: A failing AssertStmt should preserve one explicit message string.
    """
    builder = LLVMBuilder()
    expected_failure = AssertionFailureReport(
        source="tests/main.x",
        line=8,
        col=5,
        message="fib(5) should be 5",
    )
    module = _module_with_assert(
        AssertStmt(
            astx.LiteralBoolean(value=False),
            astx.LiteralUTF8String(expected_failure.message),
            loc=astx.SourceLocation(
                line=expected_failure.line, col=expected_failure.col
            ),
        )
    )

    result = build_and_run(builder, module)

    assert result.returncode == 1
    assert parse_assert_failure_output(result.stderr) == expected_failure


def test_parse_assert_failure_line_preserves_pipe_characters() -> None:
    """
    title: >-
      Assertion failure parsing should preserve pipe-delimited message text.
    """
    line = "ARX_ASSERT_FAIL|tests/main.x|12|3|left|right"
    expected_failure = AssertionFailureReport(
        source="tests/main.x",
        line=12,
        col=3,
        message="left|right",
    )

    assert parse_assert_failure_line(line) == expected_failure


def test_assert_stmt_requires_boolean_condition() -> None:
    """
    title: Semantic analysis should reject non-Boolean assertion conditions.
    """
    builder = LLVMBuilder()
    module = _module_with_assert(AssertStmt(astx.LiteralInt32(1)))

    with pytest.raises(
        SemanticError, match="assert condition must be Boolean"
    ):
        builder.translate(module)

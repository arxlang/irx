"""
title: Tests for LiteralDateTime lowering using project conventions.
"""

from __future__ import annotations

import shutil

from typing import cast

import pytest

from irx import astx
from irx.analysis import SemanticError
from irx.builders.llvmliteir import Builder as LLVMBuilder
from irx.builders.llvmliteir import Visitor as LLVMVisitor
from llvmlite import ir

from .conftest import (
    assert_build_succeeds,
    assert_ir_parses,
    make_main_module,
    translate_ir,
)

HAS_LITERAL_DATETIME = hasattr(astx, "LiteralDateTime")
HAS_CLANG = shutil.which("clang") is not None

pytestmark = pytest.mark.skipif(
    not HAS_LITERAL_DATETIME,
    reason="astx.LiteralDateTime not available",
)


def _datetime_values(const: ir.Constant) -> list[int]:
    """
    title: Extract lowered datetime fields from a literal struct constant.
    parameters:
      const:
        type: ir.Constant
    returns:
      type: list[int]
    """
    items = cast(list[ir.Constant], const.constant)
    return [cast(int, item.constant) for item in items]


@pytest.mark.parametrize(
    ("datetime_str", "expected_values"),
    [
        ("2025-10-30T12:34:56", [2025, 10, 30, 12, 34, 56]),
        ("2025-10-30 12:34", [2025, 10, 30, 12, 34, 0]),
        ("2025-01-02 03:04:05", [2025, 1, 2, 3, 4, 5]),
        ("2024-12-31T23:59:59", [2024, 12, 31, 23, 59, 59]),
    ],
)
def test_literal_datetime_lowers_expected_struct(
    llvm_visitor: LLVMVisitor,
    datetime_str: str,
    expected_values: list[int],
) -> None:
    """
    title: >-
      LiteralDateTime should lower to the expected constant struct payload.
    parameters:
      llvm_visitor:
        type: LLVMVisitor
      datetime_str:
        type: str
      expected_values:
        type: list[int]
    """
    llvm_visitor.visit(astx.LiteralDateTime(datetime_str))
    const = llvm_visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert const.type == llvm_visitor._llvm.DATETIME_TYPE
    assert _datetime_values(const) == expected_values
    assert not llvm_visitor.result_stack


def test_literal_datetime_translate_smoke(
    llvm_builder: LLVMBuilder,
) -> None:
    """
    title: LiteralDateTime should survive the translate pipeline.
    parameters:
      llvm_builder:
        type: LLVMBuilder
    """
    module = make_main_module(
        astx.LiteralDateTime("2025-10-30T12:34:56"),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    ir_text = translate_ir(llvm_builder, module)

    assert 'define i32 @"main"()' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.skipif(not HAS_CLANG, reason="clang not available")
def test_literal_datetime_build_smoke(
    llvm_builder: LLVMBuilder,
) -> None:
    """
    title: LiteralDateTime should survive the build pipeline.
    parameters:
      llvm_builder:
        type: LLVMBuilder
    """
    module = make_main_module(
        astx.LiteralDateTime("2025-10-30T12:34:56"),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    assert_build_succeeds(llvm_builder, module)


@pytest.mark.parametrize(
    ("datetime_str", "message"),
    [
        ("2025-10-30T12:34:56.123", "fractional seconds are not supported"),
        ("2025-10-30T12:34:56Z", "timezone offsets are not supported"),
        ("2025-13-01T00:00:00", "invalid calendar date/time"),
        ("2025-12-32T00:00:00", "invalid calendar date/time"),
        ("2025-10-30T24:00:00", "hour out of range"),
        ("2025-10-30T12:60:00", "minute out of range"),
        ("2025-10-30T12:34:60", "second out of range"),
    ],
)
def test_literal_datetime_invalid_inputs_raise_semantic_error(
    llvm_builder: LLVMBuilder,
    datetime_str: str,
    message: str,
) -> None:
    """
    title: Invalid LiteralDateTime values should fail during semantic analysis.
    parameters:
      llvm_builder:
        type: LLVMBuilder
      datetime_str:
        type: str
      message:
        type: str
    """
    module = make_main_module(
        astx.LiteralDateTime(datetime_str),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    with pytest.raises(SemanticError, match=message):
        translate_ir(llvm_builder, module)

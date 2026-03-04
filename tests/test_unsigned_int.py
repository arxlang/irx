"""
title: Tests for unsigned integer literal lowering.
"""

from __future__ import annotations

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_uint8(builder_class: Type[Builder]) -> None:
    """
    title: LiteralUInt8 lowers to i8 constant.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralUInt8(200))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.IntType)
    assert const.type.width == 8  # noqa: PLR2004
    assert not visitor.result_stack


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_uint16(builder_class: Type[Builder]) -> None:
    """
    title: LiteralUInt16 lowers to i16 constant.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralUInt16(50000))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.IntType)
    assert const.type.width == 16  # noqa: PLR2004
    assert not visitor.result_stack


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_uint32(builder_class: Type[Builder]) -> None:
    """
    title: LiteralUInt32 lowers to i32 constant.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralUInt32(42))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.IntType)
    assert const.type.width == 32  # noqa: PLR2004
    expected = ir.Constant(ir.IntType(32), 42)
    assert str(const) == str(expected)
    assert not visitor.result_stack


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_uint64(builder_class: Type[Builder]) -> None:
    """
    title: LiteralUInt64 lowers to i64 constant.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralUInt64(2**40))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.IntType)
    assert const.type.width == 64  # noqa: PLR2004
    assert not visitor.result_stack


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_uint128(builder_class: Type[Builder]) -> None:
    """
    title: LiteralUInt128 lowers to i128 constant.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralUInt128(2**100))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.IntType)
    assert const.type.width == 128  # noqa: PLR2004
    assert not visitor.result_stack


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_uint32_zero(builder_class: Type[Builder]) -> None:
    """
    title: LiteralUInt32 zero value lowers correctly.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralUInt32(0))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.IntType)
    assert const.type.width == 32  # noqa: PLR2004
    expected = ir.Constant(ir.IntType(32), 0)
    assert str(const) == str(expected)
    assert not visitor.result_stack

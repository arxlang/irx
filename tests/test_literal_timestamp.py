"""
title: Tests for LiteralTimestamp lowering using project conventions.
"""

from __future__ import annotations

import re

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from irx.system import PrintExpr
from llvmlite import ir

from .conftest import check_result

HAS_LITERAL_TIMESTAMP = hasattr(astx, "LiteralTimestamp")
NANOS_PER_MILLISECOND = 123_000_000


def _timestamp_values(const: ir.Constant) -> list[int]:
    """
    title: Extract i32 values from the literal struct constant.
    parameters:
      const:
        type: ir.Constant
    returns:
      type: list[int]
    """
    return [int(v) for v in re.findall(r"i32\s+(-?\d+)", str(const))]


@pytest.mark.skipif(
    not HAS_LITERAL_TIMESTAMP, reason="astx.LiteralTimestamp not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_timestamp_basic(builder_class: Type[Builder]) -> None:
    """
    title: LiteralTimestamp with fractional seconds via 'T' separator.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    visitor.visit(astx.LiteralTimestamp("2025-10-30T12:34:56.123"))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert const.type == visitor._llvm.TIMESTAMP_TYPE

    vals = _timestamp_values(const)
    assert vals[:6] == [2025, 10, 30, 12, 34, 56]
    assert vals[6] == NANOS_PER_MILLISECOND


@pytest.mark.skipif(
    not HAS_LITERAL_TIMESTAMP, reason="astx.LiteralTimestamp not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_timestamp_fraction_truncated(
    builder_class: Type[Builder],
) -> None:
    """
    title: Fractions longer than 9 digits are truncated to nanoseconds.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    visitor.visit(astx.LiteralTimestamp("2025-01-02 03:04:05.1234567897"))
    const = visitor.result_stack.pop()

    vals = _timestamp_values(const)
    assert vals == [2025, 1, 2, 3, 4, 5, 123_456_789]


@pytest.mark.skipif(
    not HAS_LITERAL_TIMESTAMP, reason="astx.LiteralTimestamp not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_timestamp_invalid_date(builder_class: Type[Builder]) -> None:
    """
    title: Reject impossible calendar dates (e.g., February 30).
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="invalid date"):
        visitor.visit(astx.LiteralTimestamp("2025-02-30T00:00:00"))


@pytest.mark.skipif(
    not HAS_LITERAL_TIMESTAMP, reason="astx.LiteralTimestamp not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_timestamp_timezone_rejected(
    builder_class: Type[Builder],
) -> None:
    """
    title: Reject timestamps that include timezone markers.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="timezone"):
        visitor.visit(astx.LiteralTimestamp("2025-10-30T12:34:56Z"))


def test_literal_timestamp_valid() -> None:
    """
    title: Test valid LiteralTimestamp (lines 1726-1810).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("2025-03-06T14:30:00")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(PrintExpr(astx.LiteralUTF8String("2025-03-06T14:30:00")))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result(
        "build", builder, module, expected_output="2025-03-06T14:30:00"
    )


def test_literal_timestamp_with_space() -> None:
    """
    title: Test LiteralTimestamp with space separator.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("2025-03-06 14:30:00")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(PrintExpr(astx.LiteralUTF8String("2025-03-06 14:30:00")))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result(
        "build", builder, module, expected_output="2025-03-06 14:30:00"
    )


def test_literal_timestamp_invalid_format() -> None:
    """
    title: Test LiteralTimestamp with invalid format (line 1734).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("20250306")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="LiteralTimestamp"):
        check_result("build", builder, module)


def test_literal_timestamp_hour_out_of_range() -> None:
    """
    title: Test LiteralTimestamp with out-of-range hour (line 1797).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("2025-03-06T25:00:00")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="hour out of range"):
        check_result("build", builder, module)


def test_literal_timestamp_minute_out_of_range() -> None:
    """
    title: Test LiteralTimestamp with out-of-range minute (line 1801).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("2025-03-06T12:60:00")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="minute out of range"):
        check_result("build", builder, module)


def test_literal_timestamp_second_out_of_range() -> None:
    """
    title: Test LiteralTimestamp with out-of-range second (line 1805).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("2025-03-06T12:30:61")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="second out of range"):
        check_result("build", builder, module)

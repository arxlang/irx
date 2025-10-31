"""Tests for LiteralTimestamp lowering using project conventions."""

from __future__ import annotations

import re

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

HAS_LITERAL_TIMESTAMP = hasattr(astx, "LiteralTimestamp")
NANOS_PER_MILLISECOND = 123_000_000


def _timestamp_values(const: ir.Constant) -> list[int]:
    """Extract i32 values from the literal struct constant."""
    return [int(v) for v in re.findall(r"i32\s+(-?\d+)", str(const))]


@pytest.mark.skipif(
    not HAS_LITERAL_TIMESTAMP, reason="astx.LiteralTimestamp not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_timestamp_basic(builder_class: Type[Builder]) -> None:
    """LiteralTimestamp with fractional seconds via 'T' separator."""
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
    """Fractions longer than 9 digits are truncated to nanoseconds."""
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
    """Reject impossible calendar dates (e.g., February 30)."""
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
    """Reject timestamps that include timezone markers."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="timezone"):
        visitor.visit(astx.LiteralTimestamp("2025-10-30T12:34:56Z"))

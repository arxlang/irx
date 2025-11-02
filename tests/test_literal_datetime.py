"""Tests for LiteralDateTime lowering using project conventions."""

from __future__ import annotations

import re

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

HAS_LITERAL_DATETIME = hasattr(astx, "LiteralDateTime")


def _datetime_values(const: ir.Constant) -> list[int]:
    """Extract i32 values from the literal struct constant."""
    return [int(v) for v in re.findall(r"i32\s+(-?\d+)", str(const))]


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_basic_hms(builder_class: Type[Builder]) -> None:
    """LiteralDateTime with hour:minute:second via 'T' separator."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    visitor.visit(astx.LiteralDateTime("2025-10-30T12:34:56"))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)

    vals = _datetime_values(const)
    assert vals == [2025, 10, 30, 12, 34, 56]


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_basic_hm(builder_class: Type[Builder]) -> None:
    """LiteralDateTime with hour:minute only (defaults to :00 for seconds)."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    visitor.visit(astx.LiteralDateTime("2025-10-30 12:34"))
    const = visitor.result_stack.pop()

    vals = _datetime_values(const)
    assert vals == [2025, 10, 30, 12, 34, 0]


@pytest.mark.parametrize(
    "datetime_str, expected_values",
    [
        ("2025-10-30T12:34:56", [2025, 10, 30, 12, 34, 56]),
        ("2025-01-02 03:04:05", [2025, 1, 2, 3, 4, 5]),
        ("2024-12-31T23:59:59", [2024, 12, 31, 23, 59, 59]),
    ],
)
@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_parsing(
    builder_class: Type[Builder],
    datetime_str: str,
    expected_values: list[int],
) -> None:
    """Parse various datetime formats correctly."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    visitor.visit(astx.LiteralDateTime(datetime_str))
    const = visitor.result_stack.pop()

    vals = _datetime_values(const)
    assert vals == expected_values


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_fractional_rejected(
    builder_class: Type[Builder],
) -> None:
    """Reject timestamps with fractional seconds."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="fractional seconds"):
        visitor.visit(astx.LiteralDateTime("2025-10-30T12:34:56.123"))


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_timezone_rejected(
    builder_class: Type[Builder],
) -> None:
    """Reject timestamps that include timezone markers."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="timezone"):
        visitor.visit(astx.LiteralDateTime("2025-10-30T12:34:56Z"))


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_invalid_month(builder_class: Type[Builder]) -> None:
    """Reject months outside 1-12 range."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="calendar date"):
        visitor.visit(astx.LiteralDateTime("2025-13-01T00:00:00"))


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_invalid_day(builder_class: Type[Builder]) -> None:
    """Reject impossible calendar dates (e.g., December 32)."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="calendar date"):
        visitor.visit(astx.LiteralDateTime("2025-12-32T00:00:00"))


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_invalid_hour(builder_class: Type[Builder]) -> None:
    """Reject hours outside 0-23 range."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="hour out of range"):
        visitor.visit(astx.LiteralDateTime("2025-10-30T24:00:00"))


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_invalid_minute(builder_class: Type[Builder]) -> None:
    """Reject minutes outside 0-59 range."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="minute out of range"):
        visitor.visit(astx.LiteralDateTime("2025-10-30T12:60:00"))


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_invalid_second(builder_class: Type[Builder]) -> None:
    """Reject seconds outside 0-59 range."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="second out of range"):
        visitor.visit(astx.LiteralDateTime("2025-10-30T12:34:60"))

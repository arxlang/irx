"""
title: Tests for LiteralTime lowering using project conventions.
"""

from __future__ import annotations

import re

from typing import cast

import pytest

from irx import astx
from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder
from irx.builders.llvmliteir import Visitor as LLVMVisitor
from llvmlite import ir

HAS_LITERAL_TIME = hasattr(astx, "LiteralTime")


def _time_values(const: ir.Constant) -> list[int]:
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
    not HAS_LITERAL_TIME, reason="astx.LiteralTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_time_hh_mm_ss(
    builder_class: type[Builder],
) -> None:
    """
    title: HH:MM:SS time lowers to constant struct.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTime("12:34:56"))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert const.type == visitor._llvm.TIME_TYPE
    vals = _time_values(const)
    assert vals == [12, 34, 56]
    assert not visitor.result_stack


@pytest.mark.skipif(
    not HAS_LITERAL_TIME, reason="astx.LiteralTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_time_hh_mm(
    builder_class: type[Builder],
) -> None:
    """
    title: HH:MM time defaults second to zero.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTime("08:15"))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert const.type == visitor._llvm.TIME_TYPE
    vals = _time_values(const)
    assert vals == [8, 15, 0]
    assert not visitor.result_stack


@pytest.mark.skipif(
    not HAS_LITERAL_TIME, reason="astx.LiteralTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_time_midnight(
    builder_class: type[Builder],
) -> None:
    """
    title: Midnight 00:00:00 lowers correctly.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTime("00:00:00"))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert const.type == visitor._llvm.TIME_TYPE
    vals = _time_values(const)
    assert vals == [0, 0, 0]
    assert not visitor.result_stack


@pytest.mark.skipif(
    not HAS_LITERAL_TIME, reason="astx.LiteralTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_time_hour_out_of_range(
    builder_class: type[Builder],
) -> None:
    """
    title: Reject hour outside 0-23 range.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="hour out of range"):
        visitor.visit(astx.LiteralTime("25:00:00"))


@pytest.mark.skipif(
    not HAS_LITERAL_TIME, reason="astx.LiteralTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_time_fractional_rejected(
    builder_class: type[Builder],
) -> None:
    """
    title: Reject fractional seconds.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()
    with pytest.raises(Exception, match="fractional seconds"):
        visitor.visit(astx.LiteralTime("12:34:56.789"))

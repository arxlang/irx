"""
title: Tests for LiteralSet lowering using project conventions.
"""

from __future__ import annotations

import re

from typing import cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

EXPECTED_SET_LENGTH = 2
EXPECTED_PROMOTED_WIDTH = 32


def _array_i32_values(const: ir.Constant) -> list[int]:
    """
    title: Extract i32-like values from array constant via regex (suite style).
    parameters:
      const:
        type: ir.Constant
    returns:
      type: list[int]
    """
    return [int(v) for v in re.findall(r"i\d+\s+(-?\d+)", str(const))]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_empty(builder_class: type[Builder]) -> None:
    """
    title: Empty set lowers to constant [0 x i32].
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralSet(elements=set()))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 0
    assert const.type.element == ir.IntType(32)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_homogeneous_ints(builder_class: type[Builder]) -> None:
    """
    title: Homogeneous integer constants lower to constant array [N x i32].
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralSet(
            elements={
                astx.LiteralInt32(1),
                astx.LiteralInt32(2),
                astx.LiteralInt32(3),
            }
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 3  # noqa: PLR2004
    assert const.type.element == ir.IntType(32)
    # Values should be deterministically sorted
    vals = _array_i32_values(const)
    assert vals == [1, 2, 3]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_mixed_int_widths(builder_class: type[Builder]) -> None:
    """
    title: Mixed-width integer constants lower correctly.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralSet(elements={astx.LiteralInt16(1), astx.LiteralInt32(2)})
    )

    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == EXPECTED_SET_LENGTH

    # Check promoted type is i32 (widest type)
    assert isinstance(const.type.element, ir.IntType)
    assert const.type.element.width == EXPECTED_PROMOTED_WIDTH

    # Check values are correct after promotion
    vals = _array_i32_values(const)
    assert vals == [1, 2]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_non_integer_unsupported(
    builder_class: type[Builder],
) -> None:
    """
    title: Non-integer homogeneous sets are not yet supported.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError, match="integer constants"):
        visitor.visit(
            astx.LiteralSet(
                elements={astx.LiteralFloat32(1.0), astx.LiteralFloat32(2.0)}
            )
        )


def _make_set(*vals: int) -> astx.LiteralSet:
    return astx.LiteralSet(elements={astx.LiteralInt32(v) for v in vals})


def _set_values(const: ir.Value) -> list[int]:
    return _array_i32_values(const)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_set_union(builder_class: type[Builder]) -> None:
    """
    title: BinaryOp | on two LiteralSets produces their union.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.BinaryOp(op_code="|", lhs=_make_set(1, 2), rhs=_make_set(2, 3))
    visitor.visit(expr)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)
    assert isinstance(result.type, ir.ArrayType)
    assert _set_values(result) == [1, 2, 3]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_set_intersection(builder_class: type[Builder]) -> None:
    """
    title: BinaryOp & on two LiteralSets produces their intersection.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.BinaryOp(
        op_code="&", lhs=_make_set(1, 2, 3), rhs=_make_set(2, 3, 4)
    )
    visitor.visit(expr)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)
    assert _set_values(result) == [2, 3]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_set_difference(builder_class: type[Builder]) -> None:
    """
    title: BinaryOp - on two LiteralSets produces their difference.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.BinaryOp(
        op_code="-", lhs=_make_set(1, 2, 3), rhs=_make_set(2, 3)
    )
    visitor.visit(expr)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)
    assert _set_values(result) == [1]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_set_symmetric_difference(builder_class: type[Builder]) -> None:
    """
    title: BinaryOp ^ on two LiteralSets produces their symmetric difference.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.BinaryOp(
        op_code="^", lhs=_make_set(1, 2), rhs=_make_set(2, 3)
    )
    visitor.visit(expr)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)
    assert _set_values(result) == [1, 3]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_set_disjoint_intersection_is_empty(
    builder_class: type[Builder],
) -> None:
    """
    title: Intersection of disjoint sets is an empty constant array.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.BinaryOp(
        op_code="&", lhs=_make_set(1, 2), rhs=_make_set(3, 4)
    )
    visitor.visit(expr)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)
    assert isinstance(result.type, ir.ArrayType)
    assert result.type.count == 0

"""
title: Tests for LiteralList lowering using project conventions.
"""

from __future__ import annotations

import re

from typing import cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

HAS_LITERAL_LIST = hasattr(astx, "LiteralList")


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


def _make_visitor_in_function() -> LLVMLiteIRVisitor:
    """
    title: Return a visitor whose ir_builder is inside a live basic block.
    summary: >-
      _coerce_to and the alloca path both need an active insertion point.
      We create a dummy function and position the builder at its entry block
      so instructions can be emitted without an AssertionError from llvmlite.
    returns:
      type: LLVMLiteIRVisitor
    """
    builder = LLVMLiteIR()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    # Create a dummy void() function to give the builder a valid block
    fn_ty = ir.FunctionType(visitor._llvm.VOID_TYPE, [])
    fn = ir.Function(visitor._llvm.module, fn_ty, name="_test_dummy")
    bb = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(bb)

    return visitor


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_empty(builder_class: type[Builder]) -> None:
    """
    title: Empty list lowers to constant [0 x i32].
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralList(elements=[]))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 0


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_homogeneous_ints(builder_class: type[Builder]) -> None:
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
        astx.LiteralList(
            elements=[
                astx.LiteralInt32(1),
                astx.LiteralInt32(2),
                astx.LiteralInt32(3),
            ]
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 3  # noqa: PLR2004
    vals = _array_i32_values(const)
    assert vals == [1, 2, 3]


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_mixed_int_widths_widens(
    builder_class: Type[Builder],
) -> None:
    """
    title: Mixed-width integer list widens all elements to the widest type.
    summary: >-
      [i16(1), i32(2)] should produce a constant [2 x i32] where the i16
      has been sign-extended.  Requires a live IR builder block.
    parameters:
      builder_class:
        type: type[Builder]
    """
    visitor = _make_visitor_in_function()

    visitor.visit(
        astx.LiteralList(
            elements=[astx.LiteralInt16(1), astx.LiteralInt32(2)]
        )
    )
    const = visitor.result_stack.pop()

    # Must be a constant array of i32 with count 2
    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 2  # noqa: PLR2004
    assert const.type.element == ir.IntType(32)
    vals = _array_i32_values(const)
    assert vals == [1, 2]


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_homogeneous_floats(
    builder_class: Type[Builder],
) -> None:
    """
    title: Homogeneous float constants lower to constant array [N x float].
    parameters:
      builder_class:
        type: type[Builder]
    """
    visitor = _make_visitor_in_function()

    visitor.visit(
        astx.LiteralList(
            elements=[
                astx.LiteralFloat32(1.0),
                astx.LiteralFloat32(2.0),
                astx.LiteralFloat32(3.0),
            ]
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 3  # noqa: PLR2004
    assert isinstance(const.type.element, ir.FloatType)


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_mixed_int_and_float_promotes(
    builder_class: Type[Builder],
) -> None:
    """
    title: Mixed int+float list promotes the integer to float.
    summary: >-
      [i32(1), float(2.0)] should produce a constant [2 x float].
    parameters:
      builder_class:
        type: Type[Builder]
    """
    visitor = _make_visitor_in_function()

    visitor.visit(
        astx.LiteralList(
            elements=[astx.LiteralInt32(1), astx.LiteralFloat32(2.0)]
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 2  # noqa: PLR2004
    assert isinstance(const.type.element, ir.FloatType)


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_incompatible_types_raises(
    builder_class: Type[Builder],
) -> None:
    """
    title: Incompatible types (e.g. pointer + int) raise TypeError.
    summary: >-
      Mixing a string (i8*) with an integer has no valid common type and
      must raise TypeError.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    visitor = _make_visitor_in_function()

    with pytest.raises(TypeError):
        visitor.visit(
            astx.LiteralList(
                elements=[
                    astx.LiteralUTF8String("hello"),
                    astx.LiteralInt32(1),
                ]
            )
        )


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_nested_unsupported(
    builder_class: type[Builder],
) -> None:
    """
    title: Nested lists (list containing lists) are not yet supported.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    # Nested lists fail at AST construction time (before lowering)
    with pytest.raises(TypeError, match=r"missing.*argument.*element_types"):
        visitor.visit(
            astx.LiteralList(elements=[astx.LiteralList(elements=[])])
        )
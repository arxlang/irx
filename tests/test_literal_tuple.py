"""Tests for LiteralTuple lowering using project conventions."""

from __future__ import annotations

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir


def _assert_unpacked_literal_struct(alloca: ir.AllocaInstr) -> None:
    """Assert alloca points to an unpacked literal struct."""
    assert isinstance(alloca, ir.AllocaInstr)
    pointee = alloca.type.pointee
    assert isinstance(pointee, ir.LiteralStructType)
    assert not pointee.packed


def _assert_empty_stack(visitor: LLVMLiteIRVisitor) -> None:
    """Assert translator result stack is empty after evaluation."""
    assert len(visitor.result_stack) == 0


def _setup_function_context(visitor: LLVMLiteIRVisitor) -> None:
    """Create a dummy function so the visitor can emit alloca instructions."""
    fn_ty = ir.FunctionType(ir.VoidType(), [])
    fn = ir.Function(visitor._llvm.module, fn_ty, "test_tuple_fn")
    bb = fn.append_basic_block("entry")
    visitor._llvm.ir_builder.position_at_end(bb)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_empty(builder_class: Type[Builder]) -> None:
    """Empty tuple lowers to alloca of {} (empty literal struct)."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    _setup_function_context(visitor)

    visitor.visit(astx.LiteralTuple(elements=()))
    result = visitor.result_stack.pop()

    _assert_unpacked_literal_struct(result)
    assert len(result.type.pointee.elements) == 0
    _assert_empty_stack(visitor)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_homogeneous_ints(
    builder_class: Type[Builder],
) -> None:
    """Homogeneous integer constants lower to alloca of struct."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    _setup_function_context(visitor)

    visitor.visit(
        astx.LiteralTuple(
            elements=(
                astx.LiteralInt32(1),
                astx.LiteralInt32(2),
                astx.LiteralInt32(3),
            )
        )
    )
    result = visitor.result_stack.pop()

    _assert_unpacked_literal_struct(result)
    elem_count = 3
    pointee = result.type.pointee
    assert len(pointee.elements) == elem_count
    assert all(
        isinstance(t, ir.IntType) and t.width == 32  # noqa: PLR2004
        for t in pointee.elements
    )
    _assert_empty_stack(visitor)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_heterogeneous(
    builder_class: Type[Builder],
) -> None:
    """Heterogeneous tuple (int, float) lowers to alloca of struct."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    _setup_function_context(visitor)

    visitor.visit(
        astx.LiteralTuple(
            elements=(
                astx.LiteralInt32(42),
                astx.LiteralFloat32(2.5),
            )
        )
    )
    result = visitor.result_stack.pop()

    _assert_unpacked_literal_struct(result)
    elem_count = 2
    pointee = result.type.pointee
    assert len(pointee.elements) == elem_count
    assert isinstance(pointee.elements[0], ir.IntType)
    assert isinstance(pointee.elements[1], ir.FloatType)
    assert pointee.elements[1] == ir.FloatType()
    _assert_empty_stack(visitor)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_single_element(
    builder_class: Type[Builder],
) -> None:
    """Single-element tuple lowers to alloca of struct {i32}."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    _setup_function_context(visitor)

    visitor.visit(astx.LiteralTuple(elements=(astx.LiteralInt32(42),)))
    result = visitor.result_stack.pop()

    _assert_unpacked_literal_struct(result)
    pointee = result.type.pointee
    assert len(pointee.elements) == 1
    assert isinstance(pointee.elements[0], ir.IntType)
    _assert_empty_stack(visitor)

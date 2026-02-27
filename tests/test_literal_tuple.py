"""Tests for LiteralTuple lowering using project conventions."""

from __future__ import annotations

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

HAS_LITERAL_TUPLE = hasattr(astx, "LiteralTuple")


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_empty(builder_class: Type[Builder]) -> None:
    """Empty tuple lowers to constant {} (empty literal struct)."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTuple(elements=()))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == 0


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_homogeneous_ints(
    builder_class: Type[Builder],
) -> None:
    """Homogeneous integer constants lower to constant struct."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralTuple(
            elements=(
                astx.LiteralInt32(1),
                astx.LiteralInt32(2),
                astx.LiteralInt32(3),
            )
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    elem_count = 3
    assert len(const.type.elements) == elem_count
    assert all(
        isinstance(t, ir.IntType) and t.width == 32  # noqa: PLR2004
        for t in const.type.elements
    )


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_heterogeneous(
    builder_class: Type[Builder],
) -> None:
    """Heterogeneous tuple (int, float) lowers to constant struct."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralTuple(
            elements=(
                astx.LiteralInt32(42),
                astx.LiteralFloat32(2.5),
            )
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    elem_count = 2
    assert len(const.type.elements) == elem_count
    assert isinstance(const.type.elements[0], ir.IntType)
    assert isinstance(const.type.elements[1], ir.FloatType)


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_single_element(
    builder_class: Type[Builder],
) -> None:
    """Single-element tuple lowers to constant struct {i32}."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTuple(elements=(astx.LiteralInt32(42),)))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == 1
    assert isinstance(const.type.elements[0], ir.IntType)

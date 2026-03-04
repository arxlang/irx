"""
title: Tests for LiteralTuple lowering using project conventions.
"""

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
    """
    title: Empty tuple lowers to constant empty literal struct {}.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTuple(elements=()))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert not const.type.elements


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_homogeneous_constants(
    builder_class: Type[Builder],
) -> None:
    """
    title: Homogeneous tuple constants lower to constant struct.
    parameters:
      builder_class:
        type: Type[Builder]
    """
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
    assert len(const.type.elements) == 3  # noqa: PLR2004


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_heterogeneous_constants(
    builder_class: Type[Builder],
) -> None:
    """
    title: Heterogeneous tuple constants lower to constant struct.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralTuple(
            elements=(
                astx.LiteralInt32(1),
                astx.LiteralFloat32(2.5),
            )
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == 2  # noqa: PLR2004


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_single_element(builder_class: Type[Builder]) -> None:
    """
    title: Single element tuple lowers to single field struct.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTuple(elements=(astx.LiteralBoolean(True),)))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == 1

"""
title: LiteralTuple lowering tests
"""

from __future__ import annotations

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

EXPECTED_TUPLE_LENGTH = 3

HAS_LITERAL_TUPLE = hasattr(astx, "LiteralTuple")


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_empty(builder_class: Type[Builder]) -> None:
    """
    title: Empty LiteralTuple lowering
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
    assert len(const.type.elements) == 0


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_homogeneous_ints(
    builder_class: Type[Builder],
) -> None:
    """
    title: Homogeneous LiteralTuple lowering
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
    assert len(const.type.elements) == EXPECTED_TUPLE_LENGTH
    assert all(isinstance(t, ir.IntType) for t in const.type.elements)


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_heterogeneous_unsupported(
    builder_class: Type[Builder],
) -> None:
    """
    title: Heterogeneous LiteralTuple rejection
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError, match="homogeneous"):
        visitor.visit(
            astx.LiteralTuple(
                elements=(
                    astx.LiteralInt32(1),
                    astx.LiteralFloat32(2.0),
                )
            )
        )

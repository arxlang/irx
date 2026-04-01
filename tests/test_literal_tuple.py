"""
title: LiteralTuple lowering tests
"""

from __future__ import annotations

import re

from typing import cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder
from irx.builders.llvmliteir import Visitor as LLVMVisitor
from llvmlite import ir

EXPECTED_TUPLE_LENGTH = 3


def _struct_int_values(const: ir.Constant) -> list[int]:
    """
    title: Extract integer values from struct constant
    parameters:
      const:
        type: ir.Constant
    returns:
      type: list[int]
    """
    return [int(v) for v in re.findall(r"i\d+\s+(-?\d+)", str(const))]


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_tuple_empty(builder_class: type[Builder]) -> None:
    """
    title: Empty LiteralTuple lowering
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTuple(elements=()))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == 0


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_tuple_homogeneous_ints(
    builder_class: type[Builder],
) -> None:
    """
    title: Homogeneous LiteralTuple lowering
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
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

    vals = _struct_int_values(const)
    assert vals == [1, 2, 3]


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_tuple_heterogeneous_unsupported(
    builder_class: type[Builder],
) -> None:
    """
    title: Heterogeneous LiteralTuple rejection
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
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

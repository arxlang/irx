"""
title: Tests for LiteralSet lowering
"""

from __future__ import annotations

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

EXPECTED_SET_LENGTH = 3


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_empty(builder_class: Type[Builder]) -> None:
    """
    title: Test empty set lowering
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralSet(elements=set()))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 0


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_homogeneous_int_constants(
    builder_class: Type[Builder],
) -> None:
    """
    title: Test homogeneous integer set lowering
    parameters:
      builder_class:
        type: Type[Builder]
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
    assert const.type.count == EXPECTED_SET_LENGTH

    # element type should be integer
    assert isinstance(const.type.element, ir.IntType)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_mixed_int_widths_unsupported(
    builder_class: Type[Builder],
) -> None:
    """
    title: Test mixed integer width set unsupported
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError, match="homogeneous"):
        visitor.visit(
            astx.LiteralSet(
                elements={
                    astx.LiteralInt16(1),
                    astx.LiteralInt32(2),
                }
            )
        )


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_non_integer_unsupported(
    builder_class: Type[Builder],
) -> None:
    """
    title: Test non integer set unsupported
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError, match="homogeneous"):
        visitor.visit(
            astx.LiteralSet(
                elements={
                    astx.LiteralFloat32(1.0),
                    astx.LiteralFloat32(2.0),
                }
            )
        )

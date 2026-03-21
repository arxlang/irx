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
        astx.LiteralSet(
            elements={astx.LiteralInt16(1), astx.LiteralInt32(2)}
        )
    )

    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == EXPECTED_SET_LENGTH

    assert isinstance(const.type.element, ir.IntType)
    assert const.type.element.width == EXPECTED_PROMOTED_WIDTH

    vals = _array_i32_values(const)
    assert vals == [1, 2]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_float_constants(builder_class: type[Builder]) -> None:
    """
    title: Homogeneous float constants lower to constant array [N x float].
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
                astx.LiteralFloat32(1.0),
                astx.LiteralFloat32(2.0),
            }
        )
    )

    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == EXPECTED_SET_LENGTH
    assert isinstance(const.type.element, ir.FloatType)

    # Format-independent float validation
    ir_str = str(const)
    assert "float" in ir_str

    vals = re.findall(r"float\s+([^\],]+)", ir_str)
    assert len(vals) == EXPECTED_SET_LENGTH


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_heterogeneous_unsupported(
    builder_class: type[Builder],
) -> None:
    """
    title: Heterogeneous sets are not supported.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError):
        visitor.visit(
            astx.LiteralSet(
                elements={
                    astx.LiteralInt32(1),
                    astx.LiteralFloat32(2.0),
                }
            )
        )


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_runtime_lowering(builder_class: type[Builder]) -> None:
    """
    title: Runtime lowering for mixed-width integer sets.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)

    module = ir.Module()
    func_ty = ir.FunctionType(ir.VoidType(), [])
    func = ir.Function(module, func_ty, name="test")

    block = func.append_basic_block(name="entry")
    ir_builder = ir.IRBuilder(block)

    visitor._llvm.ir_builder = ir_builder
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralSet(
            elements={
                astx.LiteralInt16(1),
                astx.LiteralInt32(2),
            }
        )
    )

    result = visitor.result_stack.pop()

    assert isinstance(result, ir.instructions.AllocaInstr)

    assert isinstance(result.type.pointee, ir.ArrayType)
    assert result.type.pointee.count == EXPECTED_SET_LENGTH

    # Validate emitted IR instructions
    ir_str = str(visitor._llvm.ir_builder.function)

    assert "sext i16 1" in ir_str
    assert "store i32 2" in ir_str
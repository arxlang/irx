"""
title: LiteralDict lowering tests
"""

from __future__ import annotations

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

EXPECTED_DICT_LENGTH = 2
EXPECTED_STRUCT_FIELDS = 2


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_empty(builder_class: Type[Builder]) -> None:
    """
    title: Empty LiteralDict lowering
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralDict(elements={}))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 0


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_homogeneous_int_constants(
    builder_class: Type[Builder],
) -> None:
    """
    title: Homogeneous LiteralDict lowering
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralDict(
            elements={
                astx.LiteralInt32(1): astx.LiteralInt32(10),
                astx.LiteralInt32(2): astx.LiteralInt32(20),
            }
        )
    )

    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == EXPECTED_DICT_LENGTH

    # Check element struct type
    assert isinstance(const.type.element, ir.LiteralStructType)
    struct_ty = const.type.element
    assert len(struct_ty.elements) == EXPECTED_STRUCT_FIELDS
    assert all(isinstance(t, ir.IntType) for t in struct_ty.elements)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_heterogeneous_constants_unsupported(
    builder_class: Type[Builder],
) -> None:
    """
    title: Heterogeneous LiteralDict rejection
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError, match="heterogeneous"):
        visitor.visit(
            astx.LiteralDict(
                elements={
                    astx.LiteralInt32(1): astx.LiteralInt32(10),
                    astx.LiteralInt32(2): astx.LiteralFloat32(3.5),
                }
            )
        )


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_runtime_lowering(builder_class: Type[Builder]) -> None:
    """
    title: Runtime LiteralDict lowering (non-constant path)
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    # Simulate runtime values by manually constructing LLVM values
    key = ir.Constant(visitor._llvm.INT32_TYPE, 1)
    val_alloca = visitor._llvm.ir_builder.alloca(visitor._llvm.INT32_TYPE)

    visitor.result_stack.append(key)
    visitor.result_stack.append(val_alloca)

    visitor.visit(
        astx.LiteralDict(
            elements={
                astx.LiteralInt32(1): astx.LiteralInt32(2),
            }
        )
    )

    result = visitor.result_stack.pop()

    # When no function context exists, runtime lowering falls back to constant
    assert isinstance(result, ir.Constant)
    assert isinstance(result.type, ir.ArrayType)

"""
title: LiteralDict lowering tests
"""

from __future__ import annotations

from typing import cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder
from irx.builders.llvmliteir import Visitor as LLVMVisitor
from llvmlite import ir

EXPECTED_DICT_LENGTH = 2
EXPECTED_STRUCT_FIELDS = 2


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_dict_empty(builder_class: type[Builder]) -> None:
    """
    title: Empty LiteralDict lowering
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralDict(elements={}))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 0


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_dict_homogeneous_int_constants(
    builder_class: type[Builder],
) -> None:
    """
    title: Homogeneous LiteralDict lowering
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
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


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_literal_dict_heterogeneous_constants_unsupported(
    builder_class: type[Builder],
) -> None:
    """
    title: Heterogeneous LiteralDict rejection
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
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

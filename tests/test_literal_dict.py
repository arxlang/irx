"""Tests for LiteralDict lowering using project conventions."""

from __future__ import annotations

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

HAS_LITERAL_DICT = hasattr(astx, "LiteralDict")


@pytest.mark.skipif(
    not HAS_LITERAL_DICT, reason="astx.LiteralDict not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_empty(builder_class: Type[Builder]) -> None:
    """Empty dict lowers to constant [0 x {i32, i32}]."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralDict(elements={}))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 0


@pytest.mark.skipif(
    not HAS_LITERAL_DICT, reason="astx.LiteralDict not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_homogeneous_int_constants(
    builder_class: Type[Builder],
) -> None:
    """Homogeneous integer constant dict lowers to constant array."""
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
    assert const.type.count == 2

    # Check element struct type
    assert isinstance(const.type.element, ir.LiteralStructType)
    struct_ty = const.type.element
    assert len(struct_ty.elements) == 2
    assert all(isinstance(t, ir.IntType) for t in struct_ty.elements)


@pytest.mark.skipif(
    not HAS_LITERAL_DICT, reason="astx.LiteralDict not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_heterogeneous_constants_unsupported(
    builder_class: Type[Builder],
) -> None:
    """Heterogeneous constant key/value types are not yet supported."""
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


@pytest.mark.skipif(
    not HAS_LITERAL_DICT, reason="astx.LiteralDict not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_non_constant_unsupported(
    builder_class: Type[Builder],
) -> None:
    """Non-constant dict elements are not yet supported."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    # Use a variable to simulate non-constant
    var = astx.Variable(name="x")

    with pytest.raises(TypeError, match="only empty or all-constant"):
        visitor.visit(
            astx.LiteralDict(
                elements={
                    var: astx.LiteralInt32(10),
                }
            )
        )

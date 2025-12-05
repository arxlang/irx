"""Tests for LiteralList lowering using project conventions."""

from __future__ import annotations

import re

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

HAS_LITERAL_LIST = hasattr(astx, "LiteralList")


def _array_i32_values(const: ir.Constant) -> list[int]:
    """Extract i32-like values from array constant via regex (suite style)."""
    return [int(v) for v in re.findall(r"i\d+\s+(-?\d+)", str(const))]


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_empty(builder_class: Type[Builder]) -> None:
    """Empty list lowers to constant [0 x i32]."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralList(elements=[]))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    # Expect [0 x i32]
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 0


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_homogeneous_ints(builder_class: Type[Builder]) -> None:
    """Homogeneous integer constants lower to constant array [N x i32]."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralList(
            elements=[
                astx.LiteralInt32(1),
                astx.LiteralInt32(2),
                astx.LiteralInt32(3),
            ]
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 3  # noqa: PLR2004
    vals = _array_i32_values(const)
    assert vals == [1, 2, 3]


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_mixed_int_widths_unsupported(
    builder_class: Type[Builder],
) -> None:
    """Mixed-width integer lists are not yet supported."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError, match="only empty or homogeneous integer"):
        visitor.visit(
            astx.LiteralList(
                elements=[astx.LiteralInt16(1), astx.LiteralInt32(2)]
            )
        )


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_non_integer_unsupported(
    builder_class: Type[Builder],
) -> None:
    """Non-integer homogeneous lists are not yet supported."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError, match="only empty or homogeneous integer"):
        visitor.visit(
            astx.LiteralList(
                elements=[astx.LiteralFloat32(1.0), astx.LiteralFloat32(2.0)]
            )
        )


@pytest.mark.skipif(
    not HAS_LITERAL_LIST, reason="astx.LiteralList not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_list_nested_unsupported(
    builder_class: Type[Builder],
) -> None:
    """Nested lists (list containing lists) are not yet supported."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    # Nested lists fail at AST construction time (before lowering)
    with pytest.raises(TypeError, match=r"missing.*argument.*element_types"):
        visitor.visit(
            astx.LiteralList(elements=[astx.LiteralList(elements=[])])
        )

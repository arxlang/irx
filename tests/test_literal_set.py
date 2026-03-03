"""Tests for LiteralSet lowering using project conventions."""

from __future__ import annotations

import re

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

HAS_LITERAL_SET = hasattr(astx, "LiteralSet")


def _array_i32_values(const: ir.Constant) -> list[int]:
    """Extract i32-like values from array constant via regex (suite style)."""
    return [int(v) for v in re.findall(r"i\d+\s+(-?\d+)", str(const))]


@pytest.mark.skipif(
    not HAS_LITERAL_SET, reason="astx.LiteralSet not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_empty(builder_class: Type[Builder]) -> None:
    """Empty set lowers to constant [0 x i32]."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralSet(elements=set()))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    # Expect [0 x i32]
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 0


@pytest.mark.skipif(
    not HAS_LITERAL_SET, reason="astx.LiteralSet not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_homogeneous_ints(builder_class: Type[Builder]) -> None:
    """Homogeneous integer constants lower to constant array [N x i32]."""
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
    # Values should be deterministically sorted
    vals = _array_i32_values(const)
    assert sorted(vals) == [1, 2, 3]


@pytest.mark.skipif(
    not HAS_LITERAL_SET, reason="astx.LiteralSet not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_mixed_int_widths_unsupported(
    builder_class: Type[Builder],
) -> None:
    """Mixed-width integer sets are not yet supported."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError, match="only empty or homogeneous integer"):
        visitor.visit(
            astx.LiteralSet(
                elements={astx.LiteralInt16(1), astx.LiteralInt32(2)}
            )
        )


@pytest.mark.skipif(
    not HAS_LITERAL_SET, reason="astx.LiteralSet not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_set_non_integer_unsupported(
    builder_class: Type[Builder],
) -> None:
    """Non-integer homogeneous sets are not yet supported."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError, match="only empty or homogeneous integer"):
        visitor.visit(
            astx.LiteralSet(
                elements={astx.LiteralFloat32(1.0), astx.LiteralFloat32(2.0)}
            )
        )

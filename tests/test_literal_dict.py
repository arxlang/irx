"""Tests for LiteralDict lowering using project conventions."""

from __future__ import annotations

import re

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir


def _struct_array_values(
    const: ir.Constant,
) -> list[tuple[int, int]]:
    """Extract {iN k, iN v} pairs from a constant struct array."""
    pairs = re.findall(
        r"\{\s*i\d+\s+(-?\d+),\s*i\d+\s+(-?\d+)\s*\}",
        str(const),
    )
    return [(int(k), int(v)) for k, v in pairs]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_empty(
    builder_class: Type[Builder],
) -> None:
    """Empty dict lowers to constant [0 x {i32, i32}]."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralDict(elements={}))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 0


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_homogeneous_ints(
    builder_class: Type[Builder],
) -> None:
    """Homogeneous integer key/value pairs produce a constant array."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralDict(
            elements={
                astx.LiteralInt32(1): astx.LiteralInt32(10),
                astx.LiteralInt32(2): astx.LiteralInt32(20),
                astx.LiteralInt32(3): astx.LiteralInt32(30),
            }
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == 3  # noqa: PLR2004
    pairs = _struct_array_values(const)
    assert pairs == [(1, 10), (2, 20), (3, 30)]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_mixed_int_widths(
    builder_class: Type[Builder],
) -> None:
    """Mixed-width integer dict uses alloca + sext (needs function)."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    # Build a function wrapper so the IR builder has a block
    fn_type = ir.FunctionType(ir.VoidType(), [])
    fn = ir.Function(visitor._llvm.module, fn_type, "test_fn")
    bb = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(bb)

    visitor.visit(
        astx.LiteralDict(
            elements={
                astx.LiteralInt8(1): astx.LiteralInt32(100),
                astx.LiteralInt32(2): astx.LiteralInt8(50),
            }
        )
    )
    result = visitor.result_stack.pop()

    # Result should be an alloca pointer
    assert isinstance(result, ir.instructions.AllocaInstr)
    inner = result.type.pointee
    assert isinstance(inner, ir.ArrayType)
    assert inner.count == 2  # noqa: PLR2004


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_dict_unsupported_types(
    builder_class: Type[Builder],
) -> None:
    """Non-integer entry types raise TypeError."""
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(
        TypeError,
        match="only empty, homogeneous constant, or integer",
    ):
        visitor.visit(
            astx.LiteralDict(
                elements={
                    astx.LiteralInt32(1): astx.LiteralFloat32(1.5),
                }
            )
        )

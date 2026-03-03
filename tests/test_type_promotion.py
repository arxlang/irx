"""
title: Tests for type promotion functionality.
"""

import pytest

from irx.builders.llvmliteir import LLVMLiteIRVisitor
from llvmlite import ir


@pytest.fixture
def visitor() -> LLVMLiteIRVisitor:
    """
    title: Create a LLVMLiteIRVisitor with a dummy module, function, and block.
    summary: This ensures a valid IRBuilder context.
    returns:
      type: LLVMLiteIRVisitor
    """
    v = LLVMLiteIRVisitor()
    dummy_module = ir.Module("dummy")
    dummy_func_ty = ir.FunctionType(ir.VoidType(), [])
    dummy_func = ir.Function(dummy_module, dummy_func_ty, name="dummy")
    entry_block = dummy_func.append_basic_block("entry")
    v._llvm.ir_builder = ir.IRBuilder(entry_block)
    return v


def test_same_type(visitor: LLVMLiteIRVisitor) -> None:
    """
    title: Test promotion in case of same type (no promotion).
    parameters:
      visitor:
        type: LLVMLiteIRVisitor
    """
    a = ir.Constant(ir.IntType(32), 5)
    b = ir.Constant(ir.IntType(32), 10)
    a_promoted, b_promoted = visitor.promote_operands(a, b)
    assert a_promoted.type == b_promoted.type
    assert str(a_promoted.type) == "i32"


def test_integer_promotion(visitor: LLVMLiteIRVisitor) -> None:
    """
    title: Test integer type promotion.
    parameters:
      visitor:
        type: LLVMLiteIRVisitor
    """
    a = ir.Constant(ir.IntType(16), 5)
    b = ir.Constant(ir.IntType(32), 10)
    a_promoted, b_promoted = visitor.promote_operands(a, b)
    assert a_promoted.type == b_promoted.type
    assert str(a_promoted.type) == "i32"


def test_float_promotion(visitor: LLVMLiteIRVisitor) -> None:
    """
    title: Test floating-point type promotion.
    parameters:
      visitor:
        type: LLVMLiteIRVisitor
    """
    a = ir.Constant(ir.FloatType(), 3.14)
    b = ir.Constant(ir.DoubleType(), 3.14)
    a_promoted, b_promoted = visitor.promote_operands(a, b)
    assert a_promoted.type == b_promoted.type
    assert str(a_promoted.type) == "double"


def test_same_type_returns_original_operands(
    visitor: LLVMLiteIRVisitor,
) -> None:
    """
    title: Test returning original operands in case of same type.
    parameters:
      visitor:
        type: LLVMLiteIRVisitor
    """
    a = ir.Constant(ir.IntType(8), 2)
    b = ir.Constant(ir.IntType(8), 3)
    a_promoted, b_promoted = visitor.promote_operands(a, b)
    assert a_promoted is a
    assert b_promoted is b

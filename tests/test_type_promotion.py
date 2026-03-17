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
    a_promoted, b_promoted = visitor._unify_numeric_operands(a, b)
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
    a_promoted, b_promoted = visitor._unify_numeric_operands(a, b)
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
    a_promoted, b_promoted = visitor._unify_numeric_operands(a, b)
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
    a_promoted, b_promoted = visitor._unify_numeric_operands(a, b)
    assert a_promoted is a
    assert b_promoted is b


def test_integer_promotion_extends_rhs(visitor: LLVMLiteIRVisitor) -> None:
    """
    title: Integer promotion should extend rhs when lhs is wider.
    parameters:
      visitor:
        type: LLVMLiteIRVisitor
    """
    lhs = ir.Constant(ir.IntType(32), 10)
    rhs = ir.Constant(ir.IntType(16), 3)
    lhs_promoted, rhs_promoted = visitor._unify_numeric_operands(lhs, rhs)

    assert lhs_promoted.type == rhs_promoted.type
    assert str(lhs_promoted.type) == "i32"
    assert getattr(rhs_promoted, "opname", "") == "sext"


def test_float_promotion_extends_rhs(visitor: LLVMLiteIRVisitor) -> None:
    """
    title: Float promotion should extend rhs when lhs has wider fp type.
    parameters:
      visitor:
        type: LLVMLiteIRVisitor
    """
    lhs = ir.Constant(ir.DoubleType(), 3.14)
    rhs = ir.Constant(ir.FloatType(), 2.71)
    lhs_promoted, rhs_promoted = visitor._unify_numeric_operands(lhs, rhs)

    assert lhs_promoted.type == rhs_promoted.type
    assert str(lhs_promoted.type) == "double"
    assert getattr(rhs_promoted, "opname", "") == "fpext"


def test_int_to_float_promotion_lhs(visitor: LLVMLiteIRVisitor) -> None:
    """
    title: Integer lhs should be converted to fp when rhs is floating-point.
    parameters:
      visitor:
        type: LLVMLiteIRVisitor
    """
    lhs = ir.Constant(ir.IntType(32), 2)
    rhs = ir.Constant(ir.FloatType(), 1.5)
    lhs_promoted, rhs_promoted = visitor._unify_numeric_operands(lhs, rhs)

    assert lhs_promoted.type == rhs_promoted.type
    assert str(lhs_promoted.type) == "float"
    assert getattr(lhs_promoted, "opname", "") == "sitofp"


def test_int_to_float_promotion_rhs(visitor: LLVMLiteIRVisitor) -> None:
    """
    title: Integer rhs should be converted to fp when lhs is floating-point.
    parameters:
      visitor:
        type: LLVMLiteIRVisitor
    """
    lhs = ir.Constant(ir.FloatType(), 1.5)
    rhs = ir.Constant(ir.IntType(32), 2)
    lhs_promoted, rhs_promoted = visitor._unify_numeric_operands(lhs, rhs)

    assert lhs_promoted.type == rhs_promoted.type
    assert str(lhs_promoted.type) == "float"
    assert getattr(rhs_promoted, "opname", "") == "sitofp"

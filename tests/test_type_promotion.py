"""Tests for type promotion functionality."""

import pytest

from irx.builders.llvmliteir import LLVMLiteIRVisitor
from llvmlite import ir


@pytest.fixture
def visitor() -> LLVMLiteIRVisitor:
    """Create a LLVMLiteIRVisitor with a dummy module, function, and block.

    This ensures a valid IRBuilder context.
    """
    v = LLVMLiteIRVisitor()
    dummy_module = ir.Module("dummy")
    dummy_func_ty = ir.FunctionType(ir.VoidType(), [])
    dummy_func = ir.Function(dummy_module, dummy_func_ty, name="dummy")
    entry_block = dummy_func.append_basic_block("entry")
    v._llvm.ir_builder = ir.IRBuilder(entry_block)
    return v


def test_same_type(visitor: LLVMLiteIRVisitor) -> None:
    """Test promotion in case of same type (no promotion)."""
    a = ir.Constant(ir.IntType(32), 5)
    b = ir.Constant(ir.IntType(32), 10)
    a_promoted, b_promoted = visitor.promote_operands(a, b)
    assert a_promoted.type == b_promoted.type
    assert str(a_promoted.type) == "i32"


def test_integer_promotion(visitor: LLVMLiteIRVisitor) -> None:
    """Test integer type promotion."""
    a = ir.Constant(ir.IntType(16), 5)
    b = ir.Constant(ir.IntType(32), 10)
    a_promoted, b_promoted = visitor.promote_operands(a, b)
    assert a_promoted.type == b_promoted.type
    assert str(a_promoted.type) == "i32"


def test_float_promotion(visitor: LLVMLiteIRVisitor) -> None:
    """Test floating-point type promotion."""
    a = ir.Constant(ir.FloatType(), 3.14)
    b = ir.Constant(ir.DoubleType(), 3.14)
    a_promoted, b_promoted = visitor.promote_operands(a, b)
    assert a_promoted.type == b_promoted.type
    assert str(a_promoted.type) == "double"


def test_same_type_returns_original_operands(
    visitor: LLVMLiteIRVisitor,
) -> None:
    """Test returning original operands in case of same type."""
    a = ir.Constant(ir.IntType(8), 2)
    b = ir.Constant(ir.IntType(8), 3)
    a_promoted, b_promoted = visitor.promote_operands(a, b)
    assert a_promoted is a
    assert b_promoted is b

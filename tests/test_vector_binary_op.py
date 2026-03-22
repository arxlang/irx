"""
title: Tests for vector BinaryOp arithmetic and scalar-vector promotion.
"""

from __future__ import annotations

import astx
import pytest

from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

VEC_LEN = 4


def _make_visitor_in_fn() -> LLVMLiteIRVisitor:
    """
    title: Create a visitor with an active function context.
    returns:
      type: LLVMLiteIRVisitor
    """
    builder = LLVMLiteIR()
    visitor: LLVMLiteIRVisitor = builder.translator
    float_ty = visitor._llvm.FLOAT_TYPE
    fn_ty = ir.FunctionType(float_ty, [])
    fn = ir.Function(visitor._llvm.module, fn_ty, name="vec_test")
    block = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(block)
    return visitor


def _visit_binop(
    visitor: LLVMLiteIRVisitor,
    lhs: ir.Value,
    rhs: ir.Value,
    op: str,
) -> ir.Value:
    """
    title: Push two values and visit a BinaryOp through the vector path.
    parameters:
      visitor:
        type: LLVMLiteIRVisitor
      lhs:
        type: ir.Value
      rhs:
        type: ir.Value
      op:
        type: str
    returns:
      type: ir.Value
    """
    visitor.result_stack.clear()
    visitor.result_stack.append(lhs)
    visitor.result_stack.append(rhs)

    mock_lhs = astx.LiteralInt32(0)
    mock_rhs = astx.LiteralInt32(0)
    node = astx.BinaryOp(op_code=op, lhs=mock_lhs, rhs=mock_rhs)

    original_visit = visitor.visit

    def _patched_visit(n: object) -> None:
        if n is mock_lhs or n is mock_rhs:
            return
        original_visit(n)

    visitor.visit = _patched_visit  # type: ignore[method-assign]
    try:
        visitor.visit(node)
    finally:
        visitor.visit = original_visit  # type: ignore[method-assign]

    return visitor.result_stack.pop()


# -- Integer vector arithmetic --


@pytest.mark.parametrize(
    "op,expected_opname",
    [
        ("+", "add"),
        ("-", "sub"),
        ("*", "mul"),
        ("/", "sdiv"),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_vec_int_arithmetic(
    builder_class: type, op: str, expected_opname: str
) -> None:
    """
    title: >-
      Integer vector BinaryOp emits the correct LLVM instruction and preserves
      the vector type.
    parameters:
      builder_class:
        type: type
      op:
        type: str
      expected_opname:
        type: str
    """
    visitor = _make_visitor_in_fn()
    i32 = visitor._llvm.INT32_TYPE
    vec_ty = ir.VectorType(i32, VEC_LEN)
    lhs = ir.Constant(vec_ty, [10, 20, 30, 40])
    rhs = ir.Constant(vec_ty, [2, 4, 5, 8])

    result = _visit_binop(visitor, lhs, rhs, op)

    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == VEC_LEN
    assert result.type.element == i32
    assert getattr(result, "opname", "") == expected_opname


# -- Float vector arithmetic --


@pytest.mark.parametrize(
    "op,expected_opname",
    [
        ("+", "fadd"),
        ("-", "fsub"),
        ("*", "fmul"),
        ("/", "fdiv"),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_vec_float_arithmetic(
    builder_class: type, op: str, expected_opname: str
) -> None:
    """
    title: >-
      Float vector BinaryOp emits the correct LLVM instruction and preserves
      the vector type.
    parameters:
      builder_class:
        type: type
      op:
        type: str
      expected_opname:
        type: str
    """
    visitor = _make_visitor_in_fn()
    f32 = visitor._llvm.FLOAT_TYPE
    vec_ty = ir.VectorType(f32, VEC_LEN)
    lhs = ir.Constant(vec_ty, [10.0, 20.0, 30.0, 40.0])
    rhs = ir.Constant(vec_ty, [2.0, 4.0, 5.0, 8.0])

    result = _visit_binop(visitor, lhs, rhs, op)

    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == VEC_LEN
    assert result.type.element == f32
    assert getattr(result, "opname", "") == expected_opname


# -- Vector error cases --


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_vec_unsupported_op_raises(builder_class: type) -> None:
    """
    title: Unsupported vector op code raises an exception.
    parameters:
      builder_class:
        type: type
    """
    visitor = _make_visitor_in_fn()
    i32 = visitor._llvm.INT32_TYPE
    vec_ty = ir.VectorType(i32, VEC_LEN)
    lhs = ir.Constant(vec_ty, [1, 2, 3, 4])
    rhs = ir.Constant(vec_ty, [5, 6, 7, 8])
    with pytest.raises(Exception, match="not implemented"):
        _visit_binop(visitor, lhs, rhs, "%")


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_vec_size_mismatch_raises(builder_class: type) -> None:
    """
    title: Mismatched vector lane counts raise an exception.
    parameters:
      builder_class:
        type: type
    """
    visitor = _make_visitor_in_fn()
    i32 = visitor._llvm.INT32_TYPE
    lhs = ir.Constant(ir.VectorType(i32, 4), [1, 2, 3, 4])
    rhs = ir.Constant(ir.VectorType(i32, 2), [5, 6])
    with pytest.raises(Exception, match="size mismatch"):
        _visit_binop(visitor, lhs, rhs, "+")


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_vec_element_type_mismatch_raises(builder_class: type) -> None:
    """
    title: Mismatched vector element types raise an exception.
    parameters:
      builder_class:
        type: type
    """
    visitor = _make_visitor_in_fn()
    i32 = visitor._llvm.INT32_TYPE
    f32 = visitor._llvm.FLOAT_TYPE
    lhs = ir.Constant(ir.VectorType(i32, VEC_LEN), [1, 2, 3, 4])
    rhs = ir.Constant(ir.VectorType(f32, VEC_LEN), [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(Exception, match="element type mismatch"):
        _visit_binop(visitor, lhs, rhs, "+")


# -- Scalar-vector promotion --


@pytest.mark.parametrize(
    "vec_side",
    ["lhs", "rhs"],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_scalar_vec_promote_matching_type(
    builder_class: type, vec_side: str
) -> None:
    """
    title: >-
      Matching-type scalar is splatted to the vector width, producing a valid
      fadd regardless of which side is the vector.
    parameters:
      builder_class:
        type: type
      vec_side:
        type: str
    """
    visitor = _make_visitor_in_fn()
    f32 = visitor._llvm.FLOAT_TYPE
    vec_ty = ir.VectorType(f32, VEC_LEN)
    vec_val = ir.Constant(vec_ty, [1.0, 2.0, 3.0, 4.0])
    scalar_val = ir.Constant(f32, 10.0)

    if vec_side == "lhs":
        result = _visit_binop(visitor, vec_val, scalar_val, "+")
    else:
        result = _visit_binop(visitor, scalar_val, vec_val, "+")

    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == VEC_LEN
    assert result.type.element == f32
    assert getattr(result, "opname", "") == "fadd"


@pytest.mark.parametrize(
    "vec_side,vec_elem,scalar_elem,expected_elem,conversion",
    [
        ("lhs", "float", "double", "float", "fptrunc"),
        ("lhs", "double", "float", "double", "fpext"),
        ("rhs", "float", "double", "float", "fptrunc"),
        ("rhs", "double", "float", "double", "fpext"),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_scalar_vec_promote_float_conversion(
    builder_class: type,
    vec_side: str,
    vec_elem: str,
    scalar_elem: str,
    expected_elem: str,
    conversion: str,
) -> None:
    """
    title: >-
      Float width mismatch between scalar and vector element triggers the
      correct truncation or extension before splatting.
    parameters:
      builder_class:
        type: type
      vec_side:
        type: str
      vec_elem:
        type: str
      scalar_elem:
        type: str
      expected_elem:
        type: str
      conversion:
        type: str
    """
    visitor = _make_visitor_in_fn()
    types = {
        "float": visitor._llvm.FLOAT_TYPE,
        "double": visitor._llvm.DOUBLE_TYPE,
    }
    vec_ty = ir.VectorType(types[vec_elem], VEC_LEN)
    vec_val = ir.Constant(vec_ty, [1.0, 2.0, 3.0, 4.0])
    scalar_val = ir.Constant(types[scalar_elem], 10.0)

    if vec_side == "lhs":
        result = _visit_binop(visitor, vec_val, scalar_val, "+")
    else:
        result = _visit_binop(visitor, scalar_val, vec_val, "+")

    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == VEC_LEN
    assert result.type.element == types[expected_elem]

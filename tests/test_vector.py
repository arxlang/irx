"""
title: Tests for vector operations in the LLVM-IR builder.
"""

from typing import Any

import astx
import pytest

from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

VEC4 = 4


def setup_builder() -> LLVMLiteIRVisitor:
    main_builder = LLVMLiteIR()
    visitor = main_builder.translator
    func_type = ir.FunctionType(visitor._llvm.INT32_TYPE, [])
    fn = ir.Function(visitor._llvm.module, func_type, name="main")
    bb = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(bb)
    return visitor


def _run_vector_binop(
    op_code: str,
    lhs_val: ir.Value,
    rhs_val: ir.Value,
    unsigned: Any = None,
    fma_rhs: ir.Value = None,
) -> ir.Value:
    builder = setup_builder()
    original_visit = builder.visit

    def mock_visit(node: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(node, astx.Identifier):
            if node.name == "LHS":
                builder.result_stack.append(lhs_val)
            elif node.name == "RHS":
                builder.result_stack.append(rhs_val)
            elif node.name == "FMA_RHS":
                builder.result_stack.append(fma_rhs)
            else:
                return original_visit(node, *args, **kwargs)
        else:
            return original_visit(node, *args, **kwargs)

    builder.visit = mock_visit  # type: ignore[method-assign]

    bin_op = astx.BinaryOp(
        op_code, astx.Identifier("LHS"), astx.Identifier("RHS")
    )
    if unsigned is not None:
        bin_op.unsigned = unsigned  # type: ignore[attr-defined]
    if fma_rhs is not None:
        bin_op.fma = True  # type: ignore[attr-defined]
        bin_op.fma_rhs = astx.Identifier("FMA_RHS")  # type: ignore[attr-defined]

    builder.visit(bin_op)
    return builder.result_stack.pop()


def test_float_vector_add() -> None:
    """
    title: float vector add emits fadd instruction.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1.0] * VEC4)
    v2 = ir.Constant(vec_ty, [2.0] * VEC4)
    result = _run_vector_binop("+", v1, v2)
    assert "fadd" in str(result)
    assert isinstance(result.type, ir.VectorType)


def test_float_vector_sub() -> None:
    """
    title: float vector sub emits fsub instruction.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [4.0] * VEC4)
    v2 = ir.Constant(vec_ty, [1.0] * VEC4)
    result = _run_vector_binop("-", v1, v2)
    assert "fsub" in str(result)


def test_float_vector_mul() -> None:
    """
    title: float vector mul emits fmul instruction.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [2.0] * VEC4)
    v2 = ir.Constant(vec_ty, [3.0] * VEC4)
    result = _run_vector_binop("*", v1, v2)
    assert "fmul" in str(result)


def test_float_vector_div() -> None:
    """
    title: float vector div emits fdiv instruction.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [6.0] * VEC4)
    v2 = ir.Constant(vec_ty, [2.0] * VEC4)
    result = _run_vector_binop("/", v1, v2)
    assert "fdiv" in str(result)


def test_double_vector_add() -> None:
    """
    title: double vector add emits fadd and preserves element type.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.DOUBLE_TYPE, 2)
    v1 = ir.Constant(vec_ty, [1.0, 2.0])
    v2 = ir.Constant(vec_ty, [3.0, 4.0])
    result = _run_vector_binop("+", v1, v2)
    assert "fadd" in str(result)
    assert result.type.element == builder._llvm.DOUBLE_TYPE


def test_int_vector_add() -> None:
    """
    title: int vector add emits add instruction.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1] * VEC4)
    v2 = ir.Constant(vec_ty, [2] * VEC4)
    result = _run_vector_binop("+", v1, v2)
    assert "add" in str(result)
    assert isinstance(result.type, ir.VectorType)


def test_int_vector_sub() -> None:
    """
    title: int vector sub emits sub instruction.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [5] * VEC4)
    v2 = ir.Constant(vec_ty, [3] * VEC4)
    result = _run_vector_binop("-", v1, v2)
    assert "sub" in str(result)


def test_int_vector_mul() -> None:
    """
    title: int vector mul emits mul instruction.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [2] * VEC4)
    v2 = ir.Constant(vec_ty, [3] * VEC4)
    result = _run_vector_binop("*", v1, v2)
    assert "mul" in str(result)


def test_int_vector_sdiv() -> None:
    """
    title: int vector signed div emits sdiv instruction.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [10] * VEC4)
    v2 = ir.Constant(vec_ty, [2] * VEC4)
    result = _run_vector_binop("/", v1, v2, unsigned=False)
    assert "sdiv" in str(result)


def test_int_vector_udiv() -> None:
    """
    title: int vector unsigned div emits udiv instruction.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [10] * VEC4)
    v2 = ir.Constant(vec_ty, [2] * VEC4)
    result = _run_vector_binop("/", v1, v2, unsigned=True)
    assert "udiv" in str(result)


def test_lhs_vec_rhs_scalar_same_type() -> None:
    """
    title: vec + same-type scalar splats scalar to vector width.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1.0] * VEC4)
    scalar = ir.Constant(builder._llvm.FLOAT_TYPE, 2.0)
    result = _run_vector_binop("+", v1, scalar)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == VEC4


def test_rhs_vec_lhs_scalar_same_type() -> None:
    """
    title: scalar + same-type vec splats scalar to vector width.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1.0] * VEC4)
    scalar = ir.Constant(builder._llvm.FLOAT_TYPE, 2.0)
    result = _run_vector_binop("+", scalar, v1)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == VEC4


def test_lhs_float_vec_rhs_double_scalar() -> None:
    """
    title: float vec + double scalar truncates scalar to float.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1.0] * VEC4)
    scalar = ir.Constant(builder._llvm.DOUBLE_TYPE, 2.0)
    result = _run_vector_binop("+", v1, scalar)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.FLOAT_TYPE


def test_rhs_float_vec_lhs_double_scalar() -> None:
    """
    title: double scalar + float vec truncates scalar to float.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1.0] * VEC4)
    scalar = ir.Constant(builder._llvm.DOUBLE_TYPE, 2.0)
    result = _run_vector_binop("+", scalar, v1)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.FLOAT_TYPE


def test_lhs_double_vec_rhs_float_scalar() -> None:
    """
    title: double vec + float scalar extends scalar to double.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.DOUBLE_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1.0] * VEC4)
    scalar = ir.Constant(builder._llvm.FLOAT_TYPE, 2.0)
    result = _run_vector_binop("+", v1, scalar)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.DOUBLE_TYPE


def test_rhs_double_vec_lhs_float_scalar() -> None:
    """
    title: float scalar + double vec extends scalar to double.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.DOUBLE_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1.0] * VEC4)
    scalar = ir.Constant(builder._llvm.FLOAT_TYPE, 2.0)
    result = _run_vector_binop("+", scalar, v1)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.DOUBLE_TYPE


def test_int_lhs_vec_rhs_scalar() -> None:
    """
    title: int vec + int scalar splats scalar to vector.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1] * VEC4)
    scalar = ir.Constant(builder._llvm.INT32_TYPE, 2)
    result = _run_vector_binop("+", v1, scalar)
    assert isinstance(result.type, ir.VectorType)


def test_float_vector_fma() -> None:
    """
    title: float vector FMA uses llvm.fma intrinsic fallback.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [2.0] * VEC4)
    v2 = ir.Constant(vec_ty, [3.0] * VEC4)
    v3 = ir.Constant(vec_ty, [1.0] * VEC4)
    result = builder._emit_fma(v1, v2, v3)
    assert result is not None
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.FLOAT_TYPE
    assert "llvm.fma" in str(builder._llvm.module)


def test_double_vector_fma() -> None:
    """
    title: double vector FMA uses llvm.fma intrinsic fallback.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.DOUBLE_TYPE, 2)
    v1 = ir.Constant(vec_ty, [2.0] * 2)
    v2 = ir.Constant(vec_ty, [3.0] * 2)
    v3 = ir.Constant(vec_ty, [1.0] * 2)
    result = builder._emit_fma(v1, v2, v3)
    assert result is not None
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.DOUBLE_TYPE
    assert "llvm.fma" in str(builder._llvm.module)


def test_fma_missing_fma_rhs_raises() -> None:
    """
    title: FMA without fma_rhs operand raises an exception.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1.0] * VEC4)
    v2 = ir.Constant(vec_ty, [2.0] * VEC4)

    original_visit = builder.visit

    def mock_visit(node: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(node, astx.Identifier):
            builder.result_stack.append(v1 if node.name == "LHS" else v2)
        else:
            return original_visit(node, *args, **kwargs)

    builder.visit = mock_visit  # type: ignore[method-assign]

    bin_op = astx.BinaryOp("*", astx.Identifier("LHS"), astx.Identifier("RHS"))
    bin_op.fma = True  # type: ignore[attr-defined]
    # deliberately omit fma_rhs

    with pytest.raises(Exception, match="FMA requires a third operand"):
        builder.visit(bin_op)


def test_fast_math_flag_cleared_after_op() -> None:
    """
    title: fast_math flag is cleared on the builder after the operation.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1.0] * VEC4)
    v2 = ir.Constant(vec_ty, [2.0] * VEC4)

    original_visit = builder.visit

    def mock_visit(node: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(node, astx.Identifier):
            builder.result_stack.append(v1 if node.name == "LHS" else v2)
        else:
            return original_visit(node, *args, **kwargs)

    builder.visit = mock_visit  # type: ignore[method-assign]

    bin_op = astx.BinaryOp("+", astx.Identifier("LHS"), astx.Identifier("RHS"))
    bin_op.fast_math = True  # type: ignore[attr-defined]
    builder.visit(bin_op)

    assert builder._fast_math_enabled is False


def test_vector_size_mismatch_raises() -> None:
    """
    title: mismatched vector sizes raise an exception.
    """
    builder = setup_builder()
    v1 = ir.Constant(ir.VectorType(builder._llvm.INT32_TYPE, VEC4), [1] * VEC4)
    v2 = ir.Constant(ir.VectorType(builder._llvm.INT32_TYPE, 2), [1] * 2)
    with pytest.raises(Exception, match="Vector size mismatch"):
        _run_vector_binop("+", v1, v2)


def test_vector_element_type_mismatch_raises() -> None:
    """
    title: mismatched vector element types raise an exception.
    """
    builder = setup_builder()
    v1 = ir.Constant(ir.VectorType(builder._llvm.INT32_TYPE, 2), [1] * 2)
    v2 = ir.Constant(ir.VectorType(builder._llvm.INT64_TYPE, 2), [1] * 2)
    with pytest.raises(Exception, match="Vector element type mismatch"):
        _run_vector_binop("+", v1, v2)


def test_vector_unsupported_op_raises() -> None:
    """
    title: unsupported vector binary op raises an exception.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1] * VEC4)
    v2 = ir.Constant(vec_ty, [2] * VEC4)
    with pytest.raises(Exception, match=r"Vector binop .* not implemented"):
        _run_vector_binop("%", v1, v2)


@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_float_vector_comparison_raises(op: str) -> None:
    """
    title: comparison ops are not implemented.
    parameters:
      op:
        type: str
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v1 = ir.Constant(vec_ty, [1.0] * VEC4)
    v2 = ir.Constant(vec_ty, [2.0] * VEC4)
    with pytest.raises(Exception):
        _run_vector_binop(op, v1, v2)

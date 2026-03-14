"""
title: Tests for vector operations in the LLVM-IR builder.
"""

from typing import Any

import astx
import pytest

from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir


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

    builder.visit = mock_visit

    bin_op = astx.BinaryOp(op_code, astx.Identifier("LHS"), astx.Identifier("RHS"))
    if unsigned is not None:
        bin_op.unsigned = unsigned  # type: ignore[attr-defined]
    if fma_rhs is not None:
        bin_op.fma = True  # type: ignore[attr-defined]
        bin_op.fma_rhs = astx.Identifier("FMA_RHS")  # type: ignore[attr-defined]

    builder.visit(bin_op)
    return builder.result_stack.pop()


def test_float_vector_add() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1.0] * 4)
    v2 = ir.Constant(vec_ty, [2.0] * 4)
    result = _run_vector_binop("+", v1, v2)
    assert "fadd" in str(result)
    assert isinstance(result.type, ir.VectorType)


def test_float_vector_sub() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [4.0] * 4)
    v2 = ir.Constant(vec_ty, [1.0] * 4)
    result = _run_vector_binop("-", v1, v2)
    assert "fsub" in str(result)


def test_float_vector_mul() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [2.0] * 4)
    v2 = ir.Constant(vec_ty, [3.0] * 4)
    result = _run_vector_binop("*", v1, v2)
    assert "fmul" in str(result)


def test_float_vector_div() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [6.0] * 4)
    v2 = ir.Constant(vec_ty, [2.0] * 4)
    result = _run_vector_binop("/", v1, v2)
    assert "fdiv" in str(result)


def test_double_vector_add() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.DOUBLE_TYPE, 2)
    v1 = ir.Constant(vec_ty, [1.0, 2.0])
    v2 = ir.Constant(vec_ty, [3.0, 4.0])
    result = _run_vector_binop("+", v1, v2)
    assert "fadd" in str(result)
    assert result.type.element == builder._llvm.DOUBLE_TYPE


def test_int_vector_add() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1] * 4)
    v2 = ir.Constant(vec_ty, [2] * 4)
    result = _run_vector_binop("+", v1, v2)
    assert "add" in str(result)
    assert isinstance(result.type, ir.VectorType)


def test_int_vector_sub() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    v1 = ir.Constant(vec_ty, [5] * 4)
    v2 = ir.Constant(vec_ty, [3] * 4)
    result = _run_vector_binop("-", v1, v2)
    assert "sub" in str(result)


def test_int_vector_mul() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    v1 = ir.Constant(vec_ty, [2] * 4)
    v2 = ir.Constant(vec_ty, [3] * 4)
    result = _run_vector_binop("*", v1, v2)
    assert "mul" in str(result)


def test_int_vector_sdiv() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    v1 = ir.Constant(vec_ty, [10] * 4)
    v2 = ir.Constant(vec_ty, [2] * 4)
    result = _run_vector_binop("/", v1, v2, unsigned=False)
    assert "sdiv" in str(result)


def test_int_vector_udiv() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    v1 = ir.Constant(vec_ty, [10] * 4)
    v2 = ir.Constant(vec_ty, [2] * 4)
    result = _run_vector_binop("/", v1, v2, unsigned=True)
    assert "udiv" in str(result)


def test_lhs_vec_rhs_scalar_same_type() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1.0] * 4)
    scalar = ir.Constant(builder._llvm.FLOAT_TYPE, 2.0)
    result = _run_vector_binop("+", v1, scalar)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == 4


def test_rhs_vec_lhs_scalar_same_type() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1.0] * 4)
    scalar = ir.Constant(builder._llvm.FLOAT_TYPE, 2.0)
    result = _run_vector_binop("+", scalar, v1)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == 4


def test_lhs_float_vec_rhs_double_scalar() -> None:
    """float vec + double scalar: scalar is truncated to float then splatted."""
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1.0] * 4)
    scalar = ir.Constant(builder._llvm.DOUBLE_TYPE, 2.0)
    result = _run_vector_binop("+", v1, scalar)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.FLOAT_TYPE


def test_rhs_float_vec_lhs_double_scalar() -> None:
    """double scalar + float vec: scalar is truncated to float then splatted."""
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1.0] * 4)
    scalar = ir.Constant(builder._llvm.DOUBLE_TYPE, 2.0)
    result = _run_vector_binop("+", scalar, v1)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.FLOAT_TYPE


def test_lhs_double_vec_rhs_float_scalar() -> None:
    """double vec + float scalar: scalar is extended to double then splatted."""
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.DOUBLE_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1.0] * 4)
    scalar = ir.Constant(builder._llvm.FLOAT_TYPE, 2.0)
    result = _run_vector_binop("+", v1, scalar)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.DOUBLE_TYPE


def test_rhs_double_vec_lhs_float_scalar() -> None:
    """float scalar + double vec: scalar is extended to double then splatted."""
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.DOUBLE_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1.0] * 4)
    scalar = ir.Constant(builder._llvm.FLOAT_TYPE, 2.0)
    result = _run_vector_binop("+", scalar, v1)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.DOUBLE_TYPE


def test_int_lhs_vec_rhs_scalar() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1] * 4)
    scalar = ir.Constant(builder._llvm.INT32_TYPE, 2)
    result = _run_vector_binop("+", v1, scalar)
    assert isinstance(result.type, ir.VectorType)


def test_float_vector_fma() -> None:
    """Vector FMA goes via the llvm.fma.* intrinsic — llvmlite's builder.fma
    rejects VectorType operands, so _emit_fma must use the intrinsic fallback."""
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [2.0] * 4)
    v2 = ir.Constant(vec_ty, [3.0] * 4)
    v3 = ir.Constant(vec_ty, [1.0] * 4)
    result = builder._emit_fma(v1, v2, v3)
    assert result is not None
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == builder._llvm.FLOAT_TYPE
    assert "llvm.fma" in str(builder._llvm.module)


def test_double_vector_fma() -> None:
    """Same as above for double vectors."""
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
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1.0] * 4)
    v2 = ir.Constant(vec_ty, [2.0] * 4)

    original_visit = builder.visit

    def mock_visit(node: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(node, astx.Identifier):
            builder.result_stack.append(v1 if node.name == "LHS" else v2)
        else:
            return original_visit(node, *args, **kwargs)

    builder.visit = mock_visit

    bin_op = astx.BinaryOp("*", astx.Identifier("LHS"), astx.Identifier("RHS"))
    bin_op.fma = True  # type: ignore[attr-defined]
    # deliberately omit fma_rhs

    with pytest.raises(Exception, match="FMA requires a third operand"):
        builder.visit(bin_op)


def test_fast_math_flag_cleared_after_op() -> None:
    """fast_math=True on a BinaryOp must not leave the flag set afterward."""
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1.0] * 4)
    v2 = ir.Constant(vec_ty, [2.0] * 4)

    original_visit = builder.visit

    def mock_visit(node: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(node, astx.Identifier):
            builder.result_stack.append(v1 if node.name == "LHS" else v2)
        else:
            return original_visit(node, *args, **kwargs)

    builder.visit = mock_visit

    bin_op = astx.BinaryOp("+", astx.Identifier("LHS"), astx.Identifier("RHS"))
    bin_op.fast_math = True  # type: ignore[attr-defined]
    builder.visit(bin_op)

    assert builder._fast_math_enabled is False


def test_vector_size_mismatch_raises() -> None:
    builder = setup_builder()
    v1 = ir.Constant(ir.VectorType(builder._llvm.INT32_TYPE, 4), [1] * 4)
    v2 = ir.Constant(ir.VectorType(builder._llvm.INT32_TYPE, 2), [1] * 2)
    with pytest.raises(Exception, match="Vector size mismatch"):
        _run_vector_binop("+", v1, v2)


def test_vector_element_type_mismatch_raises() -> None:
    builder = setup_builder()
    v1 = ir.Constant(ir.VectorType(builder._llvm.INT32_TYPE, 2), [1] * 2)
    v2 = ir.Constant(ir.VectorType(builder._llvm.INT64_TYPE, 2), [1] * 2)
    with pytest.raises(Exception, match="Vector element type mismatch"):
        _run_vector_binop("+", v1, v2)


def test_vector_unsupported_op_raises() -> None:
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1] * 4)
    v2 = ir.Constant(vec_ty, [2] * 4)
    with pytest.raises(Exception, match="Vector binop .* not implemented"):
        _run_vector_binop("%", v1, v2)


@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_float_vector_comparison_raises(op: str) -> None:
    """Comparison ops are not implemented for vectors."""
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1 = ir.Constant(vec_ty, [1.0] * 4)
    v2 = ir.Constant(vec_ty, [2.0] * 4)
    with pytest.raises(Exception):
        _run_vector_binop(op, v1, v2)
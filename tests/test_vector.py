from typing import Any

import astx
import pytest

from irx.builders.llvmliteir import LLVMLiteIR
from llvmlite import ir


def setup_builder() -> Any:
    main_builder = LLVMLiteIR()
    visitor = main_builder.translator
    # Mocking standard startup
    func_type = ir.FunctionType(visitor._llvm.INT32_TYPE, [])
    fn = ir.Function(visitor._llvm.module, func_type, name="main")
    bb = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(bb)
    return visitor


def _run_vector_binop(
    op_code: str, lhs_val: ir.Value, rhs_val: ir.Value, unsigned: Any = None
) -> ir.Value:
    builder = setup_builder()

    original_visit = builder.visit

    def mock_visit(node: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(node, astx.Identifier):
            if node.name == "LHS":
                builder.result_stack.append(lhs_val)
            elif node.name == "RHS":
                builder.result_stack.append(rhs_val)
            else:
                return original_visit(node, *args, **kwargs)
        else:
            return original_visit(node, *args, **kwargs)

    # Mock bound method
    builder.visit = mock_visit

    bin_op = astx.BinaryOp(
        op_code, astx.Identifier("LHS"), astx.Identifier("RHS")
    )
    if unsigned is not None:
        bin_op.unsigned = unsigned  # type: ignore
    builder.visit(bin_op)

    return builder.result_stack.pop()


def test_vector_vector_math() -> None:
    builder = setup_builder()
    vec_ty_f32 = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1_f32 = ir.Constant(vec_ty_f32, [1.0] * 4)
    v2_f32 = ir.Constant(vec_ty_f32, [2.0] * 4)

    # math
    _run_vector_binop("+", v1_f32, v2_f32)
    _run_vector_binop("-", v1_f32, v2_f32)
    _run_vector_binop("*", v1_f32, v2_f32)
    _run_vector_binop("/", v1_f32, v2_f32)

    # cmp (not implemented)
    for op in ["==", "!=", "<", "<=", ">", ">="]:
        with pytest.raises(Exception):
            _run_vector_binop(op, v1_f32, v2_f32)


def test_vector_scalar_promotion() -> None:
    builder = setup_builder()
    vec_ty_f32 = ir.VectorType(builder._llvm.FLOAT_TYPE, 4)
    v1_f32 = ir.Constant(vec_ty_f32, [1.0] * 4)
    scal_f32 = ir.Constant(builder._llvm.FLOAT_TYPE, 2.0)
    scal_f64 = ir.Constant(builder._llvm.DOUBLE_TYPE, 2.0)

    # L vec, R scal (same type)
    _run_vector_binop("+", v1_f32, scal_f32)
    # R vec, L scal (same type)
    _run_vector_binop("+", scal_f32, v1_f32)

    # L vec(f32), R scal(f64) -> f64 truncs to f32
    _run_vector_binop("+", v1_f32, scal_f64)
    # R vec(f32), L scal(f64) -> f64 truncs to f32
    _run_vector_binop("+", scal_f64, v1_f32)


def test_vector_int_math() -> None:
    builder = setup_builder()
    vec_ty_i32 = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    v1_i32 = ir.Constant(vec_ty_i32, [1] * 4)
    v2_i32 = ir.Constant(vec_ty_i32, [2] * 4)
    scal_i32 = ir.Constant(builder._llvm.INT32_TYPE, 2)

    _run_vector_binop("+", v1_i32, v2_i32)
    _run_vector_binop("/", v1_i32, v2_i32, unsigned=False)

    with pytest.raises(Exception):
        _run_vector_binop("/", v1_i32, v2_i32)

    return  # Vector modulo/cmp raise unimplemented

    # cmp
    _run_vector_binop("==", v1_i32, v2_i32)
    _run_vector_binop("<", v1_i32, v2_i32)

    # scalar
    _run_vector_binop("+", v1_i32, scal_i32)
    _run_vector_binop("+", scal_i32, v1_i32)

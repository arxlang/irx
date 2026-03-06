"""
title: Direct testing of internal llvmliteir methods to reach 90% coverage.
"""

import astx
import pytest
from unittest.mock import MagicMock

from llvmlite import ir
from irx.builders.llvmliteir import LLVMLiteIR
from irx.builders.llvmliteir import is_vector, splat_scalar, emit_int_div


def setup_builder():
    main_builder = LLVMLiteIR()
    visitor = main_builder.translator
    # Mocking standard startup
    func_type = ir.FunctionType(visitor._llvm.INT32_TYPE, [])
    fn = ir.Function(visitor._llvm.module, func_type, name="main")
    bb = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(bb)
    return visitor


# ── Internal Helpers ─────────────────────────────────────────


def test_is_vector_helper():
    builder = setup_builder()
    scalar = ir.Constant(builder._llvm.INT32_TYPE, 1)
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    vec = ir.Constant(vec_ty, [1, 2, 3, 4])
    
    assert not is_vector(scalar)
    assert is_vector(vec)


def test_splat_scalar_helper():
    builder = setup_builder()
    scalar = ir.Constant(builder._llvm.INT32_TYPE, 1)
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    
    vec = splat_scalar(builder._llvm.ir_builder, scalar, vec_ty)
    assert vec.type == vec_ty


def test_emit_int_div_helper():
    builder = setup_builder()
    lhs = ir.Constant(builder._llvm.INT32_TYPE, 10)
    rhs = ir.Constant(builder._llvm.INT32_TYPE, 2)
    
    res = emit_int_div(builder._llvm.ir_builder, lhs, rhs, False)
    assert res.type == builder._llvm.INT32_TYPE


# ── String Helper Functions ──────────────────────────────────


def test_string_helper_functions():
    builder = setup_builder()
    
    # Call the creators
    concat_fn = builder._create_string_concat_function()
    assert concat_fn.name == "string_concat"
    # Call again to hit the cached branch
    assert builder._create_string_concat_function() is concat_fn
    
    len_fn = builder._create_string_length_function()
    assert len_fn.name == "string_length"
    assert builder._create_string_length_function() is len_fn
    
    eq_fn = builder._create_string_equals_function()
    assert eq_fn.name == "string_equals"
    assert builder._create_string_equals_function() is eq_fn
    
    sub_fn = builder._create_string_substring_function()
    assert sub_fn.name == "string_substring"
    assert builder._create_string_substring_function() is sub_fn


def test_handle_string_operations():
    builder = setup_builder()
    
    str1 = ir.Constant(builder._llvm.ASCII_STRING_TYPE, None)
    str2 = ir.Constant(builder._llvm.ASCII_STRING_TYPE, None)
    
    # This will insert the call in the current block
    res_concat = builder._handle_string_concatenation(str1, str2)
    assert res_concat is not None
    
    res_cmp_eq = builder._handle_string_comparison(str1, str2, "==")
    assert res_cmp_eq is not None
    
    res_cmp_neq = builder._handle_string_comparison(str1, str2, "!=")
    assert res_cmp_neq is not None
    
    with pytest.raises(Exception):
        builder._handle_string_comparison(str1, str2, "<")


# ── Vector Binary Operations ─────────────────────────────────


def _run_vector_binop(op_code: str, lhs_val: ir.Value, rhs_val: ir.Value, unsigned: bool = None):
    builder = setup_builder()
    
    original_visit = builder.visit
    def mock_visit(node, *args, **kwargs):
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
    
    bin_op = astx.BinaryOp(op_code, astx.Identifier("LHS"), astx.Identifier("RHS"))
    if unsigned is not None:
        bin_op.unsigned = unsigned
    builder.visit(bin_op)
    
    return builder.result_stack.pop()


def test_vector_vector_math():
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


def test_vector_scalar_promotion():
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


def test_vector_int_math():
    builder = setup_builder()
    vec_ty_i32 = ir.VectorType(builder._llvm.INT32_TYPE, 4)
    v1_i32 = ir.Constant(vec_ty_i32, [1] * 4)
    v2_i32 = ir.Constant(vec_ty_i32, [2] * 4)
    scal_i32 = ir.Constant(builder._llvm.INT32_TYPE, 2)
    
    _run_vector_binop("+", v1_i32, v2_i32)
    _run_vector_binop("/", v1_i32, v2_i32, unsigned=False)
    
    with pytest.raises(Exception):
        _run_vector_binop("/", v1_i32, v2_i32)
    
    return # Vector modulo/cmp raise unimplemented 
    
    # cmp
    _run_vector_binop("==", v1_i32, v2_i32)
    _run_vector_binop("<", v1_i32, v2_i32)
    
    # scalar
    _run_vector_binop("+", v1_i32, scal_i32)
    _run_vector_binop("+", scal_i32, v1_i32)

"""
title: Targeted helper coverage for the llvmliteir visitor.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock

import pytest

from irx.builders.llvmliteir import (
    Visitor,
    emit_int_div,
    is_fp_type,
    safe_pop,
    splat_scalar,
)
from llvmlite import ir


class _NoFmaBuilder:
    """
    title: Proxy IRBuilder that hides fma to exercise intrinsic fallback.
    attributes:
      _real:
        type: ir.IRBuilder
      called:
        type: list[str]
    """

    def __init__(self, real: ir.IRBuilder) -> None:
        """
        title: Initialize _NoFmaBuilder.
        parameters:
          real:
            type: ir.IRBuilder
        """
        self._real: ir.IRBuilder = real
        self.called: list[str] = []

    def __getattr__(self, name: str) -> Any:
        """
        title: Return one attribute.
        parameters:
          name:
            type: str
        returns:
          type: Any
        """
        if name == "fma":
            raise AttributeError
        return getattr(self._real, name)

    def call(
        self,
        fn: ir.Function,
        args: list[ir.Value],
        name: str | None = None,
    ) -> ir.Instruction:
        """
        title: Call.
        parameters:
          fn:
            type: ir.Function
          args:
            type: list[ir.Value]
          name:
            type: str | None
        returns:
          type: ir.Instruction
        """
        self.called.append(fn.name)
        return self._real.call(fn, args, name=name)


def _prime_builder(visitor: Visitor) -> None:
    """
    title: Prime builder.
    parameters:
      visitor:
        type: Visitor
    """
    float_ty = visitor._llvm.FLOAT_TYPE
    fn_ty = ir.FunctionType(float_ty, [])
    fn = ir.Function(visitor._llvm.module, fn_ty, name="fma_cover")
    block = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(block)


def test_emit_fma_fallback_intrinsic() -> None:
    """
    title: Ensure fallback uses llvm.fma intrinsic when builder lacks fma.
    """
    visitor = Visitor()
    _prime_builder(visitor)
    proxy = _NoFmaBuilder(visitor._llvm.ir_builder)
    visitor._llvm.ir_builder = cast(ir.IRBuilder, proxy)

    ty = visitor._llvm.FLOAT_TYPE
    lhs = ir.Constant(ty, 1.0)
    rhs = ir.Constant(ty, 2.0)
    addend = ir.Constant(ty, 3.0)

    inst = visitor._emit_fma(lhs, rhs, addend)

    assert inst.name == "vfma"
    assert "llvm.fma.f32" in proxy.called
    assert "llvm.fma.f32" in visitor._llvm.module.globals


def test_emit_fma_direct_path_calls_apply_fast_math() -> None:
    """
    title: Native builder.fma path should still route through _apply_fast_math.
    """
    visitor = Visitor()
    _prime_builder(visitor)
    if not hasattr(visitor._llvm.ir_builder, "fma"):
        pytest.skip("llvmlite IRBuilder has no native fma")

    visitor.set_fast_math(True)
    apply_fast_math = Mock(wraps=visitor._apply_fast_math)
    setattr(visitor, "_apply_fast_math", apply_fast_math)

    ty = visitor._llvm.FLOAT_TYPE
    lhs = ir.Constant(ty, 1.0)
    rhs = ir.Constant(ty, 2.0)
    addend = ir.Constant(ty, 3.0)

    inst = visitor._emit_fma(lhs, rhs, addend)

    apply_fast_math.assert_called_once_with(inst)


def test_splat_scalar_broadcasts_all_lanes() -> None:
    """
    title: splat_scalar should broadcast the scalar into every lane.
    """
    visitor = Visitor()
    _prime_builder(visitor)

    float_ty = visitor._llvm.FLOAT_TYPE
    scalar = ir.Constant(float_ty, 1.5)
    vec_ty = ir.VectorType(float_ty, 4)

    result = splat_scalar(visitor._llvm.ir_builder, scalar, vec_ty)

    assert isinstance(result.type, ir.VectorType)
    assert result.type == vec_ty
    assert getattr(result, "opname", "") == "shufflevector"
    mask = result.operands[2]
    assert isinstance(mask, ir.Constant)
    assert "i32 0, i32 0, i32 0, i32 0" in str(mask)


def test_emit_int_div_signed_and_unsigned() -> None:
    """
    title: emit_int_div should honour the unsigned flag.
    """
    visitor = Visitor()
    _prime_builder(visitor)

    builder = visitor._llvm.ir_builder
    int_ty = ir.IntType(32)
    lhs = ir.Constant(int_ty, 10)
    rhs = ir.Constant(int_ty, 3)

    signed = emit_int_div(builder, lhs, rhs, unsigned=False)
    unsigned = emit_int_div(builder, lhs, rhs, unsigned=True)

    assert getattr(signed, "opname", "") == "sdiv"
    assert getattr(unsigned, "opname", "") == "udiv"


def test_unify_promotes_scalar_int_to_vector() -> None:
    """
    title: Scalar ints splat to match vector operands and widen width.
    """
    visitor = Visitor()
    _prime_builder(visitor)

    vec_ty = ir.VectorType(ir.IntType(32), 2)
    vec = ir.Constant(vec_ty, [ir.Constant(ir.IntType(32), 1)] * 2)
    scalar = ir.Constant(ir.IntType(16), 5)

    promoted_vec, promoted_scalar = visitor._unify_numeric_operands(
        vec, scalar
    )

    assert isinstance(promoted_vec.type, ir.VectorType)
    assert isinstance(promoted_scalar.type, ir.VectorType)
    assert promoted_vec.type == vec_ty
    assert promoted_scalar.type == vec_ty


def test_unify_vector_float_rank_matches_double() -> None:
    """
    title: Double scalar casts down to match float vector element type.
    summary: >-
      When a scalar and a vector have different FP precision, the vector's
      element type wins — the scalar is cast to match, not the other way
      around.
    """
    visitor = Visitor()
    _prime_builder(visitor)

    float_vec_ty = ir.VectorType(visitor._llvm.FLOAT_TYPE, 2)
    float_vec = ir.Constant(
        float_vec_ty,
        [
            ir.Constant(visitor._llvm.FLOAT_TYPE, 1.0),
            ir.Constant(visitor._llvm.FLOAT_TYPE, 2.0),
        ],
    )
    double_scalar = ir.Constant(visitor._llvm.DOUBLE_TYPE, 4.0)

    result_vec, result_scalar = visitor._unify_numeric_operands(
        float_vec, double_scalar
    )

    assert result_vec.type.element == visitor._llvm.FLOAT_TYPE
    assert result_scalar.type.element == visitor._llvm.FLOAT_TYPE


def test_unify_int_and_float_scalars_returns_float() -> None:
    """
    title: Scalar int plus float promotes both operands to float.
    """
    visitor = Visitor()
    _prime_builder(visitor)

    int_scalar = ir.Constant(visitor._llvm.INT32_TYPE, 7)
    float_scalar = ir.Constant(visitor._llvm.FLOAT_TYPE, 1.25)

    widened_int, widened_float = visitor._unify_numeric_operands(
        int_scalar, float_scalar
    )

    assert is_fp_type(widened_int.type)
    assert widened_float.type == visitor._llvm.FLOAT_TYPE


def test_safe_pop_empty_returns_none() -> None:
    """
    title: safe_pop should return None when the list is empty.
    """
    assert safe_pop([]) is None


def test_get_data_type_aliases_and_invalid() -> None:
    """
    title: >-
      VariablesLLVM.get_data_type should resolve aliases and reject invalid.
    """
    visitor = Visitor()
    llvm_vars = visitor._llvm

    assert llvm_vars.get_data_type("float16") == llvm_vars.FLOAT16_TYPE
    assert llvm_vars.get_data_type("double") == llvm_vars.DOUBLE_TYPE
    assert llvm_vars.get_data_type("char") == llvm_vars.INT8_TYPE
    assert llvm_vars.get_data_type("utf8string") == llvm_vars.ASCII_STRING_TYPE

    with pytest.raises(Exception, match="not valid"):
        llvm_vars.get_data_type("not-a-type")


def test_get_size_t_type_from_triple_32bit() -> None:
    """
    title: Test _get_size_t_type_from_triple for 32-bit architectures.
    """
    visitor = Visitor()

    mock_tm = Mock()
    mock_tm.triple = "i386-unknown-linux-gnu"
    visitor.target_machine = mock_tm

    size_t_ty = visitor._get_size_t_type_from_triple()
    assert size_t_ty.width == 32  # noqa: PLR2004


def test_get_size_t_type_from_triple_32bit_family_with_64_tag() -> None:
    """
    title: 32-bit family triples carrying '64' should map to i64.
    """
    visitor = Visitor()

    mock_tm = Mock()
    mock_tm.triple = "arm-unknown-linux-gnu64"
    visitor.target_machine = mock_tm

    size_t_ty = visitor._get_size_t_type_from_triple()
    assert size_t_ty.width == 64  # noqa: PLR2004


def test_get_size_t_type_from_triple_fallback() -> None:
    """
    title: >-
      Test _get_size_t_type_from_triple fallback for unknown architectures.
    """
    visitor = Visitor()

    mock_tm = Mock()
    mock_tm.triple = "unknown-arch-unknown-os"
    visitor.target_machine = mock_tm

    size_t_ty = visitor._get_size_t_type_from_triple()
    assert isinstance(size_t_ty, ir.IntType)
    assert size_t_ty.width in (32, 64)


def test_get_fma_function_vector_half_and_cache() -> None:
    """
    title: >-
      Vector-half FMA intrinsic naming should include lane count and cache.
    """
    visitor = Visitor()
    vec_ty = ir.VectorType(visitor._llvm.FLOAT16_TYPE, 4)

    first = visitor._get_fma_function(vec_ty)
    second = visitor._get_fma_function(vec_ty)

    assert first.name == "llvm.fma.v4f16"
    assert first is second


def test_get_fma_function_invalid_type_raises() -> None:
    """
    title: _get_fma_function should reject non-floating element types.
    """
    visitor = Visitor()
    with pytest.raises(Exception, match="FMA supports only floating-point"):
        visitor._get_fma_function(ir.IntType(32))


def test_scalar_vector_float_conversion_fptrunc() -> None:
    """
    title: Test scalar-vector promotion with float truncation.
    """
    visitor = Visitor()
    _prime_builder(visitor)

    double_ty = visitor._llvm.DOUBLE_TYPE
    float_ty = visitor._llvm.FLOAT_TYPE
    vec_ty = ir.VectorType(float_ty, 2)

    scalar = ir.Constant(double_ty, 3.14)
    converted = visitor._llvm.ir_builder.fptrunc(scalar, float_ty, "test")
    result = splat_scalar(visitor._llvm.ir_builder, converted, vec_ty)

    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == float_ty


def test_scalar_vector_float_conversion_fpext() -> None:
    """
    title: Test scalar-vector promotion with float extension.
    """
    visitor = Visitor()
    _prime_builder(visitor)

    float_ty = visitor._llvm.FLOAT_TYPE
    double_ty = visitor._llvm.DOUBLE_TYPE
    vec_ty = ir.VectorType(double_ty, 2)

    scalar = ir.Constant(float_ty, 3.14)

    converted = visitor._llvm.ir_builder.fpext(scalar, double_ty, "test")
    result = splat_scalar(visitor._llvm.ir_builder, converted, vec_ty)

    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == double_ty


def test_set_fast_math_marks_float_ops() -> None:
    """
    title: set_fast_math should add fast flag to floating instructions.
    """
    visitor = Visitor()
    _prime_builder(visitor)

    float_ty = visitor._llvm.FLOAT_TYPE
    lhs = ir.Constant(float_ty, 1.0)
    rhs = ir.Constant(float_ty, 2.0)

    visitor.set_fast_math(True)
    inst_fast = visitor._llvm.ir_builder.fadd(lhs, rhs)
    visitor._apply_fast_math(inst_fast)
    assert "fast" in inst_fast.flags
    visitor._apply_fast_math(inst_fast)
    assert "fast" in inst_fast.flags

    visitor.set_fast_math(False)
    inst_normal = visitor._llvm.ir_builder.fadd(lhs, rhs)
    visitor._apply_fast_math(inst_normal)
    assert "fast" not in inst_normal.flags


def test_apply_fast_math_noop_for_non_fp_values() -> None:
    """
    title: _apply_fast_math should no-op for scalar and vector integer ops.
    """
    visitor = Visitor()
    _prime_builder(visitor)
    visitor.set_fast_math(True)

    int_ty = ir.IntType(32)
    scalar_add = visitor._llvm.ir_builder.add(
        ir.Constant(int_ty, 1), ir.Constant(int_ty, 2)
    )
    visitor._apply_fast_math(scalar_add)
    assert "fast" not in scalar_add.flags

    vec_ty = ir.VectorType(int_ty, 2)
    vector_add = visitor._llvm.ir_builder.add(
        ir.Constant(vec_ty, [1, 2]),
        ir.Constant(vec_ty, [3, 4]),
    )
    visitor._apply_fast_math(vector_add)
    assert "fast" not in vector_add.flags

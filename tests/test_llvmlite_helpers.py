"""Targeted helper coverage for LLVMLiteIRVisitor."""

from __future__ import annotations

from typing import Any, cast

from irx.builders.llvmliteir import (
    LLVMLiteIRVisitor,
    emit_int_div,
    is_fp_type,
    splat_scalar,
)
from llvmlite import ir


class _NoFmaBuilder:
    """Proxy IRBuilder that hides fma to exercise intrinsic fallback."""

    def __init__(self, real: ir.IRBuilder) -> None:
        self._real = real
        self.called: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name == "fma":
            raise AttributeError
        return getattr(self._real, name)

    def call(
        self,
        fn: ir.Function,
        args: list[ir.Value],
        name: str | None = None,
    ) -> ir.Instruction:
        self.called.append(fn.name)
        return self._real.call(fn, args, name=name)


def _prime_builder(visitor: LLVMLiteIRVisitor) -> None:
    float_ty = visitor._llvm.FLOAT_TYPE
    fn_ty = ir.FunctionType(float_ty, [])
    fn = ir.Function(visitor._llvm.module, fn_ty, name="fma_cover")
    block = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(block)


def test_emit_fma_fallback_intrinsic() -> None:
    """Ensure fallback uses llvm.fma intrinsic when builder lacks fma."""
    visitor = LLVMLiteIRVisitor()
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


def test_splat_scalar_broadcasts_all_lanes() -> None:
    """splat_scalar should broadcast the scalar into every lane."""
    visitor = LLVMLiteIRVisitor()
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
    """emit_int_div should honour the unsigned flag."""
    visitor = LLVMLiteIRVisitor()
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
    """Scalar ints splat to match vector operands and widen width."""
    visitor = LLVMLiteIRVisitor()
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
    """Float vectors upgrade to match double scalars."""
    visitor = LLVMLiteIRVisitor()
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

    widened_vec, widened_scalar = visitor._unify_numeric_operands(
        float_vec, double_scalar
    )

    assert widened_vec.type.element == visitor._llvm.DOUBLE_TYPE
    assert widened_scalar.type.element == visitor._llvm.DOUBLE_TYPE


def test_unify_int_and_float_scalars_returns_float() -> None:
    """Scalar int + float promotes to float for both operands."""
    visitor = LLVMLiteIRVisitor()
    _prime_builder(visitor)

    int_scalar = ir.Constant(visitor._llvm.INT32_TYPE, 7)
    float_scalar = ir.Constant(visitor._llvm.FLOAT_TYPE, 1.25)

    widened_int, widened_float = visitor._unify_numeric_operands(
        int_scalar, float_scalar
    )

    assert is_fp_type(widened_int.type)
    assert widened_float.type == visitor._llvm.FLOAT_TYPE


def test_set_fast_math_marks_float_ops() -> None:
    """set_fast_math should add fast flag to floating instructions."""
    visitor = LLVMLiteIRVisitor()
    _prime_builder(visitor)

    float_ty = visitor._llvm.FLOAT_TYPE
    lhs = ir.Constant(float_ty, 1.0)
    rhs = ir.Constant(float_ty, 2.0)

    visitor.set_fast_math(True)
    inst_fast = visitor._llvm.ir_builder.fadd(lhs, rhs)
    visitor._apply_fast_math(inst_fast)
    assert "fast" in inst_fast.flags

    visitor.set_fast_math(False)
    inst_normal = visitor._llvm.ir_builder.fadd(lhs, rhs)
    visitor._apply_fast_math(inst_normal)
    assert "fast" not in inst_normal.flags

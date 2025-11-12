"""Targeted helper coverage for LLVMLiteIRVisitor."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock

from irx.builders.llvmliteir import (
    LLVMLiteIRVisitor,
    emit_int_div,
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


def test_get_size_t_type_from_triple_32bit() -> None:
    """Test _get_size_t_type_from_triple for 32-bit architectures."""
    visitor = LLVMLiteIRVisitor()
    
    mock_tm = Mock()
    mock_tm.triple = "i386-unknown-linux-gnu"
    visitor.target_machine = mock_tm
    
    size_t_ty = visitor._get_size_t_type_from_triple()
    assert size_t_ty.width == 32  # noqa: PLR2004


def test_get_size_t_type_from_triple_fallback() -> None:
    """Test _get_size_t_type_from_triple fallback for unknown architectures."""
    visitor = LLVMLiteIRVisitor()
    
    mock_tm = Mock()
    mock_tm.triple = "unknown-arch-unknown-os"
    visitor.target_machine = mock_tm
    
    size_t_ty = visitor._get_size_t_type_from_triple()
    assert isinstance(size_t_ty, ir.IntType)
    assert size_t_ty.width in (32, 64)


def test_scalar_vector_float_conversion_fptrunc() -> None:
    """Test scalar-vector promotion with float truncation."""
    visitor = LLVMLiteIRVisitor()
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
    """Test scalar-vector promotion with float extension."""
    visitor = LLVMLiteIRVisitor()
    _prime_builder(visitor)
    
    float_ty = visitor._llvm.FLOAT_TYPE
    double_ty = visitor._llvm.DOUBLE_TYPE
    vec_ty = ir.VectorType(double_ty, 2)
    
    scalar = ir.Constant(float_ty, 3.14)
    
    converted = visitor._llvm.ir_builder.fpext(scalar, double_ty, "test")
    result = splat_scalar(visitor._llvm.ir_builder, converted, vec_ty)
    
    assert isinstance(result.type, ir.VectorType)
    assert result.type.element == double_ty

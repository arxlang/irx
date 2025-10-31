"""Targeted helper coverage for LLVMLiteIRVisitor."""

from __future__ import annotations

from typing import Any

from llvmlite import ir

from irx.builders.llvmliteir import LLVMLiteIRVisitor


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
    visitor = LLVMLiteIRVisitor()
    _prime_builder(visitor)
    proxy = _NoFmaBuilder(visitor._llvm.ir_builder)
    visitor._llvm.ir_builder = proxy  # type: ignore[assignment]

    ty = visitor._llvm.FLOAT_TYPE
    lhs = ir.Constant(ty, 1.0)
    rhs = ir.Constant(ty, 2.0)
    addend = ir.Constant(ty, 3.0)

    inst = visitor._emit_fma(lhs, rhs, addend)

    assert inst.name == "vfma"
    assert "llvm.fma.f32" in proxy.called
    assert "llvm.fma.f32" in visitor._llvm.module.globals


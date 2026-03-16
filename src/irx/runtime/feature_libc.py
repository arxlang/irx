"""
title: Libc runtime feature declarations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llvmlite import ir

from irx.runtime.features import (
    ExternalSymbolSpec,
    RuntimeFeature,
    declare_external_function,
)

if TYPE_CHECKING:
    from irx.builders.llvmliteir import LLVMLiteIRVisitor


def build_libc_runtime_feature() -> RuntimeFeature:
    """
    title: Build the libc runtime feature specification.
    returns:
      type: RuntimeFeature
    """
    return RuntimeFeature(
        name="libc",
        symbols={
            "malloc": ExternalSymbolSpec("malloc", _declare_malloc),
            "puts": ExternalSymbolSpec("puts", _declare_puts),
            "snprintf": ExternalSymbolSpec("snprintf", _declare_snprintf),
        },
    )


def _declare_malloc(visitor: LLVMLiteIRVisitor) -> ir.Function:
    fn_type = ir.FunctionType(
        visitor._llvm.INT8_TYPE.as_pointer(),
        [visitor._llvm.SIZE_T_TYPE],
    )
    return declare_external_function(visitor._llvm.module, "malloc", fn_type)


def _declare_puts(visitor: LLVMLiteIRVisitor) -> ir.Function:
    fn_type = ir.FunctionType(
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.INT8_TYPE.as_pointer()],
    )
    return declare_external_function(visitor._llvm.module, "puts", fn_type)


def _declare_snprintf(visitor: LLVMLiteIRVisitor) -> ir.Function:
    fn_type = ir.FunctionType(
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.INT8_TYPE.as_pointer(),
            visitor._llvm.SIZE_T_TYPE,
            visitor._llvm.INT8_TYPE.as_pointer(),
        ],
        var_arg=True,
    )
    return declare_external_function(visitor._llvm.module, "snprintf", fn_type)

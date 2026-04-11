"""
title: Libm runtime feature declarations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llvmlite import ir

from irx.builder.runtime.features import (
    ExternalSymbolSpec,
    RuntimeFeature,
    declare_external_function,
)
from irx.typecheck import typechecked

if TYPE_CHECKING:
    from irx.builder.protocols import VisitorProtocol


@typechecked
def build_libm_runtime_feature() -> RuntimeFeature:
    """
    title: Build the libm runtime feature specification.
    returns:
      type: RuntimeFeature
    """
    return RuntimeFeature(
        name="libm",
        symbols={
            "sqrt": ExternalSymbolSpec("sqrt", _declare_sqrt),
        },
        linker_flags=("-lm",),
    )


@typechecked
def _declare_sqrt(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare sqrt.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.DOUBLE_TYPE,
        [visitor._llvm.DOUBLE_TYPE],
    )
    return declare_external_function(visitor._llvm.module, "sqrt", fn_type)

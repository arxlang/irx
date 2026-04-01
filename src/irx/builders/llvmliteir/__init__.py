"""
title: Package-based LLVMLiteIR builder.
"""

from irx.builders.llvmliteir.facade import LLVMLiteIR, LLVMLiteIRVisitor
from irx.builders.llvmliteir.runtime import safe_pop
from irx.builders.llvmliteir.types import (
    VariablesLLVM,
    is_fp_type,
    is_int_type,
)
from irx.builders.llvmliteir.vector import (
    emit_add,
    emit_int_div,
    is_vector,
    splat_scalar,
)

__all__ = [
    "LLVMLiteIR",
    "LLVMLiteIRVisitor",
    "VariablesLLVM",
    "emit_add",
    "emit_int_div",
    "is_fp_type",
    "is_int_type",
    "is_vector",
    "safe_pop",
    "splat_scalar",
]

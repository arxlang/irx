"""
title: Package-based llvmliteir backend.
"""

from irx.builders.llvmliteir.facade import Builder, Visitor
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
    "Builder",
    "VariablesLLVM",
    "Visitor",
    "emit_add",
    "emit_int_div",
    "is_fp_type",
    "is_int_type",
    "is_vector",
    "safe_pop",
    "splat_scalar",
]

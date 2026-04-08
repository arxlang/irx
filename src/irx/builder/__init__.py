"""
title: Public LLVM builder package.
"""

from irx.builder.backend import Builder, Visitor
from irx.builder.runtime import safe_pop
from irx.builder.types import (
    VariablesLLVM,
    is_fp_type,
    is_int_type,
)
from irx.builder.vector import (
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

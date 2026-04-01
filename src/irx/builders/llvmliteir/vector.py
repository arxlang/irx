"""
title: Vector helpers for LLVMLiteIR codegen.
"""

from irx.builders._llvmliteir_legacy import (
    emit_add,
    emit_int_div,
    is_vector,
    splat_scalar,
)

__all__ = ["emit_add", "emit_int_div", "is_vector", "splat_scalar"]

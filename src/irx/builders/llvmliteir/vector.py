"""
title: Vector helpers for llvmliteir codegen.
"""

from __future__ import annotations

from llvmlite import ir
from llvmlite.ir import VectorType

from irx.builders.llvmliteir.types import is_fp_type


def is_vector(value: ir.Value) -> bool:
    return isinstance(getattr(value, "type", None), VectorType)


def emit_int_div(
    ir_builder: ir.IRBuilder,
    lhs: ir.Value,
    rhs: ir.Value,
    unsigned: bool,
) -> ir.Instruction:
    return (
        ir_builder.udiv(lhs, rhs, name="vdivtmp")
        if unsigned
        else ir_builder.sdiv(lhs, rhs, name="vdivtmp")
    )


def emit_add(
    ir_builder: ir.IRBuilder,
    lhs: ir.Value,
    rhs: ir.Value,
    name: str = "addtmp",
) -> ir.Instruction:
    if is_fp_type(lhs.type):
        return ir_builder.fadd(lhs, rhs, name=name)
    return ir_builder.add(lhs, rhs, name=name)


def splat_scalar(
    ir_builder: ir.IRBuilder,
    scalar: ir.Value,
    vec_type: ir.VectorType,
) -> ir.Value:
    zero_i32 = ir.Constant(ir.IntType(32), 0)
    undef_vec = ir.Constant(vec_type, ir.Undefined)
    vector_zero = ir_builder.insert_element(undef_vec, scalar, zero_i32)
    mask_ty = ir.VectorType(ir.IntType(32), vec_type.count)
    mask = ir.Constant(mask_ty, [0] * vec_type.count)
    return ir_builder.shuffle_vector(vector_zero, undef_vec, mask)


__all__ = ["emit_add", "emit_int_div", "is_vector", "splat_scalar"]

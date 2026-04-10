"""
title: LLVM type helpers for llvmliteir codegen.
"""

from __future__ import annotations

from llvmlite import ir
from llvmlite.ir import DoubleType, FloatType, HalfType

try:  # FP128 may not exist depending on llvmlite build.
    from llvmlite.ir import FP128Type
except ImportError:  # pragma: no cover - optional
    FP128Type = None

from irx.typecheck import typechecked


@typechecked
def is_fp_type(type_: ir.Type) -> bool:
    """
    title: Is fp type.
    parameters:
      type_:
        type: ir.Type
    returns:
      type: bool
    """
    fp_types = [HalfType, FloatType, DoubleType]
    if FP128Type is not None:
        fp_types.append(FP128Type)
    return isinstance(type_, tuple(fp_types))


@typechecked
def is_int_type(type_: ir.Type) -> bool:
    """
    title: Is int type.
    parameters:
      type_:
        type: ir.Type
    returns:
      type: bool
    """
    return isinstance(type_, ir.IntType)


@typechecked
class VariablesLLVM:
    FLOAT_TYPE: ir.types.Type
    FLOAT16_TYPE: ir.types.Type
    DOUBLE_TYPE: ir.types.Type
    INT8_TYPE: ir.types.Type
    INT64_TYPE: ir.types.Type
    INT16_TYPE: ir.types.Type
    INT32_TYPE: ir.types.Type
    VOID_TYPE: ir.types.Type
    BOOLEAN_TYPE: ir.types.Type
    UINT8_TYPE: ir.types.Type
    UINT16_TYPE: ir.types.Type
    UINT32_TYPE: ir.types.Type
    UINT64_TYPE: ir.types.Type
    UINT128_TYPE: ir.types.Type
    ASCII_STRING_TYPE: ir.types.Type
    UTF8_STRING_TYPE: ir.types.Type
    TIME_TYPE: ir.types.Type
    TIMESTAMP_TYPE: ir.types.Type
    DATETIME_TYPE: ir.types.Type
    SIZE_T_TYPE: ir.types.Type | None
    POINTER_BITS: int
    OPAQUE_POINTER_TYPE: ir.types.Type
    BUFFER_OWNER_HANDLE_TYPE: ir.types.Type
    BUFFER_VIEW_TYPE: ir.types.Type
    ARROW_ARRAY_BUILDER_HANDLE_TYPE: ir.types.Type
    ARROW_ARRAY_HANDLE_TYPE: ir.types.Type

    context: ir.context.Context
    module: ir.module.Module
    ir_builder: ir.builder.IRBuilder

    def get_data_type(self, type_name: str) -> ir.types.Type:
        """
        title: Get data type.
        parameters:
          type_name:
            type: str
        returns:
          type: ir.types.Type
        """
        if type_name == "float32":
            return self.FLOAT_TYPE
        if type_name == "float16":
            return self.FLOAT16_TYPE
        if type_name in ("double", "float64"):
            return self.DOUBLE_TYPE
        if type_name == "boolean":
            return self.BOOLEAN_TYPE
        if type_name == "int8":
            return self.INT8_TYPE
        if type_name == "int16":
            return self.INT16_TYPE
        if type_name == "int32":
            return self.INT32_TYPE
        if type_name == "int64":
            return self.INT64_TYPE
        if type_name == "char":
            return self.INT8_TYPE
        if type_name in ("string", "stringascii"):
            return self.ASCII_STRING_TYPE
        if type_name == "utf8string":
            return self.UTF8_STRING_TYPE
        if type_name == "uint8":
            return self.UINT8_TYPE
        if type_name == "uint16":
            return self.UINT16_TYPE
        if type_name == "uint32":
            return self.UINT32_TYPE
        if type_name == "uint64":
            return self.UINT64_TYPE
        if type_name == "uint128":
            return self.UINT128_TYPE
        if type_name == "nonetype":
            return self.VOID_TYPE

        raise Exception(f"[EE]: Type name {type_name} not valid.")


__all__ = ["VariablesLLVM", "is_fp_type", "is_int_type"]

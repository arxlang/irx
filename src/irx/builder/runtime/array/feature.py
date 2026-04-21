"""
title: Canonical array runtime feature exports.
summary: >-
  Re-export the builtin array runtime surface from the Arrow-backed runtime
  implementation.
"""

from irx.builder.runtime.arrow.feature import (
    ARRAY_PRIMITIVE_TYPE_SPECS,
    ARROW_PRIMITIVE_TYPE_SPECS,
    IRX_ARROW_TYPE_BOOL,
    IRX_ARROW_TYPE_FLOAT32,
    IRX_ARROW_TYPE_FLOAT64,
    IRX_ARROW_TYPE_INT8,
    IRX_ARROW_TYPE_INT16,
    IRX_ARROW_TYPE_INT32,
    IRX_ARROW_TYPE_INT64,
    IRX_ARROW_TYPE_UINT8,
    IRX_ARROW_TYPE_UINT16,
    IRX_ARROW_TYPE_UINT32,
    IRX_ARROW_TYPE_UINT64,
    IRX_ARROW_TYPE_UNKNOWN,
    ArrayPrimitiveTypeSpec,
    ArrowPrimitiveTypeSpec,
    build_array_runtime_feature,
    build_arrow_runtime_feature,
)

__all__ = [
    "ARRAY_PRIMITIVE_TYPE_SPECS",
    "ARROW_PRIMITIVE_TYPE_SPECS",
    "IRX_ARROW_TYPE_BOOL",
    "IRX_ARROW_TYPE_FLOAT32",
    "IRX_ARROW_TYPE_FLOAT64",
    "IRX_ARROW_TYPE_INT8",
    "IRX_ARROW_TYPE_INT16",
    "IRX_ARROW_TYPE_INT32",
    "IRX_ARROW_TYPE_INT64",
    "IRX_ARROW_TYPE_UINT8",
    "IRX_ARROW_TYPE_UINT16",
    "IRX_ARROW_TYPE_UINT32",
    "IRX_ARROW_TYPE_UINT64",
    "IRX_ARROW_TYPE_UNKNOWN",
    "ArrayPrimitiveTypeSpec",
    "ArrowPrimitiveTypeSpec",
    "build_array_runtime_feature",
    "build_arrow_runtime_feature",
]

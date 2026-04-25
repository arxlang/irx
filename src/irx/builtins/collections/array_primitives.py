"""
title: Shared primitive array storage metadata.
summary: >-
  Define stable primitive type metadata shared by the builtin Arrow C++ backed
  array runtime and higher-level Tensor helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

from irx.buffer import (
    BUFFER_DTYPE_BOOL,
    BUFFER_DTYPE_FLOAT32,
    BUFFER_DTYPE_FLOAT64,
    BUFFER_DTYPE_INT8,
    BUFFER_DTYPE_INT16,
    BUFFER_DTYPE_INT32,
    BUFFER_DTYPE_INT64,
    BUFFER_DTYPE_UINT8,
    BUFFER_DTYPE_UINT16,
    BUFFER_DTYPE_UINT32,
    BUFFER_DTYPE_UINT64,
)
from irx.typecheck import typechecked

IRX_ARROW_TYPE_UNKNOWN = 0
IRX_ARROW_TYPE_INT32 = 1
IRX_ARROW_TYPE_INT8 = 2
IRX_ARROW_TYPE_INT16 = 3
IRX_ARROW_TYPE_INT64 = 4
IRX_ARROW_TYPE_UINT8 = 5
IRX_ARROW_TYPE_UINT16 = 6
IRX_ARROW_TYPE_UINT32 = 7
IRX_ARROW_TYPE_UINT64 = 8
IRX_ARROW_TYPE_FLOAT32 = 9
IRX_ARROW_TYPE_FLOAT64 = 10
IRX_ARROW_TYPE_BOOL = 11


@typechecked
@dataclass(frozen=True)
class ArrayPrimitiveTypeSpec:
    """
    title: Supported builtin array primitive storage type metadata.
    attributes:
      name:
        type: str
      type_id:
        type: int
      dtype_token:
        type: int
      element_size_bytes:
        type: int | None
      buffer_view_compatible:
        type: bool
    """

    name: str
    type_id: int
    dtype_token: int
    element_size_bytes: int | None
    buffer_view_compatible: bool


ARRAY_PRIMITIVE_TYPE_SPECS = {
    spec.name: spec
    for spec in (
        ArrayPrimitiveTypeSpec(
            "int8",
            IRX_ARROW_TYPE_INT8,
            BUFFER_DTYPE_INT8,
            1,
            True,
        ),
        ArrayPrimitiveTypeSpec(
            "int16",
            IRX_ARROW_TYPE_INT16,
            BUFFER_DTYPE_INT16,
            2,
            True,
        ),
        ArrayPrimitiveTypeSpec(
            "int32",
            IRX_ARROW_TYPE_INT32,
            BUFFER_DTYPE_INT32,
            4,
            True,
        ),
        ArrayPrimitiveTypeSpec(
            "int64",
            IRX_ARROW_TYPE_INT64,
            BUFFER_DTYPE_INT64,
            8,
            True,
        ),
        ArrayPrimitiveTypeSpec(
            "uint8",
            IRX_ARROW_TYPE_UINT8,
            BUFFER_DTYPE_UINT8,
            1,
            True,
        ),
        ArrayPrimitiveTypeSpec(
            "uint16",
            IRX_ARROW_TYPE_UINT16,
            BUFFER_DTYPE_UINT16,
            2,
            True,
        ),
        ArrayPrimitiveTypeSpec(
            "uint32",
            IRX_ARROW_TYPE_UINT32,
            BUFFER_DTYPE_UINT32,
            4,
            True,
        ),
        ArrayPrimitiveTypeSpec(
            "uint64",
            IRX_ARROW_TYPE_UINT64,
            BUFFER_DTYPE_UINT64,
            8,
            True,
        ),
        ArrayPrimitiveTypeSpec(
            "float32",
            IRX_ARROW_TYPE_FLOAT32,
            BUFFER_DTYPE_FLOAT32,
            4,
            True,
        ),
        ArrayPrimitiveTypeSpec(
            "float64",
            IRX_ARROW_TYPE_FLOAT64,
            BUFFER_DTYPE_FLOAT64,
            8,
            True,
        ),
        ArrayPrimitiveTypeSpec(
            "bool",
            IRX_ARROW_TYPE_BOOL,
            BUFFER_DTYPE_BOOL,
            None,
            False,
        ),
    )
}


__all__ = [
    "ARRAY_PRIMITIVE_TYPE_SPECS",
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
]

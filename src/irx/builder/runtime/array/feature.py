"""
title: Builtin array runtime feature declarations backed by Arrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from arx_nanoarrow_sources import get_include_dir, get_source_files
from llvmlite import ir

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
from irx.builder.runtime.features import (
    ExternalSymbolSpec,
    NativeArtifact,
    RuntimeFeature,
    declare_external_function,
)
from irx.typecheck import typechecked

if TYPE_CHECKING:
    from irx.builder.protocols import VisitorProtocol


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


@typechecked
def _build_runtime_feature(feature_name: str) -> RuntimeFeature:
    """
    title: Build one runtime feature specification.
    parameters:
      feature_name:
        type: str
    returns:
      type: RuntimeFeature
    """
    runtime_root = Path(__file__).resolve().parent
    native_root = (runtime_root.parent / "arrow" / "native").resolve()
    buffer_native_root = (runtime_root.parent / "buffer" / "native").resolve()
    compile_flags = ("-std=c99", "-DNANOARROW_NAMESPACE=IrxNanoarrow")
    nanoarrow_include_dir = get_include_dir()
    nanoarrow_sources = get_source_files()

    if not nanoarrow_sources:
        raise RuntimeError(
            "arx-nanoarrow-sources did not provide any nanoarrow C sources"
        )

    include_dirs = (native_root, buffer_native_root, nanoarrow_include_dir)
    artifacts = [
        NativeArtifact(
            kind="c_source",
            path=native_root / "irx_arrow_runtime.c",
            include_dirs=include_dirs,
            compile_flags=compile_flags,
        )
    ]
    artifacts.extend(
        NativeArtifact(
            kind="c_source",
            path=source_path,
            include_dirs=include_dirs,
            compile_flags=compile_flags,
        )
        for source_path in nanoarrow_sources
    )

    symbols = {
        "irx_arrow_schema_import_copy": ExternalSymbolSpec(
            "irx_arrow_schema_import_copy",
            _declare_schema_import_copy,
        ),
        "irx_arrow_schema_export": ExternalSymbolSpec(
            "irx_arrow_schema_export",
            _declare_schema_export,
        ),
        "irx_arrow_schema_type_id": ExternalSymbolSpec(
            "irx_arrow_schema_type_id",
            _declare_schema_type_id,
        ),
        "irx_arrow_schema_is_nullable": ExternalSymbolSpec(
            "irx_arrow_schema_is_nullable",
            _declare_schema_is_nullable,
        ),
        "irx_arrow_schema_retain": ExternalSymbolSpec(
            "irx_arrow_schema_retain",
            _declare_schema_retain,
        ),
        "irx_arrow_schema_release": ExternalSymbolSpec(
            "irx_arrow_schema_release",
            _declare_schema_release,
        ),
        "irx_arrow_array_builder_new": ExternalSymbolSpec(
            "irx_arrow_array_builder_new",
            _declare_builder_new,
        ),
        "irx_arrow_array_builder_append_null": ExternalSymbolSpec(
            "irx_arrow_array_builder_append_null",
            _declare_builder_append_null,
        ),
        "irx_arrow_array_builder_append_int": ExternalSymbolSpec(
            "irx_arrow_array_builder_append_int",
            _declare_builder_append_int,
        ),
        "irx_arrow_array_builder_append_uint": ExternalSymbolSpec(
            "irx_arrow_array_builder_append_uint",
            _declare_builder_append_uint,
        ),
        "irx_arrow_array_builder_append_double": ExternalSymbolSpec(
            "irx_arrow_array_builder_append_double",
            _declare_builder_append_double,
        ),
        "irx_arrow_array_builder_int32_new": ExternalSymbolSpec(
            "irx_arrow_array_builder_int32_new",
            _declare_builder_int32_new,
        ),
        "irx_arrow_array_builder_append_int32": ExternalSymbolSpec(
            "irx_arrow_array_builder_append_int32",
            _declare_builder_append_int32,
        ),
        "irx_arrow_array_builder_finish": ExternalSymbolSpec(
            "irx_arrow_array_builder_finish",
            _declare_builder_finish,
        ),
        "irx_arrow_array_builder_release": ExternalSymbolSpec(
            "irx_arrow_array_builder_release",
            _declare_builder_release,
        ),
        "irx_arrow_array_length": ExternalSymbolSpec(
            "irx_arrow_array_length",
            _declare_array_length,
        ),
        "irx_arrow_array_offset": ExternalSymbolSpec(
            "irx_arrow_array_offset",
            _declare_array_offset,
        ),
        "irx_arrow_array_null_count": ExternalSymbolSpec(
            "irx_arrow_array_null_count",
            _declare_array_null_count,
        ),
        "irx_arrow_array_type_id": ExternalSymbolSpec(
            "irx_arrow_array_type_id",
            _declare_array_type_id,
        ),
        "irx_arrow_array_is_nullable": ExternalSymbolSpec(
            "irx_arrow_array_is_nullable",
            _declare_array_is_nullable,
        ),
        "irx_arrow_array_has_validity_bitmap": ExternalSymbolSpec(
            "irx_arrow_array_has_validity_bitmap",
            _declare_array_has_validity_bitmap,
        ),
        "irx_arrow_array_can_borrow_buffer_view": ExternalSymbolSpec(
            "irx_arrow_array_can_borrow_buffer_view",
            _declare_array_can_borrow_buffer_view,
        ),
        "irx_arrow_array_schema_copy": ExternalSymbolSpec(
            "irx_arrow_array_schema_copy",
            _declare_array_schema_copy,
        ),
        "irx_arrow_array_export": ExternalSymbolSpec(
            "irx_arrow_array_export",
            _declare_array_export,
        ),
        "irx_arrow_array_import": ExternalSymbolSpec(
            "irx_arrow_array_import",
            _declare_array_import_copy,
        ),
        "irx_arrow_array_import_copy": ExternalSymbolSpec(
            "irx_arrow_array_import_copy",
            _declare_array_import_copy,
        ),
        "irx_arrow_array_import_move": ExternalSymbolSpec(
            "irx_arrow_array_import_move",
            _declare_array_import_move,
        ),
        "irx_arrow_array_validity_bitmap": ExternalSymbolSpec(
            "irx_arrow_array_validity_bitmap",
            _declare_array_validity_bitmap,
        ),
        "irx_arrow_array_borrow_buffer_view": ExternalSymbolSpec(
            "irx_arrow_array_borrow_buffer_view",
            _declare_array_borrow_buffer_view,
        ),
        "irx_arrow_array_retain": ExternalSymbolSpec(
            "irx_arrow_array_retain",
            _declare_array_retain,
        ),
        "irx_arrow_array_release": ExternalSymbolSpec(
            "irx_arrow_array_release",
            _declare_array_release,
        ),
        "irx_arrow_last_error": ExternalSymbolSpec(
            "irx_arrow_last_error",
            _declare_last_error,
        ),
    }

    return RuntimeFeature(
        name=feature_name,
        symbols=symbols,
        artifacts=tuple(artifacts),
        metadata={
            "type_ids": {
                name: spec.type_id
                for name, spec in ARRAY_PRIMITIVE_TYPE_SPECS.items()
            },
            "buffer_dtype_tokens": {
                name: spec.dtype_token
                for name, spec in ARRAY_PRIMITIVE_TYPE_SPECS.items()
            },
            "supported_primitive_types": {
                name: {
                    "type_id": spec.type_id,
                    "dtype_token": spec.dtype_token,
                    "element_size_bytes": spec.element_size_bytes,
                    "buffer_view_compatible": spec.buffer_view_compatible,
                }
                for name, spec in ARRAY_PRIMITIVE_TYPE_SPECS.items()
            },
            "opaque_handles": {
                "schema": "irx_arrow_schema_handle",
                "array_builder": "irx_arrow_array_builder_handle",
                "array": "irx_arrow_array_handle",
            },
            "canonical_name": "array",
            "implementation": "arrow",
        },
    )


@typechecked
def build_array_runtime_feature() -> RuntimeFeature:
    """
    title: Build the builtin array runtime feature specification.
    returns:
      type: RuntimeFeature
    """
    return _build_runtime_feature("array")


@typechecked
def _declare_function(
    visitor: VisitorProtocol,
    name: str,
    return_type: ir.Type,
    arg_types: list[ir.Type],
) -> ir.Function:
    """
    title: Declare one Arrow runtime symbol.
    parameters:
      visitor:
        type: VisitorProtocol
      name:
        type: str
      return_type:
        type: ir.Type
      arg_types:
        type: list[ir.Type]
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(return_type, arg_types)
    return declare_external_function(visitor._llvm.module, name, fn_type)


@typechecked
def _opaque_handle_type(visitor: VisitorProtocol) -> ir.Type:
    """
    title: Return one opaque runtime handle type.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Type
    """
    return visitor._llvm.OPAQUE_POINTER_TYPE


@typechecked
def _declare_schema_import_copy(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow schema import copy.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_schema_import_copy",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.OPAQUE_POINTER_TYPE,
            _opaque_handle_type(visitor).as_pointer(),
        ],
    )


@typechecked
def _declare_schema_export(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow schema export.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_schema_export",
        visitor._llvm.INT32_TYPE,
        [
            _opaque_handle_type(visitor),
            visitor._llvm.OPAQUE_POINTER_TYPE,
        ],
    )


@typechecked
def _declare_schema_type_id(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow schema type id.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_schema_type_id",
        visitor._llvm.INT32_TYPE,
        [_opaque_handle_type(visitor)],
    )


@typechecked
def _declare_schema_is_nullable(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow schema is nullable.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_schema_is_nullable",
        visitor._llvm.INT32_TYPE,
        [_opaque_handle_type(visitor)],
    )


@typechecked
def _declare_schema_retain(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow schema retain.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_schema_retain",
        visitor._llvm.INT32_TYPE,
        [_opaque_handle_type(visitor)],
    )


@typechecked
def _declare_schema_release(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow schema release.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_schema_release",
        visitor._llvm.VOID_TYPE,
        [_opaque_handle_type(visitor)],
    )


@typechecked
def _declare_builder_new(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow builder new.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_builder_new",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.INT32_TYPE,
            visitor._llvm.ARRAY_BUILDER_HANDLE_TYPE.as_pointer(),
        ],
    )


@typechecked
def _declare_builder_append_null(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow builder append null.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_builder_append_null",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARRAY_BUILDER_HANDLE_TYPE,
            visitor._llvm.INT64_TYPE,
        ],
    )


@typechecked
def _declare_builder_append_int(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow builder append int.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_builder_append_int",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARRAY_BUILDER_HANDLE_TYPE,
            visitor._llvm.INT64_TYPE,
        ],
    )


@typechecked
def _declare_builder_append_uint(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow builder append uint.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_builder_append_uint",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARRAY_BUILDER_HANDLE_TYPE,
            visitor._llvm.INT64_TYPE,
        ],
    )


@typechecked
def _declare_builder_append_double(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow builder append double.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_builder_append_double",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARRAY_BUILDER_HANDLE_TYPE,
            visitor._llvm.DOUBLE_TYPE,
        ],
    )


@typechecked
def _declare_builder_int32_new(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow builder int32 new.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_builder_int32_new",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.ARRAY_BUILDER_HANDLE_TYPE.as_pointer()],
    )


@typechecked
def _declare_builder_append_int32(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow builder append int32.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_builder_append_int32",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARRAY_BUILDER_HANDLE_TYPE,
            visitor._llvm.INT32_TYPE,
        ],
    )


@typechecked
def _declare_builder_finish(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow builder finish.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_builder_finish",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARRAY_BUILDER_HANDLE_TYPE,
            visitor._llvm.ARRAY_HANDLE_TYPE.as_pointer(),
        ],
    )


@typechecked
def _declare_builder_release(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow builder release.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_builder_release",
        visitor._llvm.VOID_TYPE,
        [visitor._llvm.ARRAY_BUILDER_HANDLE_TYPE],
    )


@typechecked
def _declare_array_length(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array length.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_length",
        visitor._llvm.INT64_TYPE,
        [visitor._llvm.ARRAY_HANDLE_TYPE],
    )


@typechecked
def _declare_array_offset(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array offset.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_offset",
        visitor._llvm.INT64_TYPE,
        [visitor._llvm.ARRAY_HANDLE_TYPE],
    )


@typechecked
def _declare_array_null_count(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array null count.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_null_count",
        visitor._llvm.INT64_TYPE,
        [visitor._llvm.ARRAY_HANDLE_TYPE],
    )


@typechecked
def _declare_array_type_id(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array type id.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_type_id",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.ARRAY_HANDLE_TYPE],
    )


@typechecked
def _declare_array_is_nullable(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array is nullable.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_is_nullable",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.ARRAY_HANDLE_TYPE],
    )


@typechecked
def _declare_array_has_validity_bitmap(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare Arrow array has validity bitmap.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_has_validity_bitmap",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.ARRAY_HANDLE_TYPE],
    )


@typechecked
def _declare_array_can_borrow_buffer_view(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare Arrow array can borrow buffer view.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_can_borrow_buffer_view",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.ARRAY_HANDLE_TYPE],
    )


@typechecked
def _declare_array_schema_copy(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array schema copy.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_schema_copy",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARRAY_HANDLE_TYPE,
            _opaque_handle_type(visitor).as_pointer(),
        ],
    )


@typechecked
def _declare_array_export(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array export.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_export",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARRAY_HANDLE_TYPE,
            visitor._llvm.OPAQUE_POINTER_TYPE,
            visitor._llvm.OPAQUE_POINTER_TYPE,
        ],
    )


@typechecked
def _declare_array_import_copy(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array import copy.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_import_copy",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.OPAQUE_POINTER_TYPE,
            visitor._llvm.OPAQUE_POINTER_TYPE,
            visitor._llvm.ARRAY_HANDLE_TYPE.as_pointer(),
        ],
    )


@typechecked
def _declare_array_import_move(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array import move.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_import_move",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.OPAQUE_POINTER_TYPE,
            visitor._llvm.OPAQUE_POINTER_TYPE,
            visitor._llvm.ARRAY_HANDLE_TYPE.as_pointer(),
        ],
    )


@typechecked
def _declare_array_validity_bitmap(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array validity bitmap.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_validity_bitmap",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARRAY_HANDLE_TYPE,
            visitor._llvm.OPAQUE_POINTER_TYPE.as_pointer(),
            visitor._llvm.INT64_TYPE.as_pointer(),
            visitor._llvm.INT64_TYPE.as_pointer(),
        ],
    )


@typechecked
def _declare_array_borrow_buffer_view(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare Arrow array borrow buffer view.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_borrow_buffer_view",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARRAY_HANDLE_TYPE,
            visitor._llvm.BUFFER_VIEW_TYPE.as_pointer(),
        ],
    )


@typechecked
def _declare_array_retain(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array retain.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_retain",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.ARRAY_HANDLE_TYPE],
    )


@typechecked
def _declare_array_release(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow array release.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_array_release",
        visitor._llvm.VOID_TYPE,
        [visitor._llvm.ARRAY_HANDLE_TYPE],
    )


@typechecked
def _declare_last_error(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow last error.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_last_error",
        visitor._llvm.OPAQUE_POINTER_TYPE,
        [],
    )


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
    "build_array_runtime_feature",
]

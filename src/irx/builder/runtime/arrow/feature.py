"""
title: Arrow runtime feature declarations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from arx_nanoarrow_sources import get_include_dir, get_source_files
from llvmlite import ir

from irx.builder.runtime.features import (
    ExternalSymbolSpec,
    NativeArtifact,
    RuntimeFeature,
    declare_external_function,
)

if TYPE_CHECKING:
    from irx.builder.protocols import VisitorProtocol

IRX_ARROW_TYPE_INT32 = 1


def build_arrow_runtime_feature() -> RuntimeFeature:
    """
    title: Build the Arrow runtime feature specification.
    returns:
      type: RuntimeFeature
    """
    runtime_root = Path(__file__).resolve().parent
    native_root = runtime_root / "native"
    compile_flags = ("-std=c99", "-DNANOARROW_NAMESPACE=IrxNanoarrow")
    nanoarrow_include_dir = get_include_dir()
    nanoarrow_sources = get_source_files()

    if not nanoarrow_sources:
        raise RuntimeError(
            "arx-nanoarrow-sources did not provide any nanoarrow C sources"
        )

    include_dirs = (native_root, nanoarrow_include_dir)
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

    return RuntimeFeature(
        name="arrow",
        symbols={
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
            "irx_arrow_array_null_count": ExternalSymbolSpec(
                "irx_arrow_array_null_count",
                _declare_array_null_count,
            ),
            "irx_arrow_array_type_id": ExternalSymbolSpec(
                "irx_arrow_array_type_id",
                _declare_array_type_id,
            ),
            "irx_arrow_array_export": ExternalSymbolSpec(
                "irx_arrow_array_export",
                _declare_array_export,
            ),
            "irx_arrow_array_import": ExternalSymbolSpec(
                "irx_arrow_array_import",
                _declare_array_import,
            ),
            "irx_arrow_array_release": ExternalSymbolSpec(
                "irx_arrow_array_release",
                _declare_array_release,
            ),
            "irx_arrow_last_error": ExternalSymbolSpec(
                "irx_arrow_last_error",
                _declare_last_error,
            ),
        },
        artifacts=tuple(artifacts),
        metadata={
            "type_ids": {"int32": IRX_ARROW_TYPE_INT32},
            "opaque_handles": {
                "array_builder": "irx_arrow_array_builder_handle",
                "array": "irx_arrow_array_handle",
            },
        },
    )


def _declare_builder_int32_new(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare builder int32 new.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.ARROW_ARRAY_BUILDER_HANDLE_TYPE.as_pointer()],
    )
    return declare_external_function(
        visitor._llvm.module,
        "irx_arrow_array_builder_int32_new",
        fn_type,
    )


def _declare_builder_append_int32(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare builder append int32.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARROW_ARRAY_BUILDER_HANDLE_TYPE,
            visitor._llvm.INT32_TYPE,
        ],
    )
    return declare_external_function(
        visitor._llvm.module,
        "irx_arrow_array_builder_append_int32",
        fn_type,
    )


def _declare_builder_finish(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare builder finish.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARROW_ARRAY_BUILDER_HANDLE_TYPE,
            visitor._llvm.ARROW_ARRAY_HANDLE_TYPE.as_pointer(),
        ],
    )
    return declare_external_function(
        visitor._llvm.module,
        "irx_arrow_array_builder_finish",
        fn_type,
    )


def _declare_builder_release(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare builder release.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.VOID_TYPE,
        [visitor._llvm.ARROW_ARRAY_BUILDER_HANDLE_TYPE],
    )
    return declare_external_function(
        visitor._llvm.module,
        "irx_arrow_array_builder_release",
        fn_type,
    )


def _declare_array_length(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare array length.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.INT64_TYPE,
        [visitor._llvm.ARROW_ARRAY_HANDLE_TYPE],
    )
    return declare_external_function(
        visitor._llvm.module,
        "irx_arrow_array_length",
        fn_type,
    )


def _declare_array_null_count(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare array null count.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.INT64_TYPE,
        [visitor._llvm.ARROW_ARRAY_HANDLE_TYPE],
    )
    return declare_external_function(
        visitor._llvm.module,
        "irx_arrow_array_null_count",
        fn_type,
    )


def _declare_array_type_id(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare array type id.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.ARROW_ARRAY_HANDLE_TYPE],
    )
    return declare_external_function(
        visitor._llvm.module,
        "irx_arrow_array_type_id",
        fn_type,
    )


def _declare_array_export(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare array export.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    opaque_ptr = visitor._llvm.OPAQUE_POINTER_TYPE
    fn_type = ir.FunctionType(
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.ARROW_ARRAY_HANDLE_TYPE,
            opaque_ptr,
            opaque_ptr,
        ],
    )
    return declare_external_function(
        visitor._llvm.module,
        "irx_arrow_array_export",
        fn_type,
    )


def _declare_array_import(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare array import.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    opaque_ptr = visitor._llvm.OPAQUE_POINTER_TYPE
    fn_type = ir.FunctionType(
        visitor._llvm.INT32_TYPE,
        [
            opaque_ptr,
            opaque_ptr,
            visitor._llvm.ARROW_ARRAY_HANDLE_TYPE.as_pointer(),
        ],
    )
    return declare_external_function(
        visitor._llvm.module,
        "irx_arrow_array_import",
        fn_type,
    )


def _declare_array_release(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare array release.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.VOID_TYPE,
        [visitor._llvm.ARROW_ARRAY_HANDLE_TYPE],
    )
    return declare_external_function(
        visitor._llvm.module,
        "irx_arrow_array_release",
        fn_type,
    )


def _declare_last_error(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare last error.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(visitor._llvm.INT8_TYPE.as_pointer(), [])
    return declare_external_function(
        visitor._llvm.module, "irx_arrow_last_error", fn_type
    )

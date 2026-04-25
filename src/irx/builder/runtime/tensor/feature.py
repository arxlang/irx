"""
title: Builtin tensor runtime feature declarations backed by Arrow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from llvmlite import ir

from irx.builder.runtime.arrowcpp import (
    arrowcpp_compile_flags,
    arrowcpp_include_dirs,
    arrowcpp_linker_flags,
    arrowcpp_runtime_metadata,
)
from irx.builder.runtime.features import (
    ExternalSymbolSpec,
    NativeArtifact,
    RuntimeFeature,
    declare_external_function,
)
from irx.builtins.collections.array_primitives import (
    ARRAY_PRIMITIVE_TYPE_SPECS,
)
from irx.typecheck import typechecked

if TYPE_CHECKING:
    from irx.builder.protocols import VisitorProtocol


@typechecked
def build_tensor_runtime_feature() -> RuntimeFeature:
    """
    title: Build the builtin tensor runtime feature specification.
    returns:
      type: RuntimeFeature
    """
    runtime_root = Path(__file__).resolve().parent
    native_root = (runtime_root.parent / "arrow" / "native").resolve()
    buffer_native_root = (runtime_root.parent / "buffer" / "native").resolve()
    compile_flags = arrowcpp_compile_flags()
    include_dirs = (
        native_root,
        buffer_native_root,
        *arrowcpp_include_dirs(),
    )
    artifacts = [
        NativeArtifact(
            kind="cxx_source",
            path=native_root / "irx_arrow_runtime.cc",
            include_dirs=include_dirs,
            compile_flags=compile_flags,
        )
    ]

    return RuntimeFeature(
        name="tensor",
        symbols={
            "irx_arrow_tensor_builder_new": ExternalSymbolSpec(
                "irx_arrow_tensor_builder_new",
                _declare_tensor_builder_new,
            ),
            "irx_arrow_tensor_builder_append_int": ExternalSymbolSpec(
                "irx_arrow_tensor_builder_append_int",
                _declare_tensor_builder_append_int,
            ),
            "irx_arrow_tensor_builder_append_uint": ExternalSymbolSpec(
                "irx_arrow_tensor_builder_append_uint",
                _declare_tensor_builder_append_uint,
            ),
            "irx_arrow_tensor_builder_append_double": ExternalSymbolSpec(
                "irx_arrow_tensor_builder_append_double",
                _declare_tensor_builder_append_double,
            ),
            "irx_arrow_tensor_builder_finish": ExternalSymbolSpec(
                "irx_arrow_tensor_builder_finish",
                _declare_tensor_builder_finish,
            ),
            "irx_arrow_tensor_builder_release": ExternalSymbolSpec(
                "irx_arrow_tensor_builder_release",
                _declare_tensor_builder_release,
            ),
            "irx_arrow_tensor_type_id": ExternalSymbolSpec(
                "irx_arrow_tensor_type_id",
                _declare_tensor_type_id,
            ),
            "irx_arrow_tensor_ndim": ExternalSymbolSpec(
                "irx_arrow_tensor_ndim",
                _declare_tensor_ndim,
            ),
            "irx_arrow_tensor_size": ExternalSymbolSpec(
                "irx_arrow_tensor_size",
                _declare_tensor_size,
            ),
            "irx_arrow_tensor_shape": ExternalSymbolSpec(
                "irx_arrow_tensor_shape",
                _declare_tensor_shape,
            ),
            "irx_arrow_tensor_strides": ExternalSymbolSpec(
                "irx_arrow_tensor_strides",
                _declare_tensor_strides,
            ),
            "irx_arrow_tensor_borrow_buffer_view": ExternalSymbolSpec(
                "irx_arrow_tensor_borrow_buffer_view",
                _declare_tensor_borrow_buffer_view,
            ),
            "irx_arrow_tensor_retain": ExternalSymbolSpec(
                "irx_arrow_tensor_retain",
                _declare_tensor_retain,
            ),
            "irx_arrow_tensor_release": ExternalSymbolSpec(
                "irx_arrow_tensor_release",
                _declare_tensor_release,
            ),
            "irx_arrow_last_error": ExternalSymbolSpec(
                "irx_arrow_last_error",
                _declare_last_error,
            ),
        },
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
                if spec.buffer_view_compatible
            },
            "opaque_handles": {
                "tensor_builder": "irx_arrow_tensor_builder_handle",
                "tensor": "irx_arrow_tensor_handle",
            },
            "canonical_name": "tensor",
            **arrowcpp_runtime_metadata(),
        },
        linker_flags=arrowcpp_linker_flags(),
    )


@typechecked
def _declare_function(
    visitor: VisitorProtocol,
    name: str,
    return_type: ir.Type,
    arg_types: list[ir.Type],
) -> ir.Function:
    """
    title: Declare one Arrow tensor runtime symbol.
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
def _declare_tensor_builder_new(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow tensor builder new.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_builder_new",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.INT32_TYPE,
            visitor._llvm.INT32_TYPE,
            visitor._llvm.INT64_TYPE.as_pointer(),
            visitor._llvm.INT64_TYPE.as_pointer(),
            visitor._llvm.TENSOR_BUILDER_HANDLE_TYPE.as_pointer(),
        ],
    )


@typechecked
def _declare_tensor_builder_append_int(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare Arrow tensor builder append int.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_builder_append_int",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.TENSOR_BUILDER_HANDLE_TYPE, visitor._llvm.INT64_TYPE],
    )


@typechecked
def _declare_tensor_builder_append_uint(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare Arrow tensor builder append uint.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_builder_append_uint",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.TENSOR_BUILDER_HANDLE_TYPE, visitor._llvm.INT64_TYPE],
    )


@typechecked
def _declare_tensor_builder_append_double(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare Arrow tensor builder append double.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_builder_append_double",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.TENSOR_BUILDER_HANDLE_TYPE, visitor._llvm.DOUBLE_TYPE],
    )


@typechecked
def _declare_tensor_builder_finish(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow tensor builder finish.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_builder_finish",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.TENSOR_BUILDER_HANDLE_TYPE,
            visitor._llvm.TENSOR_HANDLE_TYPE.as_pointer(),
        ],
    )


@typechecked
def _declare_tensor_builder_release(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow tensor builder release.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_builder_release",
        visitor._llvm.VOID_TYPE,
        [visitor._llvm.TENSOR_BUILDER_HANDLE_TYPE],
    )


@typechecked
def _declare_tensor_type_id(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow tensor type id.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_type_id",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.TENSOR_HANDLE_TYPE],
    )


@typechecked
def _declare_tensor_ndim(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow tensor ndim.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_ndim",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.TENSOR_HANDLE_TYPE],
    )


@typechecked
def _declare_tensor_size(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow tensor size.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_size",
        visitor._llvm.INT64_TYPE,
        [visitor._llvm.TENSOR_HANDLE_TYPE],
    )


@typechecked
def _declare_tensor_shape(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow tensor shape.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_shape",
        visitor._llvm.INT64_TYPE.as_pointer(),
        [visitor._llvm.TENSOR_HANDLE_TYPE],
    )


@typechecked
def _declare_tensor_strides(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow tensor strides.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_strides",
        visitor._llvm.INT64_TYPE.as_pointer(),
        [visitor._llvm.TENSOR_HANDLE_TYPE],
    )


@typechecked
def _declare_tensor_borrow_buffer_view(
    visitor: VisitorProtocol,
) -> ir.Function:
    """
    title: Declare Arrow tensor borrow buffer view.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_borrow_buffer_view",
        visitor._llvm.INT32_TYPE,
        [
            visitor._llvm.TENSOR_HANDLE_TYPE,
            visitor._llvm.BUFFER_VIEW_TYPE.as_pointer(),
        ],
    )


@typechecked
def _declare_tensor_retain(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow tensor retain.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_retain",
        visitor._llvm.INT32_TYPE,
        [visitor._llvm.TENSOR_HANDLE_TYPE],
    )


@typechecked
def _declare_tensor_release(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare Arrow tensor release.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    return _declare_function(
        visitor,
        "irx_arrow_tensor_release",
        visitor._llvm.VOID_TYPE,
        [visitor._llvm.TENSOR_HANDLE_TYPE],
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


__all__ = ["build_tensor_runtime_feature"]

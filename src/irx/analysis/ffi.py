"""
title: Public FFI semantic helpers.
summary: >-
  Normalize runtime-feature metadata and enforce the narrow public FFI type
  policy for explicit extern declarations.
"""

from __future__ import annotations

import importlib

from typing import Iterable, cast

from irx import astx
from irx.analysis.context import SemanticContext
from irx.analysis.resolved_nodes import (
    FFIAdmissibility,
    FFICallableInfo,
    FFILinkStrategy,
    FFITypeClass,
    FFITypeInfo,
    FunctionSignature,
    SemanticStruct,
)
from irx.analysis.types import (
    display_type_name,
    is_boolean_type,
    is_float_type,
    is_integer_type,
    is_none_type,
    is_string_type,
)
from irx.diagnostics import DiagnosticBag, DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
def _runtime_feature_names() -> tuple[str, ...]:
    """
    title: Return the registered runtime feature names.
    returns:
      type: tuple[str, Ellipsis]
    """
    registry_module = importlib.import_module("irx.builder.runtime.registry")
    registry_factory = getattr(
        registry_module,
        "get_default_runtime_feature_registry",
    )
    return cast(tuple[str, ...], registry_factory().names())


@typechecked
def normalize_runtime_features(
    prototype: astx.FunctionPrototype,
    *,
    diagnostics: DiagnosticBag,
) -> tuple[str, ...]:
    """
    title: Normalize runtime-feature metadata from one prototype.
    parameters:
      prototype:
        type: astx.FunctionPrototype
      diagnostics:
        type: DiagnosticBag
    returns:
      type: tuple[str, Ellipsis]
    """
    raw_feature = getattr(prototype, "runtime_feature", None)
    raw_features = getattr(prototype, "runtime_features", None)

    if raw_feature is not None and raw_features is not None:
        diagnostics.add(
            f"Extern function '{prototype.name}' may use either "
            "'runtime_feature' or 'runtime_features', not both",
            node=prototype,
            code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
        )
        return ()

    if raw_features is None:
        values: Iterable[object] = (
            () if raw_feature is None else (raw_feature,)
        )
    elif isinstance(raw_features, str):
        values = (raw_features,)
    else:
        try:
            values = tuple(raw_features)
        except TypeError:
            diagnostics.add(
                f"Extern function '{prototype.name}' has an invalid "
                "runtime_features value",
                node=prototype,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
            return ()

    normalized: list[str] = []
    seen: set[str] = set()
    known_features = set(_runtime_feature_names())
    for value in values:
        if not isinstance(value, str) or not value.strip():
            diagnostics.add(
                f"Extern function '{prototype.name}' has an invalid runtime "
                "feature name",
                node=prototype,
                code=DiagnosticCodes.RUNTIME_FEATURE_UNKNOWN,
            )
            continue
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if value not in known_features:
            diagnostics.add(
                f"Extern function '{prototype.name}' requires unknown runtime "
                f"feature '{value}'",
                node=prototype,
                code=DiagnosticCodes.RUNTIME_FEATURE_UNKNOWN,
                notes=(
                    (
                        "known runtime features: "
                        f"{', '.join(sorted(known_features))}"
                    ),
                )
                if known_features
                else (),
            )

    return tuple(normalized)


@typechecked
def _display_type_name(type_: astx.DataType) -> str:
    """
    title: Return one stable public FFI type name.
    parameters:
      type_:
        type: astx.DataType
    returns:
      type: str
    """
    return display_type_name(type_)


@typechecked
def build_ffi_callable_info(
    context: SemanticContext,
    *,
    signature: FunctionSignature,
    prototype: astx.FunctionPrototype,
) -> FFICallableInfo | None:
    """
    title: Build validated public FFI metadata for one extern signature.
    parameters:
      context:
        type: SemanticContext
      signature:
        type: FunctionSignature
      prototype:
        type: astx.FunctionPrototype
    returns:
      type: FFICallableInfo | None
    """
    start_count = len(context.diagnostics.diagnostics)
    parameter_types: list[FFITypeInfo] = []
    for parameter in signature.parameters:
        info = _classify_ffi_type(
            context,
            type_=parameter.type_,
            prototype=prototype,
            position=f"parameter '{parameter.name}'",
            allow_void=False,
            struct_stack=(),
            referenced_via_pointer=False,
        )
        if info is not None:
            parameter_types.append(info)

    return_type = _classify_ffi_type(
        context,
        type_=signature.return_type,
        prototype=prototype,
        position="return type",
        allow_void=True,
        struct_stack=(),
        referenced_via_pointer=False,
    )

    if (
        len(context.diagnostics.diagnostics) != start_count
        or return_type is None
    ):
        return None

    link_strategy = (
        FFILinkStrategy.RUNTIME_FEATURES
        if signature.required_runtime_features
        else FFILinkStrategy.SYSTEM_LINKER
    )
    return FFICallableInfo(
        admissibility=FFIAdmissibility.PUBLIC,
        parameters=tuple(parameter_types),
        return_type=return_type,
        required_runtime_features=signature.required_runtime_features,
        link_strategy=link_strategy,
    )


@typechecked
def _classify_ffi_type(
    context: SemanticContext,
    *,
    type_: astx.DataType,
    prototype: astx.FunctionPrototype,
    position: str,
    allow_void: bool,
    struct_stack: tuple[str, ...],
    referenced_via_pointer: bool,
) -> FFITypeInfo | None:
    """
    title: Classify one declared FFI type.
    parameters:
      context:
        type: SemanticContext
      type_:
        type: astx.DataType
      prototype:
        type: astx.FunctionPrototype
      position:
        type: str
      allow_void:
        type: bool
      struct_stack:
        type: tuple[str, Ellipsis]
      referenced_via_pointer:
        type: bool
    returns:
      type: FFITypeInfo | None
    """
    if is_none_type(type_):
        if allow_void:
            return FFITypeInfo(FFITypeClass.VOID, _display_type_name(type_))
        context.diagnostics.add(
            f"extern '{prototype.name}' is not FFI-safe: {position} cannot "
            "use NoneType",
            node=prototype,
            code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
        )
        return None

    if is_integer_type(type_):
        return FFITypeInfo(FFITypeClass.INTEGER, _display_type_name(type_))
    if is_float_type(type_):
        return FFITypeInfo(FFITypeClass.FLOAT, _display_type_name(type_))
    if is_boolean_type(type_):
        return FFITypeInfo(FFITypeClass.BOOLEAN, _display_type_name(type_))
    if is_string_type(type_):
        return FFITypeInfo(
            FFITypeClass.STRING,
            _display_type_name(type_),
            metadata={"abi": "pointer"},
        )
    if isinstance(type_, astx.BufferOwnerType | astx.OpaqueHandleType):
        return FFITypeInfo(
            FFITypeClass.OPAQUE_HANDLE,
            _display_type_name(type_),
            metadata={"handle_name": type_.handle_name},
        )
    if isinstance(type_, astx.PointerType):
        if type_.pointee_type is None:
            return FFITypeInfo(
                FFITypeClass.POINTER,
                _display_type_name(type_),
                metadata={"pointee": "opaque"},
            )
        if is_none_type(type_.pointee_type):
            context.diagnostics.add(
                f"extern '{prototype.name}' is not FFI-safe: {position} uses "
                "PointerType[NoneType]; use PointerType() for an opaque "
                "pointer",
                node=prototype,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
            return None
        pointee_info = _classify_ffi_type(
            context,
            type_=type_.pointee_type,
            prototype=prototype,
            position=f"{position} pointee",
            allow_void=False,
            struct_stack=struct_stack,
            referenced_via_pointer=True,
        )
        if pointee_info is None:
            return None
        return FFITypeInfo(
            FFITypeClass.POINTER,
            _display_type_name(type_),
            metadata={"pointee": pointee_info.display_name},
        )
    if isinstance(type_, astx.ClassType):
        context.diagnostics.add(
            (
                f"extern '{prototype.name}' is not FFI-safe: {position} "
                f"uses class type '{_display_type_name(type_)}'; class "
                "ABI is internal to IRx in this phase"
            ),
            node=prototype,
            code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
        )
        return None
    if isinstance(type_, astx.BufferViewType):
        return FFITypeInfo(
            FFITypeClass.STRUCT,
            _display_type_name(type_),
            metadata={"abi_name": "irx_buffer_view"},
        )
    if isinstance(type_, astx.TensorType):
        return FFITypeInfo(
            FFITypeClass.STRUCT,
            _display_type_name(type_),
            metadata={
                "abi_name": "irx_buffer_view",
                "logical_type": "tensor",
            },
        )
    if isinstance(type_, astx.StructType):
        return _classify_ffi_struct(
            context,
            struct_type=type_,
            prototype=prototype,
            position=position,
            struct_stack=struct_stack,
            referenced_via_pointer=referenced_via_pointer,
        )

    context.diagnostics.add(
        f"extern '{prototype.name}' is not FFI-safe: {position} uses "
        f"unsupported FFI type '{_display_type_name(type_)}'",
        node=prototype,
        code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
    )
    return None


@typechecked
def _classify_ffi_struct(
    context: SemanticContext,
    *,
    struct_type: astx.StructType,
    prototype: astx.FunctionPrototype,
    position: str,
    struct_stack: tuple[str, ...],
    referenced_via_pointer: bool,
) -> FFITypeInfo | None:
    """
    title: Validate one ABI-safe struct for FFI use.
    parameters:
      context:
        type: SemanticContext
      struct_type:
        type: astx.StructType
      prototype:
        type: astx.FunctionPrototype
      position:
        type: str
      struct_stack:
        type: tuple[str, Ellipsis]
      referenced_via_pointer:
        type: bool
    returns:
      type: FFITypeInfo | None
    """
    struct = _resolve_struct(context, struct_type)
    if struct is None:
        context.diagnostics.add(
            f"extern '{prototype.name}' is not FFI-safe: {position} uses "
            f"unresolved struct type '{struct_type.name}'",
            node=prototype,
            code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
        )
        return None

    struct_id = struct.qualified_name or struct.name
    if struct_id in struct_stack and referenced_via_pointer:
        return FFITypeInfo(
            FFITypeClass.STRUCT,
            struct_id,
            metadata={"qualified_name": struct_id},
        )

    next_stack = (*struct_stack, struct_id)
    for field in struct.fields:
        info = _classify_ffi_type(
            context,
            type_=field.type_,
            prototype=prototype,
            position=f"{position} field '{field.name}'",
            allow_void=False,
            struct_stack=next_stack,
            referenced_via_pointer=False,
        )
        if info is None:
            return None

    return FFITypeInfo(
        FFITypeClass.STRUCT,
        struct_id,
        metadata={
            "qualified_name": struct_id,
            "field_order": tuple(field.name for field in struct.fields),
        },
    )


@typechecked
def _resolve_struct(
    context: SemanticContext,
    struct_type: astx.StructType,
) -> SemanticStruct | None:
    """
    title: Resolve one struct reference for FFI validation.
    parameters:
      context:
        type: SemanticContext
      struct_type:
        type: astx.StructType
    returns:
      type: SemanticStruct | None
    """
    if struct_type.module_key is None:
        return None
    lookup_name = struct_type.resolved_name or struct_type.name
    return context.get_struct(struct_type.module_key, lookup_name)


__all__ = ["build_ffi_callable_info", "normalize_runtime_features"]

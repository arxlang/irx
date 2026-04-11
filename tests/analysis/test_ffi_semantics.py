"""
title: Tests for the public FFI semantic contract.
"""

from __future__ import annotations

from typing import cast

import irx.builder.runtime.registry as runtime_registry_module
import pytest

from irx import astx
from irx.analysis import SemanticError, analyze
from irx.analysis.resolved_nodes import (
    FFILinkStrategy,
    FFITypeClass,
    SemanticInfo,
)
from irx.builder.runtime.features import RuntimeFeature
from irx.builder.runtime.registry import RuntimeFeatureRegistry


def _semantic(node: astx.AST) -> SemanticInfo:
    """
    title: Return one node's semantic info.
    parameters:
      node:
        type: astx.AST
    returns:
      type: SemanticInfo
    """
    return cast(SemanticInfo, getattr(node, "semantic"))


def _extern_prototype(
    name: str,
    *args: astx.Argument,
    return_type: astx.DataType,
    symbol_name: str | None = None,
    runtime_feature: str | None = None,
) -> astx.FunctionPrototype:
    """
    title: Build one explicit extern prototype.
    parameters:
      name:
        type: str
      return_type:
        type: astx.DataType
      symbol_name:
        type: str | None
      runtime_feature:
        type: str | None
      args:
        type: astx.Argument
        variadic: positional
    returns:
      type: astx.FunctionPrototype
    """
    prototype = astx.FunctionPrototype(
        name,
        args=astx.Arguments(*args),
        return_type=return_type,
    )
    prototype.is_extern = True
    prototype.calling_convention = "c"
    prototype.symbol_name = symbol_name or name
    if runtime_feature is not None:
        prototype.runtime_feature = runtime_feature
    return prototype


def test_analyze_attaches_ffi_metadata_for_feature_backed_extern() -> None:
    """
    title: Extern signatures should expose validated public FFI metadata.
    """
    prototype = _extern_prototype(
        "sqrt",
        astx.Argument("value", astx.Float64()),
        return_type=astx.Float64(),
        runtime_feature="libm",
    )
    module = astx.Module()
    module.block.append(prototype)

    analyze(module)

    resolved_function = _semantic(prototype).resolved_function

    assert resolved_function is not None
    assert resolved_function.signature.required_runtime_features == ("libm",)
    assert resolved_function.signature.ffi is not None
    assert (
        resolved_function.signature.ffi.link_strategy
        is FFILinkStrategy.RUNTIME_FEATURES
    )
    assert (
        resolved_function.signature.ffi.parameters[0].classification
        is FFITypeClass.FLOAT
    )
    assert (
        resolved_function.signature.ffi.return_type.classification
        is FFITypeClass.FLOAT
    )


def test_runtime_feature_validation_sees_late_registry_registrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    title: Runtime feature validation should observe later registry updates.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
    """
    registry = RuntimeFeatureRegistry()
    monkeypatch.setattr(
        runtime_registry_module,
        "get_default_runtime_feature_registry",
        lambda: registry,
    )

    first_prototype = _extern_prototype(
        "late_bound",
        return_type=astx.Int32(),
        runtime_feature="late_feature",
    )
    first_module = astx.Module()
    first_module.block.append(first_prototype)

    with pytest.raises(
        SemanticError,
        match="unknown runtime feature 'late_feature'",
    ):
        analyze(first_module)

    registry.register(RuntimeFeature(name="late_feature"))

    second_prototype = _extern_prototype(
        "late_bound",
        return_type=astx.Int32(),
        runtime_feature="late_feature",
    )
    second_module = astx.Module()
    second_module.block.append(second_prototype)

    analyze(second_module)

    resolved_function = _semantic(second_prototype).resolved_function

    assert resolved_function is not None
    assert resolved_function.signature.required_runtime_features == (
        "late_feature",
    )


def test_analyze_accepts_pointer_and_opaque_handle_ffi_types() -> None:
    """
    title: Pointer and opaque-handle extern types should be admitted.
    """
    alloc = _extern_prototype(
        "alloc_values",
        return_type=astx.PointerType(astx.Float64()),
    )
    use = _extern_prototype(
        "consume_handle",
        astx.Argument("handle", astx.OpaqueHandleType("demo_handle")),
        return_type=astx.Int32(),
    )
    module = astx.Module()
    module.block.append(alloc)
    module.block.append(use)

    analyze(module)

    alloc_function = _semantic(alloc).resolved_function
    use_function = _semantic(use).resolved_function

    assert alloc_function is not None
    assert use_function is not None

    alloc_signature = alloc_function.signature
    use_signature = use_function.signature

    assert alloc_signature.ffi is not None
    assert (
        alloc_signature.ffi.return_type.classification is FFITypeClass.POINTER
    )
    assert use_signature.ffi is not None
    assert (
        use_signature.ffi.parameters[0].classification
        is FFITypeClass.OPAQUE_HANDLE
    )


def test_analyze_accepts_ffi_safe_struct_by_value() -> None:
    """
    title: ABI-safe structs should be accepted in extern signatures.
    """
    point = astx.StructDefStmt(
        name="Point",
        attributes=[
            astx.VariableDeclaration(name="x", type_=astx.Float64()),
            astx.VariableDeclaration(name="y", type_=astx.Float64()),
        ],
    )
    prototype = _extern_prototype(
        "point_norm",
        astx.Argument("point", astx.StructType("Point")),
        return_type=astx.Float64(),
    )
    module = astx.Module()
    module.block.append(point)
    module.block.append(prototype)

    analyze(module)

    resolved_function = _semantic(prototype).resolved_function

    assert resolved_function is not None

    signature = resolved_function.signature

    assert signature.ffi is not None
    assert signature.ffi.parameters[0].classification is FFITypeClass.STRUCT


def test_analyze_rejects_runtime_feature_on_irx_defined_function() -> None:
    """
    title: Runtime features belong only on explicit extern declarations.
    """
    prototype = astx.FunctionPrototype(
        "helper",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    prototype.runtime_feature = "libm"
    body = astx.Block()
    body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=prototype,
            body=body,
        )
    )

    with pytest.raises(
        SemanticError,
        match="may declare runtime features only when declared extern",
    ):
        analyze(module)


def test_analyze_rejects_pointer_to_unsupported_ffi_type() -> None:
    """
    title: Pointer pointees must use the public FFI type subset.
    """
    prototype = _extern_prototype(
        "consume_timestamp",
        astx.Argument(
            "value",
            astx.PointerType(astx.DateTime()),
        ),
        return_type=astx.Int32(),
    )
    module = astx.Module()
    module.block.append(prototype)

    with pytest.raises(SemanticError, match="unsupported FFI type 'DateTime'"):
        analyze(module)


def test_analyze_rejects_struct_with_non_ffi_field_in_extern_signature() -> (
    None
):
    """
    title: FFI structs should reject fields outside the ABI-safe subset.
    """
    bad_struct = astx.StructDefStmt(
        name="BadRecord",
        attributes=[
            astx.VariableDeclaration(name="when", type_=astx.DateTime()),
        ],
    )
    prototype = _extern_prototype(
        "consume_bad",
        astx.Argument("value", astx.StructType("BadRecord")),
        return_type=astx.Int32(),
    )
    module = astx.Module()
    module.block.append(bad_struct)
    module.block.append(prototype)

    with pytest.raises(
        SemanticError,
        match="field 'when' uses unsupported FFI type 'DateTime'",
    ):
        analyze(module)


def test_ffi_diagnostic_includes_extern_context_and_code() -> None:
    """
    title: FFI diagnostics should render stable code and field context.
    """
    bad_struct = astx.StructDefStmt(
        name="BadRecord",
        attributes=[
            astx.VariableDeclaration(name="when", type_=astx.DateTime()),
        ],
    )
    prototype = _extern_prototype(
        "consume_bad",
        astx.Argument("value", astx.StructType("BadRecord")),
        return_type=astx.Int32(),
    )
    module = astx.Module()
    module.block.append(bad_struct)
    module.block.append(prototype)

    with pytest.raises(SemanticError) as exc_info:
        analyze(module)

    formatted = str(exc_info.value)

    assert "IRX-F001" in formatted
    assert "extern 'consume_bad' is not FFI-safe" in formatted
    assert "field 'when' uses unsupported FFI type 'DateTime'" in formatted


def test_analyze_rejects_incompatible_symbol_alias_redeclarations() -> None:
    """
    title: Duplicate extern symbols should reject incompatible ABI meaning.
    """
    first = _extern_prototype(
        "use_native_value",
        astx.Argument("value", astx.Int32()),
        return_type=astx.Int32(),
        symbol_name="native_value",
    )
    second = _extern_prototype(
        "load_native_value",
        astx.Argument("value", astx.Int32()),
        return_type=astx.Float64(),
        symbol_name="native_value",
    )
    module = astx.Module()
    module.block.append(first)
    module.block.append(second)

    with pytest.raises(
        SemanticError,
        match=r"Extern symbol 'native_value'.*return_type differs",
    ):
        analyze(module)


def test_analyze_rejects_field_access_on_opaque_handle() -> None:
    """
    title: Opaque handles are pass-through values, not visible structs.
    """
    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(
        astx.VariableDeclaration(
            name="handle",
            type_=astx.OpaqueHandleType("demo_handle"),
            mutability=astx.MutabilityKind.mutable,
        )
    )
    main.body.append(
        astx.FunctionReturn(
            astx.FieldAccess(astx.Identifier("handle"), "value")
        )
    )
    module = astx.Module()
    module.block.append(main)

    with pytest.raises(
        SemanticError, match="field access requires a struct value"
    ):
        analyze(module)

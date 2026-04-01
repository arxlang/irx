"""
title: Tests for the runtime feature registry and activation state.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import astx
import pytest

from irx.builders.llvmliteir import Builder, Visitor
from irx.runtime.features import (
    ExternalSymbolSpec,
    NativeArtifact,
    RuntimeFeature,
    declare_external_function,
)
from irx.runtime.registry import RuntimeFeatureRegistry, RuntimeFeatureState
from irx.system import PrintExpr
from llvmlite import ir

if TYPE_CHECKING:
    from irx.builders.llvmliteir.protocols import VisitorProtocol


def _declare_dummy_symbol(visitor: "VisitorProtocol") -> ir.Function:
    fn_type = ir.FunctionType(visitor._llvm.INT32_TYPE, [])
    return declare_external_function(visitor._llvm.module, "dummy_rt", fn_type)


def _main_return_zero_module() -> astx.Module:
    module = astx.Module()
    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    body = astx.Block()
    body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))
    return module


def test_runtime_feature_registry_rejects_duplicate_names() -> None:
    """
    title: Runtime feature registry should reject duplicate feature names.
    """
    registry = RuntimeFeatureRegistry()
    feature = RuntimeFeature(name="dummy")

    registry.register(feature)

    with pytest.raises(ValueError, match="already exists"):
        registry.register(feature)


def test_runtime_feature_state_reuses_symbol_declarations() -> None:
    """
    title: >-
      Runtime feature state should reuse one external declaration per module.
    """
    registry = RuntimeFeatureRegistry()
    registry.register(
        RuntimeFeature(
            name="dummy",
            symbols={
                "dummy_rt": ExternalSymbolSpec(
                    "dummy_rt",
                    _declare_dummy_symbol,
                )
            },
        )
    )
    visitor = Visitor()
    state = RuntimeFeatureState(visitor, registry)

    fn_first = state.require_symbol("dummy", "dummy_rt")
    fn_second = state.require_symbol("dummy", "dummy_rt")

    assert fn_first is fn_second
    assert state.active_feature_names() == ("dummy",)


def test_runtime_feature_state_collects_only_active_artifacts() -> None:
    """
    title: Native artifact resolution should include only active features.
    """
    registry = RuntimeFeatureRegistry()
    registry.register(
        RuntimeFeature(
            name="feature_a",
            artifacts=(
                NativeArtifact("c_source", Path("/tmp/a.c")),
                NativeArtifact("object", Path("/tmp/a.o")),
            ),
        )
    )
    registry.register(
        RuntimeFeature(
            name="feature_b",
            artifacts=(NativeArtifact("c_source", Path("/tmp/b.c")),),
        )
    )
    visitor = Visitor()
    state = RuntimeFeatureState(
        visitor,
        registry,
        active_features={"feature_b"},
    )

    artifacts = state.native_artifacts()

    assert [artifact.path for artifact in artifacts] == [Path("/tmp/b.c")]


def test_print_expr_uses_libc_feature_without_arrow() -> None:
    """
    title: PrintExpr should activate libc without pulling in Arrow.
    """
    builder = Builder()
    module = astx.Module()

    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    body = astx.Block()
    body.append(PrintExpr(astx.LiteralInt32(7)))
    body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))

    ir_text = builder.translate(module)

    active_features = (
        builder.translator.runtime_features.active_feature_names()
    )

    assert "libc" in active_features
    assert "arrow" not in active_features
    assert '@"puts"' in ir_text
    assert '@"snprintf"' in ir_text


def test_simple_module_has_no_native_runtime_artifacts() -> None:
    """
    title: A simple module should not request any native runtime artifacts.
    """
    builder = Builder()

    builder.translate(_main_return_zero_module())

    assert builder.translator.runtime_features.native_artifacts() == ()

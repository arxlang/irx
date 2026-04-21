"""
title: Tests for the runtime feature registry and activation state.
"""

from __future__ import annotations

import subprocess

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from irx import astx
from irx.builder import Builder, Visitor
from irx.builder.runtime.features import (
    ExternalSymbolSpec,
    NativeArtifact,
    RuntimeFeature,
    declare_external_function,
)
from irx.builder.runtime.linking import (
    compile_native_artifacts,
    link_executable,
)
from irx.builder.runtime.registry import (
    RuntimeFeatureRegistry,
    RuntimeFeatureState,
)
from irx.diagnostics import (
    LinkingError,
    NativeCompileError,
    RuntimeFeatureError,
)
from irx.system import PrintExpr
from llvmlite import ir

if TYPE_CHECKING:
    from irx.builder.protocols import VisitorProtocol


def _declare_dummy_symbol(visitor: "VisitorProtocol") -> ir.Function:
    """
    title: Declare dummy symbol.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(visitor._llvm.INT32_TYPE, [])
    return declare_external_function(visitor._llvm.module, "dummy_rt", fn_type)


def _main_return_zero_module() -> astx.Module:
    """
    title: Main return zero module.
    returns:
      type: astx.Module
    """
    module = astx.Module()
    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    body = astx.Block()
    body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))
    return module


def _extern_prototype(
    name: str,
    *args: astx.Argument,
    return_type: astx.DataType,
    runtime_feature: str | None = None,
) -> astx.FunctionPrototype:
    """
    title: Build one explicit extern prototype.
    parameters:
      name:
        type: str
      return_type:
        type: astx.DataType
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
    prototype.symbol_name = name
    if runtime_feature is not None:
        prototype.runtime_feature = runtime_feature
    return prototype


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


def test_print_expr_uses_libc_feature_without_array_runtime() -> None:
    """
    title: PrintExpr should activate libc without pulling in the array runtime.
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
    assert "array" not in active_features
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


def test_feature_backed_extern_collects_linker_flags() -> None:
    """
    title: >-
      Feature-backed externs should activate linker flags without artifacts.
    """
    builder = Builder()
    module = astx.Module()
    module.block.append(
        _extern_prototype(
            "sqrt",
            astx.Argument("value", astx.Float64()),
            return_type=astx.Float64(),
            runtime_feature="libm",
        )
    )
    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    body = astx.Block()
    body.append(
        astx.FunctionReturn(
            astx.Cast(
                astx.FunctionCall("sqrt", [astx.LiteralFloat64(9.0)]),
                astx.Int32(),
            )
        )
    )
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))

    builder.translate(module)

    assert builder.translator.runtime_features.active_feature_names() == (
        "libm",
    )
    assert builder.translator.runtime_features.linker_flags() == ("-lm",)
    assert builder.translator.runtime_features.native_artifacts() == ()


def test_runtime_feature_state_reports_unknown_feature_structurally() -> None:
    """
    title: Unknown runtime features should raise structured runtime errors.
    """
    state = RuntimeFeatureState(Visitor(), RuntimeFeatureRegistry())

    with pytest.raises(RuntimeFeatureError) as exc_info:
        state.activate("missing")

    formatted = str(exc_info.value)

    assert "IRX-R001" in formatted
    assert "runtime feature 'missing' is not registered" in formatted


def test_runtime_feature_state_reports_missing_symbol_structurally() -> None:
    """
    title: Missing runtime symbols should raise structured runtime errors.
    """
    registry = RuntimeFeatureRegistry()
    registry.register(RuntimeFeature(name="dummy"))
    state = RuntimeFeatureState(Visitor(), registry)

    with pytest.raises(RuntimeFeatureError) as exc_info:
        state.require_symbol("dummy", "dummy_symbol")

    formatted = str(exc_info.value)

    assert "IRX-R002" in formatted
    assert (
        "runtime feature 'dummy' does not declare symbol 'dummy_symbol'"
        in formatted
    )


def test_compile_native_artifact_failure_surfaces_compile_phase_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    title: Native compile failures should preserve command and phase context.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
      tmp_path:
        type: Path
    """
    artifact = NativeArtifact("c_source", tmp_path / "runtime.c")

    def _fail_compile(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        """
        title: Return a failing compiler result.
        parameters:
          args:
            type: object
            variadic: positional
          kwargs:
            type: object
            variadic: keyword
        returns:
          type: subprocess.CompletedProcess[str]
        """
        command = list(cast(Sequence[str], args[0])) if args else []
        _ = kwargs
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="fatal error: buffer_runtime.h not found",
        )

    monkeypatch.setattr(
        "irx.builder.runtime.linking.subprocess.run", _fail_compile
    )

    with pytest.raises(NativeCompileError) as exc_info:
        compile_native_artifacts([artifact], tmp_path)

    formatted = str(exc_info.value)

    assert "IRX-C001" in formatted
    assert "native-compile" in formatted
    assert str(artifact.path) in formatted
    assert "command:" in formatted
    assert "stderr: fatal error: buffer_runtime.h not found" in formatted


def test_link_failure_surfaces_link_phase_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    title: Link failures should preserve command and output context.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
      tmp_path:
        type: Path
    """
    primary_object = tmp_path / "main.o"

    def _fail_link(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        """
        title: Return a failing linker result.
        parameters:
          args:
            type: object
            variadic: positional
          kwargs:
            type: object
            variadic: keyword
        returns:
          type: subprocess.CompletedProcess[str]
        """
        command = list(cast(Sequence[str], args[0])) if args else []
        _ = kwargs
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="undefined reference to `sqrt'",
        )

    monkeypatch.setattr(
        "irx.builder.runtime.linking.subprocess.run", _fail_link
    )

    with pytest.raises(LinkingError) as exc_info:
        link_executable(primary_object, tmp_path / "demo", ())

    formatted = str(exc_info.value)

    assert "IRX-K001" in formatted
    assert "link failed while producing 'demo'" in formatted
    assert "command:" in formatted
    assert "stderr: undefined reference to `sqrt'" in formatted

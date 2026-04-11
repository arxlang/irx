"""
title: Integration tests for the semantic-analysis pipeline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import irx.builder as builder_api
import irx.builder.backend as builder_backend
import pytest

from irx import astx
from irx.analysis import SemanticError
from irx.builder import (
    Builder,
    VariablesLLVM,
    Visitor,
    emit_int_div,
    is_fp_type,
    safe_pop,
    splat_scalar,
)
from irx.diagnostics import (
    Diagnostic,
    DiagnosticCodes,
    LinkingError,
    LoweringError,
)
from llvmlite import ir


def _main_module(*nodes: astx.AST) -> astx.Module:
    """
    title: Main module.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    module = astx.Module()
    proto = astx.FunctionPrototype(
        "main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    body = astx.Block()
    for node in nodes:
        body.append(node)
    module.block.append(astx.FunctionDef(prototype=proto, body=body))
    return module


def test_builder_translate_runs_analysis_before_codegen() -> None:
    """
    title: Test builder translate runs analysis before codegen.
    """
    builder = Builder()
    module = _main_module(
        astx.BreakStmt(),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    with pytest.raises(SemanticError, match="Break statement outside loop"):
        builder.translate(module)


def test_direct_visitor_translate_runs_analysis_before_codegen() -> None:
    """
    title: Test direct visitor translate runs analysis before codegen.
    """
    visitor = Visitor()
    module = _main_module(astx.FunctionReturn(astx.Identifier("missing")))

    with pytest.raises(SemanticError, match="cannot resolve name"):
        visitor.translate(module)


def test_semantic_failures_stop_before_lowering_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    title: Semantic failures do not enter lowering dispatch.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
    """
    visitor = Visitor()

    def _unexpected_lowering(node: astx.AST) -> None:
        """
        title: Fail if lowering dispatch is reached.
        parameters:
          node:
            type: astx.AST
        """
        raise AssertionError(f"Lowering should not run for {node!r}")

    monkeypatch.setattr(visitor, "visit", _unexpected_lowering)

    with pytest.raises(SemanticError, match="cannot resolve name"):
        visitor.translate(astx.Identifier("missing"))


def test_valid_modules_still_emit_ir_after_analysis() -> None:
    """
    title: Test valid modules still emit ir after analysis.
    """
    builder = Builder()
    module = _main_module(astx.FunctionReturn(astx.LiteralInt32(0)))

    ir_text = builder.translate(module)

    assert 'define i32 @"main"()' in ir_text
    assert "ret i32 0" in ir_text


def test_build_surfaces_linking_failures_after_semantic_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    title: Linking failures remain outside semantic analysis.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
      tmp_path:
        type: Path
    """
    builder = Builder()
    module = _main_module(astx.FunctionReturn(astx.LiteralInt32(0)))

    def _fail_link(*args: object, **kwargs: object) -> None:
        """
        title: Fail the linking step with a runtime-style error.
        parameters:
          args:
            type: object
            variadic: positional
          kwargs:
            type: object
            variadic: keyword
        """
        _ = args
        _ = kwargs
        raise LinkingError(
            Diagnostic(
                message="link failed while producing 'main'",
                code=DiagnosticCodes.LINK_FAILED,
                phase="link",
            )
        )

    monkeypatch.setattr(builder_backend, "link_executable", _fail_link)

    with pytest.raises(LinkingError, match="IRX-K001"):
        builder.build(module, output_file=str(tmp_path / "main"))


def test_lowering_missing_semantic_metadata_is_structured() -> None:
    """
    title: Direct lowering should raise structured errors for missing sidecars.
    """
    visitor = Visitor()
    call = astx.FunctionCall("helper", [])
    call.loc = SimpleNamespace(line=5, col=2)

    with pytest.raises(LoweringError) as exc_info:
        visitor.visit(call)

    formatted = str(exc_info.value)

    assert "5:2" in formatted
    assert "IRX-L001" in formatted
    assert "lowering" in formatted
    assert (
        "missing resolved_call metadata during call to 'helper'" in formatted
    )


def test_control_flow_boolean_lowering_failure_is_not_generic_exception() -> (
    None
):
    """
    title: Boolean lowering failures should use the structured lowering error.
    """
    visitor = Visitor()
    condition = astx.LiteralInt32(1)
    condition.loc = SimpleNamespace(line=9, col=4)

    with pytest.raises(LoweringError) as exc_info:
        visitor._lower_boolean_condition(
            ir.Constant(ir.IntType(32), 1),
            node=condition,
            context="if",
        )

    formatted = str(exc_info.value)

    assert "9:4" in formatted
    assert "IRX-L010" in formatted
    assert "if condition must lower to LLVM i1, got i32" in formatted


def test_public_imports_remain_stable() -> None:
    """
    title: Test public imports remain stable.
    """
    assert Builder.__name__ == "Builder"
    assert Visitor.__name__ == "Visitor"
    assert VariablesLLVM.__name__ == "VariablesLLVM"
    assert callable(emit_int_div)
    assert callable(is_fp_type)
    assert callable(safe_pop)
    assert callable(splat_scalar)
    assert not hasattr(builder_api, "LLVMLiteIR")
    assert not hasattr(builder_api, "LLVMLiteIRVisitor")


def test_helper_reexports_work_from_package() -> None:
    """
    title: Test helper reexports work from package.
    """
    visitor = Visitor()
    fn_ty = ir.FunctionType(visitor._llvm.FLOAT_TYPE, [])
    fn = ir.Function(visitor._llvm.module, fn_ty, name="helper_cover")
    block = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(block)

    scalar = ir.Constant(visitor._llvm.FLOAT_TYPE, 1.0)
    vec_ty = ir.VectorType(visitor._llvm.FLOAT_TYPE, 2)
    result = splat_scalar(visitor._llvm.ir_builder, scalar, vec_ty)

    assert isinstance(result.type, ir.VectorType)

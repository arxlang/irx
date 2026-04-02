"""
title: Integration tests for the semantic-analysis pipeline.
"""

from __future__ import annotations

import irx.builders.llvmliteir as llvmliteir_backend
import pytest

from irx import astx
from irx.analysis import SemanticError
from irx.builders.llvmliteir import (
    Builder,
    VariablesLLVM,
    Visitor,
    emit_int_div,
    is_fp_type,
    safe_pop,
    splat_scalar,
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

    with pytest.raises(SemanticError, match="Unknown variable name"):
        visitor.translate(module)


def test_valid_modules_still_emit_ir_after_analysis() -> None:
    """
    title: Test valid modules still emit ir after analysis.
    """
    builder = Builder()
    module = _main_module(astx.FunctionReturn(astx.LiteralInt32(0)))

    ir_text = builder.translate(module)

    assert 'define i32 @"main"()' in ir_text
    assert "ret i32 0" in ir_text


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
    assert not hasattr(llvmliteir_backend, "LLVMLiteIR")
    assert not hasattr(llvmliteir_backend, "LLVMLiteIRVisitor")


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

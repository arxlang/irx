"""
title: Integration tests for the semantic-analysis pipeline.
"""

from __future__ import annotations

import astx
import pytest

from irx.analysis import SemanticError
from irx.builders.llvmliteir import (
    LLVMLiteIR,
    LLVMLiteIRVisitor,
    VariablesLLVM,
    emit_int_div,
    is_fp_type,
    safe_pop,
    splat_scalar,
)
from llvmlite import ir


def _main_module(*nodes: astx.AST) -> astx.Module:
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
    builder = LLVMLiteIR()
    module = _main_module(
        astx.BreakStmt(),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    with pytest.raises(SemanticError, match="Break statement outside loop"):
        builder.translate(module)


def test_direct_visitor_translate_runs_analysis_before_codegen() -> None:
    visitor = LLVMLiteIRVisitor()
    module = _main_module(astx.FunctionReturn(astx.Identifier("missing")))

    with pytest.raises(SemanticError, match="Unknown variable name"):
        visitor.translate(module)


def test_valid_modules_still_emit_ir_after_analysis() -> None:
    builder = LLVMLiteIR()
    module = _main_module(astx.FunctionReturn(astx.LiteralInt32(0)))

    ir_text = builder.translate(module)

    assert 'define i32 @"main"()' in ir_text
    assert "ret i32 0" in ir_text


def test_public_imports_remain_stable() -> None:
    assert LLVMLiteIR.__name__ == "LLVMLiteIR"
    assert LLVMLiteIRVisitor.__name__ == "LLVMLiteIRVisitor"
    assert VariablesLLVM.__name__ == "VariablesLLVM"
    assert callable(emit_int_div)
    assert callable(is_fp_type)
    assert callable(safe_pop)
    assert callable(splat_scalar)


def test_helper_reexports_work_from_package() -> None:
    visitor = LLVMLiteIRVisitor()
    fn_ty = ir.FunctionType(visitor._llvm.FLOAT_TYPE, [])
    fn = ir.Function(visitor._llvm.module, fn_ty, name="helper_cover")
    block = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(block)

    scalar = ir.Constant(visitor._llvm.FLOAT_TYPE, 1.0)
    vec_ty = ir.VectorType(visitor._llvm.FLOAT_TYPE, 2)
    result = splat_scalar(visitor._llvm.ir_builder, scalar, vec_ty)

    assert isinstance(result.type, ir.VectorType)

"""
title: Tests for BreakStmt and ContinueStmt lowering
"""

from __future__ import annotations

from typing import cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir


def setup_function_context(visitor: LLVMLiteIRVisitor) -> None:
    """
    title: Setup LLVM function context for IR generation.
    parameters:
      visitor:
        type: LLVMLiteIRVisitor
    """
    module = visitor._llvm.module
    func_ty = ir.FunctionType(ir.IntType(32), [])
    func = ir.Function(module, func_ty, name="test_func")

    entry_bb = func.append_basic_block("entry")
    visitor._llvm.ir_builder.position_at_end(entry_bb)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_break_exits_loop(builder_class: type[Builder]) -> None:
    """
    title: Break exits loop immediately
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    setup_function_context(visitor)

    body = astx.Block()
    body.append(astx.BreakStmt())

    loop = astx.WhileStmt(
        condition=astx.LiteralInt32(1),
        body=body,
    )

    visitor.visit(loop)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)
    assert result.constant == 0


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_continue_in_loop(builder_class: type[Builder]) -> None:
    """
    title: Continue jumps to next iteration
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    setup_function_context(visitor)

    body = astx.Block()
    body.append(astx.ContinueStmt())

    loop = astx.WhileStmt(
        condition=astx.LiteralInt32(1),
        body=body,
    )

    visitor.visit(loop)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_nested_loop_break(builder_class: type[Builder]) -> None:
    """
    title: Break affects only inner loop in nested loops
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    setup_function_context(visitor)

    inner_body = astx.Block()
    inner_body.append(astx.BreakStmt())

    inner_loop = astx.WhileStmt(
        condition=astx.LiteralInt32(1),
        body=inner_body,
    )

    outer_body = astx.Block()
    outer_body.append(inner_loop)

    outer_loop = astx.WhileStmt(
        condition=astx.LiteralInt32(1),
        body=outer_body,
    )

    visitor.visit(outer_loop)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_break_outside_loop_raises(builder_class: type[Builder]) -> None:
    """
    title: Break outside loop raises exception
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(Exception, match="Break statement outside loop"):
        visitor.visit(astx.BreakStmt())


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_continue_outside_loop_raises(builder_class: type[Builder]) -> None:
    """
    title: Continue outside loop raises exception
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(Exception, match="Continue statement outside loop"):
        visitor.visit(astx.ContinueStmt())


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_break_skips_remaining_statements(
    builder_class: type[Builder],
) -> None:
    """
    title: Break prevents execution of subsequent statements in loop body
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    setup_function_context(visitor)

    body = astx.Block()
    body.append(astx.BreakStmt())

    # should NOT execute
    body.append(
        astx.VariableAssignment(
            name="x",
            value=astx.LiteralInt32(10),
        )
    )

    loop = astx.WhileStmt(
        condition=astx.LiteralInt32(1),
        body=body,
    )

    visitor.visit(loop)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_continue_skips_remaining_statements(
    builder_class: type[Builder],
) -> None:
    """
    title: Continue prevents execution of subsequent statements in loop body
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    setup_function_context(visitor)

    body = astx.Block()
    body.append(astx.ContinueStmt())

    # should NOT execute --
    body.append(
        astx.VariableAssignment(
            name="x",
            value=astx.LiteralInt32(10),
        )
    )

    loop = astx.WhileStmt(
        condition=astx.LiteralInt32(1),
        body=body,
    )

    visitor.visit(loop)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)

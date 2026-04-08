"""
title: Strong tests for BreakStmt and ContinueStmt lowering
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.builder import Builder as LLVMBuilder
from irx.builder import Visitor as LLVMVisitor
from irx.builder.base import Builder
from llvmlite import ir


def setup_function_context(visitor: LLVMVisitor) -> None:
    """
    title: Setup LLVM function context for IR generation
    parameters:
      visitor:
        type: LLVMVisitor
    """
    module = visitor._llvm.module
    func_ty = ir.FunctionType(ir.IntType(32), [])
    func = ir.Function(module, func_ty, name="test_func")

    entry_bb = func.append_basic_block("entry")
    visitor._llvm.ir_builder.position_at_end(entry_bb)


# BREAK TESTS
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_break_exits_to_after_block(builder_class: type[Builder]) -> None:
    """
    title: Break must branch to exit block (after loop)
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    setup_function_context(visitor)

    body = astx.Block()
    body.append(astx.BreakStmt())

    loop = astx.WhileStmt(
        condition=astx.LiteralInt32(1),
        body=body,
    )

    visitor.visit(loop)

    ir_text = str(visitor._llvm.module)

    # Must contain a branch (break → after block)
    assert 'br label %"afterwhile"' in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_break_skips_remaining_statements(
    builder_class: type[Builder],
) -> None:
    """
    title: Break must skip all statements after it in loop body
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    setup_function_context(visitor)

    body = astx.Block()
    body.append(astx.BreakStmt())

    # This must NEVER execute
    body.append(
        astx.VariableAssignment(
            "x",
            astx.LiteralInt32(10),
        )
    )

    loop = astx.WhileStmt(
        condition=astx.LiteralInt32(1),
        body=body,
    )

    visitor.visit(loop)

    ir_text = str(visitor._llvm.module)

    # The unreachable assignment must not emit a store.
    assert "store i32 10" not in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_nested_loop_break_affects_only_inner(
    builder_class: type[Builder],
) -> None:
    """
    title: Break should only exit inner loop, not outer loop
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
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

    ir_text = str(visitor._llvm.module)

    # Inner break must target the inner after-block, then outer flow continues.
    assert 'br label %"afterwhile.1"' in ir_text
    assert 'br label %"whilecond"' in ir_text


# CONTINUE TESTS
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_continue_branches_to_condition(builder_class: type[Builder]) -> None:
    """
    title: Continue must jump back to loop condition
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    setup_function_context(visitor)

    body = astx.Block()
    body.append(astx.ContinueStmt())

    loop = astx.WhileStmt(
        condition=astx.LiteralInt32(1),
        body=body,
    )

    visitor.visit(loop)

    ir_text = str(visitor._llvm.module)

    # Continue should branch back (cond block)
    assert 'br label %"whilecond"' in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_continue_skips_remaining_statements(
    builder_class: type[Builder],
) -> None:
    """
    title: Continue must skip statements after it in loop body
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    setup_function_context(visitor)

    body = astx.Block()
    body.append(astx.ContinueStmt())

    # Must not execute
    body.append(
        astx.VariableAssignment(
            "y",
            astx.LiteralInt32(20),
        )
    )

    loop = astx.WhileStmt(
        condition=astx.LiteralInt32(1),
        body=body,
    )

    visitor.visit(loop)

    ir_text = str(visitor._llvm.module)

    assert "store i32 20" not in ir_text


# ERROR CASES
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_break_outside_loop_raises(builder_class: type[Builder]) -> None:
    """
    title: Break outside loop must raise exception
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(Exception, match="Break statement outside loop"):
        visitor.visit(astx.BreakStmt())


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_continue_outside_loop_raises(builder_class: type[Builder]) -> None:
    """
    title: Continue outside loop must raise exception
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(Exception, match="Continue statement outside loop"):
        visitor.visit(astx.ContinueStmt())

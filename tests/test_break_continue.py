"""
title: Strong tests for BreakStmt and ContinueStmt lowering.
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.builder import Builder as LLVMBuilder
from irx.builder import Visitor as LLVMVisitor
from irx.builder.base import Builder
from irx.diagnostics import LoweringError
from llvmlite import ir


def block_of(*nodes: astx.AST) -> astx.Block:
    """
    title: Build one block from positional AST nodes.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Block
    """
    block = astx.Block()
    for node in nodes:
        block.append(node)
    return block


def setup_function_context(visitor: LLVMVisitor) -> None:
    """
    title: Setup LLVM function context for IR generation.
    parameters:
      visitor:
        type: LLVMVisitor
    """
    module = visitor._llvm.module
    func_ty = ir.FunctionType(ir.IntType(32), [])
    func = ir.Function(module, func_ty, name="test_func")
    entry_bb = func.append_basic_block("entry")
    visitor._llvm.ir_builder.position_at_end(entry_bb)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_break_exits_to_loop_exit_block(builder_class: type[Builder]) -> None:
    """
    title: Break must branch to the canonical loop exit block.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()
    setup_function_context(visitor)

    visitor.visit(
        astx.WhileStmt(
            condition=astx.LiteralBoolean(True),
            body=block_of(astx.BreakStmt()),
        )
    )

    ir_text = str(visitor._llvm.module)

    assert 'br label %"while.exit"' in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_break_skips_remaining_statements(
    builder_class: type[Builder],
) -> None:
    """
    title: Break must skip all statements after it in the loop body.
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
    body.append(astx.VariableAssignment("x", astx.LiteralInt32(10)))

    visitor.visit(
        astx.WhileStmt(condition=astx.LiteralBoolean(True), body=body)
    )

    ir_text = str(visitor._llvm.module)

    assert "store i32 10" not in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_nested_loop_break_affects_only_inner(
    builder_class: type[Builder],
) -> None:
    """
    title: Break should only exit the innermost active loop.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()
    setup_function_context(visitor)

    inner_loop = astx.WhileStmt(
        condition=astx.LiteralBoolean(True),
        body=block_of(astx.BreakStmt()),
    )
    outer_loop = astx.WhileStmt(
        condition=astx.LiteralBoolean(True),
        body=block_of(inner_loop),
    )

    visitor.visit(outer_loop)

    ir_text = str(visitor._llvm.module)

    assert 'br label %"while.exit.1"' in ir_text
    assert 'br label %"while.cond"' in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_continue_branches_to_condition(builder_class: type[Builder]) -> None:
    """
    title: Continue must jump back to the canonical while condition block.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()
    setup_function_context(visitor)

    visitor.visit(
        astx.WhileStmt(
            condition=astx.LiteralBoolean(True),
            body=block_of(astx.ContinueStmt()),
        )
    )

    ir_text = str(visitor._llvm.module)

    assert 'br label %"while.cond"' in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_continue_skips_remaining_statements(
    builder_class: type[Builder],
) -> None:
    """
    title: Continue must skip statements after it in the loop body.
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
    body.append(astx.VariableAssignment("y", astx.LiteralInt32(20)))

    visitor.visit(
        astx.WhileStmt(condition=astx.LiteralBoolean(True), body=body)
    )

    ir_text = str(visitor._llvm.module)

    assert "store i32 20" not in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_loop_lowering_does_not_consume_preexisting_result_stack_values(
    builder_class: type[Builder],
) -> None:
    """
    title: >-
      Loop lowering should not accidentally pop unrelated result-stack values.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()
    setup_function_context(visitor)

    marker = ir.Constant(ir.IntType(32), 123)
    visitor.result_stack.append(marker)

    visitor.visit(
        astx.ForCountLoopStmt(
            initializer=astx.InlineVariableDeclaration(
                "i",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            condition=astx.LiteralBoolean(False),
            update=astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
            body=astx.Block(),
        )
    )

    assert visitor.result_stack == [marker]


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_break_outside_loop_raises(builder_class: type[Builder]) -> None:
    """
    title: Break outside a loop must raise one structured lowering diagnostic.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(
        LoweringError, match="break statement used outside an active loop"
    ) as exc_info:
        visitor.visit(astx.BreakStmt())

    assert "IRX-L011" in str(exc_info.value)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_continue_outside_loop_raises(builder_class: type[Builder]) -> None:
    """
    title: >-
      Continue outside a loop must raise one structured lowering diagnostic.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(
        LoweringError, match="continue statement used outside an active loop"
    ) as exc_info:
        visitor.visit(astx.ContinueStmt())

    assert "IRX-L011" in str(exc_info.value)

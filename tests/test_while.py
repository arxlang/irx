"""
title: Test While Loop statements.
"""

import pytest

from irx import astx
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder

from .conftest import assert_ir_parses, assert_jit_int_main_result


def build_int32_main_module(body: astx.Block) -> astx.Module:
    """
    title: Build a module with one Int32 main function.
    parameters:
      body:
        type: astx.Block
    returns:
      type: astx.Module
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                name="main",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=body,
        )
    )
    return module


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


def add_assign(name: str, value: astx.AST) -> astx.VariableAssignment:
    """
    title: Build one additive variable assignment.
    parameters:
      name:
        type: str
      value:
        type: astx.AST
    returns:
      type: astx.VariableAssignment
    """
    return astx.VariableAssignment(
        name,
        astx.BinaryOp(op_code="+", lhs=astx.Identifier(name), rhs=value),
    )


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_while_lowers_for_numeric_counters(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: >-
      While loops should lower cleanly for supported numeric counter types.
    parameters:
      builder_class:
        type: type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "a",
                type_=int_type(),
                value=literal_type(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.WhileStmt(
                condition=astx.BinaryOp(
                    op_code="<",
                    lhs=astx.Identifier("a"),
                    rhs=literal_type(5),
                ),
                body=block_of(
                    astx.UnaryOp(op_code="++", operand=astx.Identifier("a"))
                ),
            ),
            astx.FunctionReturn(astx.LiteralInt32(0)),
        )
    )

    assert_jit_int_main_result(builder, module, 0)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_while_empty_body_no_crash(
    builder_class: type[Builder],
) -> None:
    """
    title: WhileStmt with empty body must not crash on an empty result stack.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "a",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.WhileStmt(
                condition=astx.LiteralBoolean(False),
                body=astx.Block(),
            ),
            astx.FunctionReturn(astx.LiteralInt32(0)),
        )
    )

    ir_text = builder.translate(module)
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_while_body_value_does_not_break_back_edge(
    builder_class: type[Builder],
) -> None:
    """
    title: WhileStmt must keep its back-edge even when the body leaves a value.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "a",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.WhileStmt(
                condition=astx.BinaryOp(
                    op_code="<",
                    lhs=astx.Identifier("a"),
                    rhs=astx.LiteralInt32(5),
                ),
                body=block_of(
                    astx.UnaryOp(op_code="++", operand=astx.Identifier("a")),
                    astx.LiteralInt32(0),
                ),
            ),
            astx.FunctionReturn(astx.LiteralInt32(0)),
        )
    )

    assert_jit_int_main_result(builder, module, 0)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_while_continue_uses_condition_block(
    builder_class: type[Builder],
) -> None:
    """
    title: WhileStmt continue must branch back to the condition block.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = build_int32_main_module(
        block_of(
            astx.WhileStmt(
                condition=astx.LiteralBoolean(True),
                body=block_of(astx.ContinueStmt()),
            ),
            astx.FunctionReturn(astx.LiteralInt32(0)),
        )
    )

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert "while.cond" in ir_text
    assert 'br label %"while.cond"' in ir_text
    assert "while.exit" in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_while_continue_rechecks_condition(
    builder_class: type[Builder],
) -> None:
    """
    title: WhileStmt continue should re-enter through the condition path.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.WhileStmt(
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(4),
        ),
        body=block_of(
            astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
            astx.IfStmt(
                condition=astx.BinaryOp(
                    op_code="<",
                    lhs=astx.Identifier("i"),
                    rhs=astx.LiteralInt32(3),
                ),
                then=block_of(astx.ContinueStmt()),
                else_=None,
            ),
            add_assign("sum", astx.Identifier("i")),
        ),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "i",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.InlineVariableDeclaration(
                "sum",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("sum")),
        )
    )

    assert_jit_int_main_result(builder, module, 7)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_while_break_allows_following_statements(
    builder_class: type[Builder],
) -> None:
    """
    title: WhileStmt break must still allow post-loop statements to run.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "result",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.WhileStmt(
                condition=astx.LiteralBoolean(True),
                body=block_of(astx.BreakStmt()),
            ),
            astx.VariableAssignment("result", astx.LiteralInt32(7)),
            astx.FunctionReturn(astx.Identifier("result")),
        )
    )

    assert_jit_int_main_result(builder, module, 7)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_while_false_condition_preserves_post_loop_state(
    builder_class: type[Builder],
) -> None:
    """
    title: >-
      A false initial while condition should leave post-loop state untouched.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "result",
                type_=astx.Int32(),
                value=astx.LiteralInt32(41),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.WhileStmt(
                condition=astx.LiteralBoolean(False),
                body=block_of(
                    astx.VariableAssignment("result", astx.LiteralInt32(0))
                ),
            ),
            astx.FunctionReturn(astx.Identifier("result")),
        )
    )

    assert_jit_int_main_result(builder, module, 41)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_nested_while_inner_break_only_exits_inner_loop(
    builder_class: type[Builder],
) -> None:
    """
    title: Nested while with inner break should keep the outer loop running.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    inner_loop = astx.WhileStmt(
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("j"),
            rhs=astx.LiteralInt32(5),
        ),
        body=block_of(
            astx.UnaryOp(op_code="++", operand=astx.Identifier("j")),
            astx.BreakStmt(),
        ),
    )
    outer_loop = astx.WhileStmt(
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(3),
        ),
        body=block_of(
            astx.VariableAssignment("j", astx.LiteralInt32(0)),
            inner_loop,
            astx.UnaryOp(op_code="++", operand=astx.Identifier("count")),
            astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
        ),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "i",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.InlineVariableDeclaration(
                "j",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.InlineVariableDeclaration(
                "count",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            outer_loop,
            astx.FunctionReturn(astx.Identifier("count")),
        )
    )

    assert_jit_int_main_result(builder, module, 3)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_nested_while_inner_continue_only_rechecks_inner_condition(
    builder_class: type[Builder],
) -> None:
    """
    title: Nested while with inner continue should not skip outer progress.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    inner_loop = astx.WhileStmt(
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("j"),
            rhs=astx.LiteralInt32(4),
        ),
        body=block_of(
            astx.UnaryOp(op_code="++", operand=astx.Identifier("j")),
            astx.IfStmt(
                condition=astx.BinaryOp(
                    op_code="<",
                    lhs=astx.Identifier("j"),
                    rhs=astx.LiteralInt32(3),
                ),
                then=block_of(astx.ContinueStmt()),
                else_=None,
            ),
            astx.UnaryOp(op_code="++", operand=astx.Identifier("sum")),
        ),
    )
    outer_loop = astx.WhileStmt(
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(2),
        ),
        body=block_of(
            astx.VariableAssignment("j", astx.LiteralInt32(0)),
            inner_loop,
            astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
        ),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "i",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.InlineVariableDeclaration(
                "j",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.InlineVariableDeclaration(
                "sum",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            outer_loop,
            astx.FunctionReturn(astx.Identifier("sum")),
        )
    )

    assert_jit_int_main_result(builder, module, 4)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_while_conditional_assignment_and_break_leave_valid_post_loop_state(
    builder_class: type[Builder],
) -> None:
    """
    title: >-
      Conditional body assignments and early break should keep post-loop state
      valid.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.WhileStmt(
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(5),
        ),
        body=block_of(
            astx.IfStmt(
                condition=astx.BinaryOp(
                    op_code="<",
                    lhs=astx.Identifier("i"),
                    rhs=astx.LiteralInt32(2),
                ),
                then=block_of(
                    astx.VariableAssignment("result", astx.LiteralInt32(10))
                ),
                else_=block_of(
                    astx.VariableAssignment("result", astx.LiteralInt32(20))
                ),
            ),
            astx.IfStmt(
                condition=astx.BinaryOp(
                    op_code="==",
                    lhs=astx.Identifier("i"),
                    rhs=astx.LiteralInt32(3),
                ),
                then=block_of(astx.BreakStmt()),
                else_=None,
            ),
            astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
        ),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "i",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.InlineVariableDeclaration(
                "result",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("result")),
        )
    )

    assert_jit_int_main_result(builder, module, 20)

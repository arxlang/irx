"""
title: Tests for Boolean semantics, lowering, and comparisons.
"""

from __future__ import annotations

import pytest

from irx import astx
from irx.analysis import SemanticError, analyze
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder

from .conftest import assert_ir_parses, make_main_module


def _boolean_main_module(*nodes: astx.AST) -> astx.Module:
    """
    title: Build a module with one Boolean-returning main function.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    return make_main_module(*nodes, return_type=astx.Boolean())


def _int_main_module(*nodes: astx.AST) -> astx.Module:
    """
    title: Build a module with one int32-returning main function.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    return make_main_module(*nodes, return_type=astx.Int32())


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_boolean_literal_declaration_and_assignment(
    builder_class: type[Builder],
) -> None:
    """
    title: Boolean locals should round-trip through declaration and assignment.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = _boolean_main_module(
        astx.VariableDeclaration(
            name="flag",
            type_=astx.Boolean(),
            value=astx.LiteralBoolean(True),
            mutability=astx.MutabilityKind.mutable,
        ),
        astx.VariableAssignment(
            "flag",
            astx.LiteralBoolean(False),
        ),
        astx.FunctionReturn(astx.Identifier("flag")),
    )

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert "alloca i1" in ir_text
    assert "store i1 1" in ir_text
    assert "store i1 0" in ir_text
    assert "ret i1" in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_function_accepts_boolean_parameter_and_returns_boolean(
    builder_class: type[Builder],
) -> None:
    """
    title: >-
      Boolean params and returns should lower cleanly as first-class values.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    negate_proto = astx.FunctionPrototype(
        name="negate",
        args=astx.Arguments(astx.Argument("flag", astx.Boolean())),
        return_type=astx.Boolean(),
    )
    negate_body = astx.Block()
    negate_body.append(
        astx.FunctionReturn(
            astx.UnaryOp(op_code="!", operand=astx.Identifier("flag"))
        )
    )
    module.block.append(
        astx.FunctionDef(prototype=negate_proto, body=negate_body)
    )

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_body = astx.Block()
    main_body.append(astx.FunctionCall("negate", [astx.LiteralBoolean(True)]))
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=main_body))

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert "define i1" in ir_text
    assert "call i1" in ir_text
    assert "xor i1" in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_comparison_result_assigns_to_boolean_variable(
    builder_class: type[Builder],
) -> None:
    """
    title: Comparison results should assign into Boolean locals.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = _boolean_main_module(
        astx.VariableDeclaration(
            name="is_less",
            type_=astx.Boolean(),
            value=astx.BinaryOp(
                "<",
                astx.LiteralInt32(1),
                astx.LiteralInt32(2),
            ),
            mutability=astx.MutabilityKind.mutable,
        ),
        astx.FunctionReturn(astx.Identifier("is_less")),
    )

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert "icmp " in ir_text
    assert "store i1" in ir_text
    assert "ret i1" in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_comparison_result_returns_from_function(
    builder_class: type[Builder],
) -> None:
    """
    title: Comparison results should return directly from Boolean functions.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    compare_proto = astx.FunctionPrototype(
        name="less",
        args=astx.Arguments(
            astx.Argument("lhs", astx.Int32()),
            astx.Argument("rhs", astx.Int32()),
        ),
        return_type=astx.Boolean(),
    )
    compare_body = astx.Block()
    compare_body.append(
        astx.FunctionReturn(
            astx.BinaryOp(
                "<",
                astx.Identifier("lhs"),
                astx.Identifier("rhs"),
            )
        )
    )
    module.block.append(
        astx.FunctionDef(prototype=compare_proto, body=compare_body)
    )

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_body = astx.Block()
    main_body.append(
        astx.FunctionCall(
            "less",
            [astx.LiteralInt32(1), astx.LiteralInt32(2)],
        )
    )
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=main_body))

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert "define i1" in ir_text
    assert "call i1" in ir_text
    assert "icmp " in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_boolean_condition_in_if(builder_class: type[Builder]) -> None:
    """
    title: If statements should accept Boolean conditions only.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    module = _int_main_module(
        astx.VariableDeclaration(
            name="flag",
            type_=astx.Boolean(),
            value=astx.LiteralBoolean(True),
            mutability=astx.MutabilityKind.mutable,
        ),
        astx.IfStmt(
            condition=astx.Identifier("flag"),
            then=then_block,
            else_=else_block,
        ),
    )

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert "br i1" in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_boolean_condition_in_while(builder_class: type[Builder]) -> None:
    """
    title: While statements should branch on Boolean locals.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    body = astx.Block()
    body.append(astx.VariableAssignment("count", astx.LiteralInt32(1)))
    body.append(astx.VariableAssignment("flag", astx.LiteralBoolean(False)))

    module = _int_main_module(
        astx.VariableDeclaration(
            name="flag",
            type_=astx.Boolean(),
            value=astx.LiteralBoolean(True),
            mutability=astx.MutabilityKind.mutable,
        ),
        astx.VariableDeclaration(
            name="count",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        astx.WhileStmt(condition=astx.Identifier("flag"), body=body),
        astx.FunctionReturn(astx.Identifier("count")),
    )

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert "br i1" in ir_text
    assert "icmp " not in ir_text
    assert "fcmp " not in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_boolean_condition_in_for_count_loop(
    builder_class: type[Builder],
) -> None:
    """
    title: For-count loops should branch on Boolean conditions.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop_body = astx.Block()
    loop_body.append(astx.VariableAssignment("count", astx.LiteralInt32(1)))

    module = _int_main_module(
        astx.VariableDeclaration(
            name="count",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        astx.ForCountLoopStmt(
            initializer=astx.InlineVariableDeclaration(
                name="flag",
                type_=astx.Boolean(),
                value=astx.LiteralBoolean(True),
                mutability=astx.MutabilityKind.mutable,
            ),
            condition=astx.Identifier("flag"),
            update=astx.BinaryOp(
                "=",
                astx.Identifier("flag"),
                astx.LiteralBoolean(False),
            ),
            body=loop_body,
        ),
        astx.FunctionReturn(astx.Identifier("count")),
    )

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert "br i1" in ir_text
    assert "icmp " not in ir_text
    assert "fcmp " not in ir_text
    assert "loop.body" in ir_text


@pytest.mark.parametrize(
    "op",
    [
        "&&",
        "||",
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_boolean_logical_binary_operations(
    builder_class: type[Builder],
    op: str,
) -> None:
    """
    title: Logical binary operators should accept only Boolean operands.
    parameters:
      builder_class:
        type: type[Builder]
      op:
        type: str
    """
    builder = builder_class()
    module = _boolean_main_module(
        astx.FunctionReturn(
            astx.BinaryOp(
                op,
                astx.LiteralBoolean(True),
                astx.LiteralBoolean(False),
            )
        )
    )

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert ("and i1" if op == "&&" else "or i1") in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_boolean_logical_not(builder_class: type[Builder]) -> None:
    """
    title: Logical not should operate on Boolean values and return Boolean.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = _boolean_main_module(
        astx.FunctionReturn(astx.UnaryOp("!", astx.LiteralBoolean(True)))
    )

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert "xor i1" in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_boolean_if_condition_branches_without_truthiness_compare(
    builder_class: type[Builder],
) -> None:
    """
    title: >-
      Boolean conditions should branch directly on i1 without zero compares.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    module = _int_main_module(
        astx.VariableDeclaration(
            name="flag",
            type_=astx.Boolean(),
            value=astx.LiteralBoolean(True),
            mutability=astx.MutabilityKind.mutable,
        ),
        astx.IfStmt(
            condition=astx.Identifier("flag"),
            then=then_block,
            else_=else_block,
        ),
    )

    ir_text = builder.translate(module)

    assert "br i1" in ir_text
    assert "icmp " not in ir_text
    assert "fcmp " not in ir_text


def test_rejects_if_integer_condition() -> None:
    """
    title: If conditions must be Boolean semantically.
    """
    module = _int_main_module(
        astx.IfStmt(condition=astx.LiteralInt32(1), then=astx.Block()),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    with pytest.raises(SemanticError, match="if condition must be Boolean"):
        analyze(module)


def test_rejects_while_float_condition() -> None:
    """
    title: While conditions must be Boolean semantically.
    """
    module = _int_main_module(
        astx.WhileStmt(
            condition=astx.LiteralFloat64(3.14),
            body=astx.Block(),
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    with pytest.raises(SemanticError, match="while condition must be Boolean"):
        analyze(module)


def test_rejects_non_boolean_for_count_condition() -> None:
    """
    title: For-count loop conditions must be Boolean semantically.
    """
    module = _int_main_module(
        astx.ForCountLoopStmt(
            initializer=astx.InlineVariableDeclaration(
                name="i",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            condition=astx.LiteralInt32(1),
            update=astx.UnaryOp("++", astx.Identifier("i")),
            body=astx.Block(),
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    with pytest.raises(
        SemanticError,
        match="for-count loop condition must be Boolean",
    ):
        analyze(module)


def test_rejects_integer_logical_and() -> None:
    """
    title: Logical and must reject non-Boolean operands.
    """
    expr = astx.BinaryOp(
        "&&",
        astx.LiteralInt32(1),
        astx.LiteralInt32(2),
    )

    with pytest.raises(
        SemanticError,
        match=r"logical operator '&&' requires Boolean operands",
    ):
        analyze(expr)


def test_rejects_integer_logical_or() -> None:
    """
    title: Logical or must reject non-Boolean operands.
    """
    expr = astx.BinaryOp(
        "||",
        astx.LiteralInt32(1),
        astx.LiteralInt32(2),
    )

    with pytest.raises(
        SemanticError,
        match=r"logical operator '\|\|' requires Boolean operands",
    ):
        analyze(expr)


def test_rejects_integer_logical_not() -> None:
    """
    title: Logical not must reject non-Boolean operands.
    """
    expr = astx.UnaryOp("!", astx.LiteralInt32(1))

    with pytest.raises(
        SemanticError,
        match=r"unary operator '!' requires Boolean operand",
    ):
        analyze(expr)

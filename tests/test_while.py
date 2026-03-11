"""
title: Test While Loop statements.
"""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_while_expr.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_while_expr(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test the While expression translation.
    parameters:
      action:
        type: str
      expected_file:
        type: str
      builder_class:
        type: Type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()

    # Identifier declaration and initialization: int a = 0
    init_var = astx.InlineVariableDeclaration(
        "a",
        type_=int_type(),
        value=literal_type(0),
        mutability=astx.MutabilityKind.mutable,
    )

    # Condition: a < 5
    var_a = astx.Identifier("a")
    cond = astx.BinaryOp(op_code="<", lhs=var_a, rhs=literal_type(5))

    # Update: ++a
    update = astx.UnaryOp(op_code="++", operand=var_a)

    # Body
    body = astx.Block()
    body.append(update)
    body.append(literal_type(2))

    while_expr = astx.WhileStmt(condition=cond, body=body)

    # Main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    fn_block = astx.Block()
    fn_block.append(init_var)
    fn_block.append(while_expr)
    fn_block.append(astx.FunctionReturn(literal_type(0)))

    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)


def test_while_stmt_float_condition() -> None:
    """
    title: Test WhileStmt with float condition (lines 1273, 1277-1278).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="count",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )

    # while(count) { count = count - 1.0 }
    sub_expr = astx.BinaryOp(
        "-", astx.Identifier("count"), astx.LiteralFloat32(1.0)
    )
    assign = astx.VariableAssignment(name="count", value=sub_expr)
    body = astx.Block()
    body.append(assign)

    while_stmt = astx.WhileStmt(
        condition=astx.Identifier("count"),
        body=body,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(while_stmt)
    block.append(PrintExpr(astx.LiteralUTF8String("DONE")))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="DONE")


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_while_false_condition(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test While loop with a condition that is false from the start.
    parameters:
      action:
        type: str
      expected_file:
        type: str
      builder_class:
        type: Type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()

    # Condition is always false: 10 < 0
    cond = astx.BinaryOp(
        op_code="<",
        lhs=literal_type(10),
        rhs=literal_type(0),
    )

    # Body that should never execute
    body = astx.Block()
    body.append(literal_type(1))

    while_expr = astx.WhileStmt(condition=cond, body=body)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    fn_block = astx.Block()
    fn_block.append(while_expr)
    fn_block.append(astx.FunctionReturn(literal_type(0)))

    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)

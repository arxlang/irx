"""
title: Test irx with a fibonnaci function.
"""

from __future__ import annotations

import pytest

from irx import astx
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder
from irx.system import Cast

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
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_function_call_fibonacci(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test the FunctionCall class with Fibonacci.
    parameters:
      builder_class:
        type: type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()
    module = builder.module()

    # Define Fibonacci function
    fib_proto = astx.FunctionPrototype(
        name="fib",
        args=astx.Arguments(astx.Argument("n", int_type())),
        return_type=int_type(),
    )
    fib_block = astx.Block()

    astx.VariableDeclaration(
        name="a",
        type_=int_type(),
        value=literal_type(0),
        parent=fib_block,
        mutability=astx.MutabilityKind.mutable,
    )
    astx.VariableDeclaration(
        name="b",
        type_=int_type(),
        value=literal_type(1),
        parent=fib_block,
        mutability=astx.MutabilityKind.mutable,
    )
    astx.VariableDeclaration(
        name="i",
        type_=int_type(),
        value=literal_type(2),
        parent=fib_block,
        mutability=astx.MutabilityKind.mutable,
    )
    astx.VariableDeclaration(
        name="sum",
        type_=int_type(),
        value=literal_type(10),
        parent=fib_block,
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp(
        op_code="<=",
        lhs=astx.Identifier(name="i"),
        rhs=astx.Identifier(name="n"),
    )
    loop_block = astx.Block()
    loop_block.append(
        astx.VariableAssignment(
            name="sum",
            value=astx.BinaryOp(
                op_code="+", lhs=astx.Identifier("a"), rhs=astx.Identifier("b")
            ),
        )
    )
    loop_block.append(
        astx.VariableAssignment(name="a", value=astx.Identifier("b"))
    )
    loop_block.append(
        astx.VariableAssignment(name="b", value=astx.Identifier("sum"))
    )
    loop_block.append(
        astx.VariableAssignment(
            name="i",
            value=astx.BinaryOp(
                op_code="+", lhs=astx.Identifier("i"), rhs=literal_type(1)
            ),
        )
    )
    loop = astx.WhileStmt(condition=cond, body=loop_block)
    fib_block.append(loop)
    fib_block.append(astx.FunctionReturn(astx.Identifier(name="b")))

    fib_fn = astx.FunctionDef(prototype=fib_proto, body=fib_block)
    module.block.append(fib_fn)

    # Main function calling fib()
    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_block = astx.Block()
    call_fib = astx.FunctionCall(fib_fn, [literal_type(10)])
    main_block.append(
        astx.FunctionReturn(Cast(value=call_fib, target_type=astx.Int32()))
    )
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    expected_output = "55"  # fib(10) = 55

    check_result("build", builder, module, expected_output=expected_output)

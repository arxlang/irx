"""Test irx with a fibonnaci function."""

from __future__ import annotations

import subprocess

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_function_call_fibonacci(
    builder_class: Type[Builder],
) -> None:
    """Test the FunctionCall class with Fibonacci."""
    builder = builder_class()
    module = builder.module()

    # Define Fibonacci function
    fib_proto = astx.FunctionPrototype(
        name="fib",
        args=astx.Arguments(astx.Argument("n", astx.Int32())),
        return_type=astx.Int32(),
    )
    fib_block = astx.Block()

    astx.VariableDeclaration(
        name="a",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        parent=fib_block,
    )
    astx.VariableDeclaration(
        name="b",
        type_=astx.Int32(),
        value=astx.LiteralInt32(1),
        parent=fib_block,
    )
    astx.VariableDeclaration(
        name="i",
        type_=astx.Int32(),
        value=astx.LiteralInt32(2),
        parent=fib_block,
    )
    astx.VariableDeclaration(
        name="sum",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        parent=fib_block,
    )

    cond = astx.BinaryOp(
        op_code="<=",
        lhs=astx.Variable(name="i"),
        rhs=astx.Variable(name="n"),
    )
    loop_block = astx.Block()
    loop_block.append(
        astx.VariableAssignment(
            name="sum",
            value=astx.BinaryOp(
                op_code="+", lhs=astx.Variable("a"), rhs=astx.Variable("b")
            ),
        )
    )
    loop_block.append(
        astx.VariableAssignment(name="a", value=astx.Variable("b"))
    )
    loop_block.append(
        astx.VariableAssignment(name="b", value=astx.Variable("sum"))
    )
    loop_block.append(
        astx.VariableAssignment(
            name="i",
            value=astx.BinaryOp(
                op_code="+", lhs=astx.Variable("i"), rhs=astx.LiteralInt32(1)
            ),
        )
    )
    loop = astx.WhileStmt(condition=cond, body=loop_block)
    fib_block.append(loop)
    fib_block.append(astx.FunctionReturn(astx.Variable(name="b")))

    fib_fn = astx.Function(prototype=fib_proto, body=fib_block)
    module.block.append(fib_fn)

    # Main function calling fib()
    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_block = astx.Block()
    call_fib = astx.FunctionCall(fib_fn, [astx.LiteralInt32(10)])
    main_block.append(astx.FunctionReturn(call_fib))
    main_fn = astx.Function(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    expected_output = "55"  # fib(10) = 55
    success = True

    try:
        check_result("build", builder, module, expected_output=expected_output)
    except subprocess.CalledProcessError as e:
        success = False
        assert e.returncode == int(expected_output)

    assert not success

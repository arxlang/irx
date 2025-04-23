"""Test irx with a fibonnaci function."""

from __future__ import annotations

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


@pytest.mark.parametrize(
    "action,expected_file",
    [("build", "")],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_function_call_fibonacci(
    action: str, expected_file: str, builder_class: Type[Builder]
) -> None:
    """Test the FunctionCall class with Fibonacci."""
    # Initialize the ASTx module
    builder = builder_class()
    module = builder.module()

    # Define the Fibonacci function prototype
    fib_proto = astx.FunctionPrototype(
        name="fib",
        args=astx.Arguments(astx.Argument("n", astx.Int32())),
        return_type=astx.Int32(),
    )

    # Create the function body block
    fib_block = astx.Block()

    # Declare the variables
    decl_a = astx.VariableDeclaration(  # noqa: F841
        name="a",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        parent=fib_block,
    )
    decl_b = astx.VariableDeclaration(  # noqa: F841
        name="b",
        type_=astx.Int32(),
        value=astx.LiteralInt32(1),
        parent=fib_block,
    )
    decl_i = astx.VariableDeclaration(  # noqa: F841
        name="i",
        type_=astx.Int32(),
        value=astx.LiteralInt32(2),
        parent=fib_block,
    )
    decl_sum = astx.VariableDeclaration(  # noqa: F841
        name="sum",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        parent=fib_block,
    )

    # Create the loop condition
    cond = astx.BinaryOp(
        op_code="<", lhs=astx.Variable(name="i"), rhs=astx.Variable(name="n")
    )

    # Define the loop body
    loop_block = astx.Block()
    assign_sum = astx.VariableAssignment(
        name="sum",
        value=astx.BinaryOp(
            op_code="+",
            lhs=astx.Variable(name="a"),
            rhs=astx.Variable(name="b"),
        ),
    )
    assign_a = astx.VariableAssignment(name="a", value=astx.Variable(name="b"))
    assign_b = astx.VariableAssignment(
        name="b", value=astx.Variable(name="sum")
    )
    inc_i = astx.VariableAssignment(
        name="i",
        value=astx.BinaryOp(
            op_code="+", lhs=astx.Variable(name="i"), rhs=astx.LiteralInt32(1)
        ),
    )

    # Add declarations and assignments to the loop body
    loop_block.append(assign_sum)
    loop_block.append(assign_a)
    loop_block.append(assign_b)
    loop_block.append(inc_i)

    # Create the loop statement
    loop = astx.WhileStmt(condition=cond, body=loop_block)
    fib_block.append(loop)

    # Add return statement
    return_stmt = astx.FunctionReturn(astx.Variable(name="b"))
    fib_block.append(return_stmt)

    # Define the function with its body
    fib_fn = astx.Function(prototype=fib_proto, body=fib_block)

    # Append the Fibonacci function to the module block

    # Add after `fib_fn` is appended to module.block

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )

    main_block = astx.Block()

    call_fib = astx.FunctionCall(fib_fn, [astx.LiteralInt32(10)])

    main_return = astx.FunctionReturn(call_fib)
    main_block.append(main_return)

    main_fn = astx.Function(prototype=main_proto, body=main_block)
    module.block.append(fib_fn)
    module.block.append(main_fn)

    check_result(action, builder, module, expected_file)

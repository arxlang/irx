"""
title: Tests for the Module AST.
"""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


def make_fn_add(int_type: type, literal_type: type) -> astx.AST:
    """
    title: Create a fixture for a function add.
    parameters:
      int_type:
        type: type
      literal_type:
        type: type
    returns:
      type: astx.AST
    """
    var_a = astx.Argument(name="a", type_=int_type(), default=literal_type(1))
    var_b = astx.Argument(name="b", type_=int_type(), default=literal_type(2))

    proto = astx.FunctionPrototype(
        name="add", args=astx.Arguments(var_a, var_b), return_type=int_type()
    )
    block = astx.Block()
    var_sum = var_a + var_b
    block.append(astx.FunctionReturn(var_sum))
    return astx.FunctionDef(prototype=proto, body=block)


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
        # ("translate", "test_module_fn_main.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_module_fn_main(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test ASTx Module with a main function and a function called add.
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

    fn_add = make_fn_add(int_type, literal_type)

    module = builder.module()
    module.block.append(fn_add)

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(PrintExpr(astx.LiteralUTF8String("MODULE_OK")))
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result(
        action,
        builder,
        module,
        expected_file,
        expected_output="MODULE_OK",
    )


def test_multiple_function_calls() -> None:
    """
    title: Test calling a user function that uses FunctionCall.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    # Helper function: add(a, b) -> a + b
    arg_a = astx.Argument(
        name="a",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
    )
    arg_b = astx.Argument(
        name="b",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
    )
    args = astx.Arguments()
    args.append(arg_a)
    args.append(arg_b)

    add_proto = astx.FunctionPrototype(
        name="add", args=args, return_type=astx.Int32()
    )
    add_body = astx.Block()
    add_expr = astx.BinaryOp(
        op_code="+",
        lhs=astx.Identifier("a"),
        rhs=astx.Identifier("b"),
    )
    add_body.append(astx.FunctionReturn(add_expr))
    add_fn = astx.FunctionDef(prototype=add_proto, body=add_body)
    module.block.append(add_fn)

    # main calls add(10, 32)
    call = astx.FunctionCall(
        fn="add",
        args=[astx.LiteralInt32(10), astx.LiteralInt32(32)],
    )

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(astx.FunctionReturn(call))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="42")

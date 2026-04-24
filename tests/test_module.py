"""
title: Tests for the Module AST.
"""

import pytest

from irx import astx
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder

from .conftest import assert_jit_int_main_result, check_result


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


def make_fn_duplicate_default() -> astx.AST:
    """
    title: Create one function whose default argument references a prior one.
    returns:
      type: astx.AST
    """
    value = astx.Argument(name="value", type_=astx.Int32())
    copy = astx.Argument(
        name="copy",
        type_=astx.Int32(),
        default=astx.Identifier("value"),
    )
    prototype = astx.FunctionPrototype(
        name="duplicate",
        args=astx.Arguments(value, copy),
        return_type=astx.Int32(),
    )
    body = astx.Block()
    body.append(astx.FunctionReturn(value + copy))
    return astx.FunctionDef(prototype=prototype, body=body)


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
        LLVMBuilder,
    ],
)
def test_module_fn_main(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
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
        type: type[Builder]
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
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_block = astx.Block()
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_function_call_uses_declared_default_argument(
    builder_class: type[Builder],
) -> None:
    """
    title: Omitted trailing function arguments should lower through defaults.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    function = make_fn_duplicate_default()
    module = builder.module()
    module.block.append(function)

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_body = astx.Block()
    main_body.append(
        astx.FunctionReturn(
            astx.FunctionCall("duplicate", [astx.LiteralInt32(5)])
        )
    )
    module.block.append(astx.FunctionDef(prototype=main_proto, body=main_body))

    assert_jit_int_main_result(builder, module, 10)

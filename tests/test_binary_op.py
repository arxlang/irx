"""Tests for the BinaryOp."""

import subprocess

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [(astx.Int32, astx.LiteralInt32), (astx.Int16, astx.LiteralInt16)],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_binary_op_literals(
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """Test ASTx Module with a function called add."""
    builder = builder_class()
    module = builder.module()

    basic_op = literal_type(1) + literal_type(2)
    expected_output = "3"

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(astx.FunctionReturn(basic_op))
    main_fn = astx.Function(prototype=main_proto, body=main_block)

    module.block.append(main_fn)
    success = True

    # the try/except is just a workaround, because for now "PrintExpr"
    # cannot convert integer to string, and there is no casting function
    # to convert from integer to string.
    try:
        check_result("build", builder, module, expected_output=expected_output)
    except subprocess.CalledProcessError as e:
        success = False
        assert e.returncode == int(expected_output)
    assert not success


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
        # ("translate", "test_binary_op_basic.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_binary_op_basic(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """Test ASTx Module with a function called add."""
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a", type_=int_type(), value=literal_type(1)
    )
    decl_b = astx.VariableDeclaration(
        name="b", type_=int_type(), value=literal_type(2)
    )
    decl_c = astx.VariableDeclaration(
        name="c", type_=int_type(), value=literal_type(4)
    )

    a = astx.Variable("a")
    b = astx.Variable("b")
    c = astx.Variable("c")

    lit_1 = literal_type(1)

    basic_op = lit_1 + b - a * c / a + (b - a + c / a)

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(decl_b)
    main_block.append(decl_c)
    main_block.append(basic_op)
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.Function(prototype=main_proto, body=main_block)

    module.block.append(main_fn)
    check_result(action, builder, module, expected_file)

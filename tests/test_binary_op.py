"""Tests for the Module AST."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


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
    action: str, expected_file: str, builder_class: Type[Builder]
) -> None:
    """Test ASTx Module with a function called add."""
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a", type_=astx.Int32(), value=astx.LiteralInt32(1)
    )
    decl_b = astx.VariableDeclaration(
        name="b", type_=astx.Int32(), value=astx.LiteralInt32(2)
    )
    decl_c = astx.VariableDeclaration(
        name="c", type_=astx.Int32(), value=astx.LiteralInt32(4)
    )

    a = astx.Variable("a")
    b = astx.Variable("b")
    c = astx.Variable("c")

    lit_1 = astx.LiteralInt32(1)

    basic_op = lit_1 + b - a * c / a + (b - a + c / a)

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(decl_b)
    main_block.append(decl_c)
    main_block.append(astx.FunctionReturn(basic_op))
    main_fn = astx.Function(prototype=main_proto, body=main_block)

    module.block.append(main_fn)
    check_result(action, builder, module, expected_file)


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
def test_binary_op_float(
    action: str, expected_file: str, builder_class: Type[Builder]
) -> None:
    """Test ASTx Module with float operations."""
    builder = builder_class()
    module = builder.module()

    decl_x = astx.VariableDeclaration(
        name="x", type_=astx.Float32(), value=astx.LiteralFloat32(1.5)
    )
    decl_y = astx.VariableDeclaration(
        name="y", type_=astx.Float32(), value=astx.LiteralFloat32(2.5)
    )
    decl_z = astx.VariableDeclaration(
        name="z", type_=astx.Float32(), value=astx.LiteralFloat32(4.0)
    )

    x = astx.Variable("x")
    y = astx.Variable("y")
    z = astx.Variable("z")

    lit_2 = astx.LiteralFloat32(1.0)

    basic_op = lit_2 + y - z * z / x + (y - x + z / x)

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Float32()
    )
    main_block = astx.Block()
    main_block.append(decl_x)
    main_block.append(decl_y)
    main_block.append(decl_z)
    main_block.append(astx.FunctionReturn(basic_op))
    main_fn = astx.Function(prototype=main_proto, body=main_block)

    module.block.append(main_fn)
    check_result(action, builder, module, expected_file)

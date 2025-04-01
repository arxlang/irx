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
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_binary_op_int16(builder_class: Type[Builder]) -> None:
    """Test ASTx Module with int16 binary operations."""
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a", type_=astx.Int16(), value=astx.LiteralInt16(10)
    )
    decl_b = astx.VariableDeclaration(
        name="b", type_=astx.Int16(), value=astx.LiteralInt16(20)
    )

    a = astx.Variable("a")
    b = astx.Variable("b")

    lit_2 = astx.LiteralInt16(2)
    basic_op = (a * lit_2) + b

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int16()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(decl_b)
    main_block.append(astx.FunctionReturn(basic_op))
    main_fn = astx.Function(prototype=main_proto, body=main_block)

    module.block.append(main_fn)
    check_result("build", builder, module, "")
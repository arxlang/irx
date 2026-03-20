"""
title: Tests for Bitwise AND (&) operator.
"""

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
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_bitwise_and(builder_class: type[Builder], int_type: type, literal_type: type) -> None:
    """
    title: Test bitwise AND (&) operation on integer literals.
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

    expr = astx.BinaryOp("&", literal_type(6), literal_type(3))
    decl = astx.VariableDeclaration(
        name="result", type_=int_type(), value=expr, mutability=astx.MutabilityKind.mutable
    )

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl)
    main_block.append(PrintExpr(astx.Identifier("result")))
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="2")


@pytest.mark.parametrize("int_type, literal_type", [
    (astx.Int32, astx.LiteralInt32),
    (astx.Int16, astx.LiteralInt16),
    (astx.Int8, astx.LiteralInt8),
    (astx.Int64, astx.LiteralInt64),
])
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_bitwise_or(builder_class: type[Builder], int_type: type, literal_type: type) -> None:
    builder = builder_class()
    module = builder.module()
    expr = astx.BinaryOp("|", literal_type(6), literal_type(3))
    decl = astx.VariableDeclaration(
      name="result", type_=int_type(), value=expr, mutability=astx.MutabilityKind.mutable
    )
    main_proto = astx.FunctionPrototype(
      name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl)
    main_block.append(PrintExpr(astx.Identifier("result")))
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)
    check_result("build", builder, module, expected_output="7")


@pytest.mark.parametrize("int_type, literal_type", [
    (astx.Int32, astx.LiteralInt32),
    (astx.Int16, astx.LiteralInt16),
    (astx.Int8, astx.LiteralInt8),
    (astx.Int64, astx.LiteralInt64),
])
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_bitwise_xor(builder_class: type[Builder], int_type: type, literal_type: type) -> None:
    builder = builder_class()
    module = builder.module()
    expr = astx.BinaryOp("^", literal_type(6), literal_type(3))
    decl = astx.VariableDeclaration(
      name="result", type_=int_type(), value=expr, mutability=astx.MutabilityKind.mutable
    )
    main_proto = astx.FunctionPrototype(
      name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl)
    main_block.append(PrintExpr(astx.Identifier("result")))
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)
    check_result("build", builder, module, expected_output="5")


@pytest.mark.parametrize("int_type, literal_type", [
    (astx.Int32, astx.LiteralInt32),
    (astx.Int16, astx.LiteralInt16),
    (astx.Int8, astx.LiteralInt8),
    (astx.Int64, astx.LiteralInt64),
])
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_left_shift(builder_class: type[Builder], int_type: type, literal_type: type) -> None:
    builder = builder_class()
    module = builder.module()
    expr = astx.BinaryOp("<<", literal_type(3), literal_type(2))
    decl = astx.VariableDeclaration(
      name="result", type_=int_type(), value=expr, mutability=astx.MutabilityKind.mutable
    )
    main_proto = astx.FunctionPrototype(
      name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl)
    main_block.append(PrintExpr(astx.Identifier("result")))
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)
    check_result("build", builder, module, expected_output="12")


@pytest.mark.parametrize("int_type, literal_type", [
    (astx.Int32, astx.LiteralInt32),
    (astx.Int16, astx.LiteralInt16),
    (astx.Int8, astx.LiteralInt8),
    (astx.Int64, astx.LiteralInt64),
])
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_right_shift(builder_class: type[Builder], int_type: type, literal_type: type) -> None:
    builder = builder_class()
    module = builder.module()
    expr = astx.BinaryOp(">>", literal_type(12), literal_type(2))
    decl = astx.VariableDeclaration(
      name="result", type_=int_type(), value=expr, mutability=astx.MutabilityKind.mutable
    )
    main_proto = astx.FunctionPrototype(
      name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl)
    main_block.append(PrintExpr(astx.Identifier("result")))
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)
    check_result("build", builder, module, expected_output="3")


@pytest.mark.parametrize("int_type, literal_type", [
    (astx.Int32, astx.LiteralInt32),
    (astx.Int16, astx.LiteralInt16),
    (astx.Int8, astx.LiteralInt8),
    (astx.Int64, astx.LiteralInt64),
])
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_bitwise_not(builder_class: type[Builder], int_type: type, literal_type: type) -> None:
    builder = builder_class()
    module = builder.module()
    expr = astx.UnaryOp("~", literal_type(15))
    decl = astx.VariableDeclaration(
      name="result", type_=int_type(), value=expr, mutability=astx.MutabilityKind.mutable
    )
    main_proto = astx.FunctionPrototype(
      name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl)
    main_block.append(PrintExpr(astx.Identifier("result")))
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)
    check_result("build", builder, module, expected_output=str(~15))

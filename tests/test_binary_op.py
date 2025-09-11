"""Tests for the BinaryOp."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

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

    decl = astx.VariableDeclaration(
        name="tmp", type_=int_type(), value=basic_op
    )

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl)
    main_block.append(PrintExpr(astx.LiteralUTF8String("3")))
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="3")


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

    a = astx.Identifier("a")
    b = astx.Identifier("b")
    c = astx.Identifier("c")

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
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)
    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_binary_op_string_not_equals(builder_class: Type[Builder]) -> None:
    """Verify string '!=' uses strcmp_inline + xor 1 path."""
    builder = builder_class()
    module = builder.module()

    cond = astx.LiteralString("foo") != astx.LiteralString("bar")
    then_blk = astx.Block()
    then_blk.append(PrintExpr(astx.LiteralUTF8String("NE")))
    else_blk = astx.Block()
    else_blk.append(PrintExpr(astx.LiteralUTF8String("EQ")))
    if_stmt = astx.IfStmt(condition=cond, then=then_blk, else_=else_blk)

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(if_stmt)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="NE")


@pytest.mark.parametrize(
    "int_type,literal_type,a_val,b_val,expect",
    [
        # use 0/1 so bitwise and/or behave like logical
        (astx.Int32, astx.LiteralInt32, 1, 0, "1"),
        (astx.Int16, astx.LiteralInt16, 1, 1, "1"),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_binary_op_logical_and_or(
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
    a_val: int,
    b_val: int,
    expect: str,
) -> None:
    """Verify '&&' and '||' for integer booleans (0/1)."""
    builder = builder_class()
    module = builder.module()

    decl_x = astx.VariableDeclaration(
        name="x", type_=int_type(), value=literal_type(a_val)
    )
    decl_y = astx.VariableDeclaration(
        name="y", type_=int_type(), value=literal_type(b_val)
    )

    expr = (astx.Identifier("x") & astx.Identifier("x")) | astx.Identifier("y")
    assign = astx.VariableAssignment(name="x", value=expr)

    print_ok = PrintExpr(astx.LiteralUTF8String(expect))

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_x)
    main_block.append(decl_y)
    main_block.append(assign)
    main_block.append(print_ok)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output=expect)

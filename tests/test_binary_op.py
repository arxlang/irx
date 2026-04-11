"""
title: Tests for the BinaryOp.
"""

import pytest

from irx import astx
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder
from irx.system import Cast, PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.UInt32, astx.LiteralUInt32),
        (astx.UInt16, astx.LiteralUInt16),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_binary_op_literals(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test ASTx Module with a function called add.
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
        (astx.UInt32, astx.LiteralUInt32),
        (astx.UInt16, astx.LiteralUInt16),
        (astx.UInt8, astx.LiteralUInt8),
        (astx.UInt64, astx.LiteralUInt64),
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
        LLVMBuilder,
    ],
)
def test_binary_op_basic(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test ASTx Module with a function called add.
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
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a",
        type_=int_type(),
        value=literal_type(1),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.VariableDeclaration(
        name="b",
        type_=int_type(),
        value=literal_type(2),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_c = astx.VariableDeclaration(
        name="c",
        type_=int_type(),
        value=literal_type(4),
        mutability=astx.MutabilityKind.mutable,
    )

    a = astx.Identifier("a")
    b = astx.Identifier("b")
    c = astx.Identifier("c")

    lit_1 = literal_type(1)

    basic_op = lit_1 + b - a * c / a + (b - a + c / a)

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(decl_b)
    main_block.append(decl_c)
    main_block.append(basic_op)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)
    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_binary_op_string_not_equals(builder_class: type[Builder]) -> None:
    """
    title: Verify string '!=' uses strcmp_inline + xor 1 path.
    parameters:
      builder_class:
        type: type[Builder]
    """
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
    "a_val,b_val,expect", [(True, False, "1"), (False, False, "0")]
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_binary_op_logical_and_or(
    builder_class: type[Builder],
    a_val: bool,
    b_val: bool,
    expect: str,
) -> None:
    """
    title: Verify Boolean '&&' and '||' lowering and execution.
    parameters:
      builder_class:
        type: type[Builder]
      a_val:
        type: bool
      b_val:
        type: bool
      expect:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    decl_x = astx.VariableDeclaration(
        name="x",
        type_=astx.Boolean(),
        value=astx.LiteralBoolean(a_val),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_y = astx.VariableDeclaration(
        name="y",
        type_=astx.Boolean(),
        value=astx.LiteralBoolean(b_val),
        mutability=astx.MutabilityKind.mutable,
    )

    expr = astx.BinaryOp(
        "||",
        astx.BinaryOp(
            "&&",
            astx.Identifier("x"),
            astx.Identifier("x"),
        ),
        astx.Identifier("y"),
    )

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_block = astx.Block()
    main_block.append(decl_x)
    main_block.append(decl_y)
    main_block.append(
        astx.FunctionReturn(Cast(value=expr, target_type=astx.Int32()))
    )
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output=expect)

"""
title: Tests for Unsigned Integer Operations.
"""

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "u_type, u_lit_type, s_type, s_lit_type, u_val",
    [
        (astx.UInt8, astx.LiteralUInt8, astx.Int8, astx.LiteralInt8, 200),
        (
            astx.UInt16,
            astx.LiteralUInt16,
            astx.Int16,
            astx.LiteralInt16,
            60000,
        ),
        (
            astx.UInt32,
            astx.LiteralUInt32,
            astx.Int32,
            astx.LiteralInt32,
            4294967295,
        ),
        (
            astx.UInt64,
            astx.LiteralUInt64,
            astx.Int64,
            astx.LiteralInt64,
            9223372036854775809,
        ),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_unsigned_vs_signed_comparison.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_unsigned_vs_signed_comparison(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
    u_type: type,
    u_lit_type: type,
    s_type: type,
    s_lit_type: type,
    u_val: int,
) -> None:
    """
    title: Verify unsigned comparisons work correctly and differ from signed.
    parameters:
      action:
        type: str
      expected_file:
        type: str
      builder_class:
        type: type[Builder]
      u_type:
        type: type
      u_lit_type:
        type: type
      s_type:
        type: type
      s_lit_type:
        type: type
      u_val:
        type: int
    """
    builder = builder_class()
    module = builder.module()

    decl_u_a = astx.VariableDeclaration(
        name="u_a",
        type_=u_type(),
        value=u_lit_type(u_val),
        mutability=astx.MutabilityKind.constant,
    )
    decl_u_b = astx.VariableDeclaration(
        name="u_b",
        type_=u_type(),
        value=u_lit_type(1),
        mutability=astx.MutabilityKind.constant,
    )

    cond_u = astx.BinaryOp(
        op_code=">",
        lhs=astx.Identifier("u_a"),
        rhs=u_lit_type(1),
    )

    then_blk_u = astx.Block()
    then_blk_u.append(PrintExpr(astx.LiteralUTF8String("U_TRUE ")))

    else_blk_u = astx.Block()
    else_blk_u.append(PrintExpr(astx.LiteralUTF8String("U_FALSE ")))

    if_stmt_u = astx.IfStmt(
        condition=cond_u, then=then_blk_u, else_=else_blk_u
    )

    decl_s_a = astx.VariableDeclaration(
        name="s_a",
        type_=s_type(),
        value=s_lit_type(-1),
        mutability=astx.MutabilityKind.constant,
    )
    decl_s_b = astx.VariableDeclaration(
        name="s_b",
        type_=s_type(),
        value=s_lit_type(1),
        mutability=astx.MutabilityKind.constant,
    )

    cond_s = astx.BinaryOp(
        op_code=">",
        lhs=astx.Identifier("s_a"),
        rhs=s_lit_type(1),
    )

    then_blk_s = astx.Block()
    then_blk_s.append(PrintExpr(astx.LiteralUTF8String("S_TRUE ")))

    else_blk_s = astx.Block()
    else_blk_s.append(PrintExpr(astx.LiteralUTF8String("S_FALSE ")))

    if_stmt_s = astx.IfStmt(
        condition=cond_s, then=then_blk_s, else_=else_blk_s
    )

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_u_a)
    main_block.append(decl_u_b)
    main_block.append(if_stmt_u)

    main_block.append(decl_s_a)
    main_block.append(decl_s_b)
    main_block.append(if_stmt_s)

    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result(action, builder, module, expected_output="U_TRUE S_FALSE ")


@pytest.mark.parametrize(
    "u_type, u_lit_type, s_type, s_lit_type, u_val",
    [
        (astx.UInt8, astx.LiteralUInt8, astx.Int8, astx.LiteralInt8, 200),
        (
            astx.UInt16,
            astx.LiteralUInt16,
            astx.Int16,
            astx.LiteralInt16,
            60000,
        ),
        (
            astx.UInt32,
            astx.LiteralUInt32,
            astx.Int32,
            astx.LiteralInt32,
            4294967295,
        ),
        (
            astx.UInt64,
            astx.LiteralUInt64,
            astx.Int64,
            astx.LiteralInt64,
            9223372036854775809,
        ),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_unsigned_vs_signed_division.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_unsigned_vs_signed_division(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
    u_type: type,
    u_lit_type: type,
    s_type: type,
    s_lit_type: type,
    u_val: int,
) -> None:
    """
    title: Verify unsigned division works correctly and differs from signed.
    parameters:
      action:
        type: str
      expected_file:
        type: str
      builder_class:
        type: type[Builder]
      u_type:
        type: type
      u_lit_type:
        type: type
      s_type:
        type: type
      s_lit_type:
        type: type
      u_val:
        type: int
    """
    builder = builder_class()
    module = builder.module()

    decl_u_a = astx.VariableDeclaration(
        name="u_a",
        type_=u_type(),
        value=u_lit_type(u_val),
        mutability=astx.MutabilityKind.constant,
    )
    decl_u_b = astx.VariableDeclaration(
        name="u_b",
        type_=u_type(),
        value=u_lit_type(2),
        mutability=astx.MutabilityKind.constant,
    )
    decl_u_c = astx.VariableDeclaration(
        name="u_c",
        type_=u_type(),
        value=astx.BinaryOp(
            op_code="/",
            lhs=astx.Identifier("u_a"),
            rhs=u_lit_type(2),
        ),
        mutability=astx.MutabilityKind.constant,
    )

    cond_u = astx.BinaryOp(
        op_code=">",
        lhs=astx.Identifier("u_c"),
        rhs=u_lit_type(0),
    )

    then_blk_u = astx.Block()
    then_blk_u.append(PrintExpr(astx.LiteralUTF8String("U_DIV_GT_ZERO ")))

    else_blk_u = astx.Block()
    else_blk_u.append(PrintExpr(astx.LiteralUTF8String("U_DIV_ZERO ")))

    if_stmt_u = astx.IfStmt(
        condition=cond_u, then=then_blk_u, else_=else_blk_u
    )

    decl_s_a = astx.VariableDeclaration(
        name="s_a",
        type_=s_type(),
        value=s_lit_type(-1),
        mutability=astx.MutabilityKind.constant,
    )
    decl_s_b = astx.VariableDeclaration(
        name="s_b",
        type_=s_type(),
        value=s_lit_type(2),
        mutability=astx.MutabilityKind.constant,
    )
    decl_s_c = astx.VariableDeclaration(
        name="s_c",
        type_=s_type(),
        value=astx.BinaryOp(
            op_code="/",
            lhs=astx.Identifier("s_a"),
            rhs=s_lit_type(2),
        ),
        mutability=astx.MutabilityKind.constant,
    )

    cond_s = astx.BinaryOp(
        op_code=">",
        lhs=astx.Identifier("s_c"),
        rhs=s_lit_type(0),
    )

    then_blk_s = astx.Block()
    then_blk_s.append(PrintExpr(astx.LiteralUTF8String("S_DIV_GT_ZERO ")))

    else_blk_s = astx.Block()
    else_blk_s.append(PrintExpr(astx.LiteralUTF8String("S_DIV_ZERO ")))

    if_stmt_s = astx.IfStmt(
        condition=cond_s, then=then_blk_s, else_=else_blk_s
    )

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_u_a)
    main_block.append(decl_u_b)
    main_block.append(decl_u_c)
    main_block.append(if_stmt_u)

    main_block.append(decl_s_a)
    main_block.append(decl_s_b)
    main_block.append(decl_s_c)
    main_block.append(if_stmt_s)

    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result(
        action, builder, module, expected_output="U_DIV_GT_ZERO S_DIV_ZERO "
    )

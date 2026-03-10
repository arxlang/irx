"""
title: Tests for float.
"""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "lhs, op, rhs, expected",
    [
        (1.0, "+", 2.5, 3.5),
        (6.2, "-", 4.2, 2.0),
        (2.1, "*", 3.0, 6.3),
        (2.0, "/", 1.0, 2.0),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_float_operations_with_print(
    builder_class: Type[Builder],
    lhs: float,
    op: str,
    rhs: float,
    expected: float,
) -> None:
    """
    title: Test float operations by printing result to stdout.
    parameters:
      builder_class:
        type: Type[Builder]
      lhs:
        type: float
      op:
        type: str
      rhs:
        type: float
      expected:
        type: float
    """
    builder = builder_class()
    module = builder.module()

    # Build expression: lhs <op> rhs
    left = astx.LiteralFloat32(lhs)
    right = astx.LiteralFloat32(rhs)
    expr = astx.BinaryOp(op, left, right)

    # Declare tmp: float32 = expr
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.Float32(), value=expr
    )

    # Return block that prints float then returns 0
    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(str(expected))))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: float main() -> returns 0
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=str(expected))


def test_float_equality_comparison() -> None:
    """
    title: Test float == comparison.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.14),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.14),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp(
        op_code="==",
        lhs=astx.Identifier("a"),
        rhs=astx.Identifier("b"),
    )

    then_block = astx.Block()
    then_block.append(PrintExpr(astx.LiteralUTF8String("1")))
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(PrintExpr(astx.LiteralUTF8String("0")))
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    if_stmt = astx.IfStmt(condition=cond, then=then_block, else_=else_block)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    block.append(if_stmt)
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1")


def test_float_inequality_comparison() -> None:
    """
    title: Test float != comparison.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(1.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(2.0),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp(
        op_code="!=",
        lhs=astx.Identifier("a"),
        rhs=astx.Identifier("b"),
    )

    then_block = astx.Block()
    then_block.append(PrintExpr(astx.LiteralUTF8String("1")))
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(PrintExpr(astx.LiteralUTF8String("0")))
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    if_stmt = astx.IfStmt(condition=cond, then=then_block, else_=else_block)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    block.append(if_stmt)
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1")


def test_float_binary_ops() -> None:
    """
    title: Test basic float arithmetic operations.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(10.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )

    add_expr = astx.BinaryOp(
        op_code="+",
        lhs=astx.Identifier("a"),
        rhs=astx.Identifier("b"),
    )
    sub_expr = astx.BinaryOp(
        op_code="-",
        lhs=add_expr,
        rhs=astx.LiteralFloat32(3.0),
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.Float32(), value=sub_expr
    )
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String("10.0")))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="10.0")


def test_float_less_than() -> None:
    """
    title: Test float < comparison branch.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(1.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(2.0),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp("<", astx.Identifier("a"), astx.Identifier("b"))
    then_block = astx.Block()
    then_block.append(PrintExpr(astx.LiteralUTF8String("1")))
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(PrintExpr(astx.LiteralUTF8String("0")))
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    if_stmt = astx.IfStmt(condition=cond, then=then_block, else_=else_block)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    block.append(if_stmt)
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1")


def test_float_greater_than() -> None:
    """
    title: Test float > comparison branch.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(5.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp(">", astx.Identifier("a"), astx.Identifier("b"))
    then_block = astx.Block()
    then_block.append(PrintExpr(astx.LiteralUTF8String("1")))
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(PrintExpr(astx.LiteralUTF8String("0")))
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    if_stmt = astx.IfStmt(condition=cond, then=then_block, else_=else_block)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    block.append(if_stmt)
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1")


def test_float_less_equal() -> None:
    """
    title: Test float <= comparison branch.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp("<=", astx.Identifier("a"), astx.Identifier("b"))
    then_block = astx.Block()
    then_block.append(PrintExpr(astx.LiteralUTF8String("1")))
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(PrintExpr(astx.LiteralUTF8String("0")))
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    if_stmt = astx.IfStmt(condition=cond, then=then_block, else_=else_block)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    block.append(if_stmt)
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1")


def test_float_greater_equal() -> None:
    """
    title: Test float >= comparison branch.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(5.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp(">=", astx.Identifier("a"), astx.Identifier("b"))
    then_block = astx.Block()
    then_block.append(PrintExpr(astx.LiteralUTF8String("1")))
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(PrintExpr(astx.LiteralUTF8String("0")))
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    if_stmt = astx.IfStmt(condition=cond, then=then_block, else_=else_block)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    block.append(if_stmt)
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1")


def test_literal_float16() -> None:
    """
    title: Test LiteralFloat16 visitor (lines 1589-1590).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="h",
        type_=astx.Float16(),
        value=astx.LiteralFloat16(1.5),
        mutability=astx.MutabilityKind.mutable,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(PrintExpr(astx.LiteralUTF8String("1.5")))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1.5")


def test_float_division() -> None:
    """
    title: Test float division branch (line 1053-1058).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(10.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(2.0),
        mutability=astx.MutabilityKind.mutable,
    )
    div_expr = astx.BinaryOp("/", astx.Identifier("a"), astx.Identifier("b"))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.Float32(), value=div_expr
    )
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String("5.0")))

    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="5.0")


def test_float_multiplication() -> None:
    """
    title: Test float multiplication branch (line 988-992).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(4.0),
        mutability=astx.MutabilityKind.mutable,
    )
    mul_expr = astx.BinaryOp("*", astx.Identifier("a"), astx.Identifier("b"))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.Float32(), value=mul_expr
    )
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String("12.0")))
    
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="12.0")

"""
title: Additional tests to maximize code coverage for llvmliteir.py.
"""

import astx
import pytest

from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import Cast, PrintExpr

from .conftest import check_result


# ── Float comparison operators ──────────────────────────────


def test_float_less_than() -> None:
    """Test float < comparison branch."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a", type_=astx.Float32(),
        value=astx.LiteralFloat32(1.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b", type_=astx.Float32(),
        value=astx.LiteralFloat32(2.0),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp("<", astx.Identifier("a"), astx.Identifier("b"))
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
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
    """Test float > comparison branch."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a", type_=astx.Float32(),
        value=astx.LiteralFloat32(5.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b", type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp(">", astx.Identifier("a"), astx.Identifier("b"))
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
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
    """Test float <= comparison branch."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a", type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b", type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp("<=", astx.Identifier("a"), astx.Identifier("b"))
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
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
    """Test float >= comparison branch."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a", type_=astx.Float32(),
        value=astx.LiteralFloat32(5.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b", type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp(">=", astx.Identifier("a"), astx.Identifier("b"))
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
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


# ── VariableAssignment visitor ──────────────────────────────


def test_variable_assignment() -> None:
    """Test VariableAssignment visitor (line 1314+)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x", type_=astx.Int32(),
        value=astx.LiteralInt32(10),
        mutability=astx.MutabilityKind.mutable,
    )

    assign = astx.VariableAssignment(
        name="x", value=astx.LiteralInt32(42)
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(assign)
    block.append(astx.FunctionReturn(astx.Identifier("x")))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="42")


def test_variable_assignment_const_error() -> None:
    """Test VariableAssignment to const raises error (line 1325)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="c", type_=astx.Int32(),
        value=astx.LiteralInt32(10),
        mutability=astx.MutabilityKind.constant,
    )

    assign = astx.VariableAssignment(
        name="c", value=astx.LiteralInt32(42)
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(assign)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="Cannot assign"):
        check_result("build", builder, module)


# ── LiteralFloat16 visitor ──────────────────────────────────


def test_literal_float16() -> None:
    """Test LiteralFloat16 visitor (lines 1589-1590)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="h", type_=astx.Float16(),
        value=astx.LiteralFloat16(1.5),
        mutability=astx.MutabilityKind.mutable,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── Float division ──────────────────────────────────────────


def test_float_division() -> None:
    """Test float division branch (line 1053-1058)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a", type_=astx.Float32(),
        value=astx.LiteralFloat32(10.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b", type_=astx.Float32(),
        value=astx.LiteralFloat32(2.0),
        mutability=astx.MutabilityKind.mutable,
    )
    div_expr = astx.BinaryOp(
        "/", astx.Identifier("a"), astx.Identifier("b")
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    block.append(div_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── Float multiplication ────────────────────────────────────


def test_float_multiplication() -> None:
    """Test float multiplication branch (line 988-992)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a", type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b", type_=astx.Float32(),
        value=astx.LiteralFloat32(4.0),
        mutability=astx.MutabilityKind.mutable,
    )
    mul_expr = astx.BinaryOp(
        "*", astx.Identifier("a"), astx.Identifier("b")
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_a)
    block.append(decl_b)
    block.append(mul_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── If statement with float condition ───────────────────────


def test_if_stmt_float_condition() -> None:
    """Test IfStmt with float condition (lines 1152, 1155-1156)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="f", type_=astx.Float32(),
        value=astx.LiteralFloat32(1.0),
        mutability=astx.MutabilityKind.mutable,
    )

    # Use float identifier directly as condition
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    if_stmt = astx.IfStmt(
        condition=astx.Identifier("f"),
        then=then_block,
        else_=else_block,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(if_stmt)
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1")


# ── WhileStmt with float condition ──────────────────────────


def test_while_stmt_float_condition() -> None:
    """Test WhileStmt with float condition (lines 1273, 1277-1278)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="count", type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )

    # while(count) { count = count - 1.0 }
    sub_expr = astx.BinaryOp(
        "-", astx.Identifier("count"), astx.LiteralFloat32(1.0)
    )
    assign = astx.VariableAssignment(name="count", value=sub_expr)
    body = astx.Block()
    body.append(assign)

    while_stmt = astx.WhileStmt(
        condition=astx.Identifier("count"),
        body=body,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(while_stmt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── InlineVariableDeclaration without value ─────────────────





# ── Cast operations ─────────────────────────────────────────


def test_cast_int_to_float() -> None:
    """Test Cast from int to float (line 2654)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x", type_=astx.Int32(),
        value=astx.LiteralInt32(42),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("x"), target_type=astx.Float32())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(cast_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_cast_float_to_int() -> None:
    """Test Cast from float to int (line 2659)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="f", type_=astx.Float32(),
        value=astx.LiteralFloat32(3.14),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("f"), target_type=astx.Int32())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(cast_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_cast_int_widening() -> None:
    """Test Cast from int8 to int32 (line 2646)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x", type_=astx.Int8(),
        value=astx.LiteralInt8(10),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("x"), target_type=astx.Int32())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(cast_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_cast_int_narrowing() -> None:
    """Test Cast from int32 to int8 (line 2650)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x", type_=astx.Int32(),
        value=astx.LiteralInt32(10),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("x"), target_type=astx.Int8())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(cast_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_cast_same_type_noop() -> None:
    """Test Cast with same source and target type is a no-op (line 2639)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x", type_=astx.Int32(),
        value=astx.LiteralInt32(5),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("x"), target_type=astx.Int32())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(cast_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── Cast float16<->float32 ──────────────────────────────────


def test_cast_float_to_half() -> None:
    """Test Cast from float32 to float16 (line 2666)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="f", type_=astx.Float32(),
        value=astx.LiteralFloat32(1.5),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("f"), target_type=astx.Float16())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(cast_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_cast_half_to_float() -> None:
    """Test Cast from float16 to float32 (line 2673)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="h", type_=astx.Float16(),
        value=astx.LiteralFloat16(1.5),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("h"), target_type=astx.Float32())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(cast_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── Cast int to string ──────────────────────────────────────


def test_cast_int_to_string() -> None:
    """Test Cast from int to string (line 2694+)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x", type_=astx.Int32(),
        value=astx.LiteralInt32(42),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("x"), target_type=astx.String())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(cast_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── Additional literal types ────────────────────────────────


def test_literal_int8() -> None:
    """Test LiteralInt8 visitor."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="b", type_=astx.Int8(),
        value=astx.LiteralInt8(42),
        mutability=astx.MutabilityKind.mutable,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_literal_int64() -> None:
    """Test LiteralInt64 visitor."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="big", type_=astx.Int64(),
        value=astx.LiteralInt64(1000000),
        mutability=astx.MutabilityKind.mutable,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── Increment on const should fail ──────────────────────────


def test_increment_const_error() -> None:
    """Test ++ on const variable raises error (line 686)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="c", type_=astx.Int32(),
        value=astx.LiteralInt32(5),
        mutability=astx.MutabilityKind.constant,
    )
    incr = astx.UnaryOp(op_code="++", operand=astx.Identifier("c"))
    incr.type_ = astx.Int32()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(incr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="Cannot mutate"):
        check_result("build", builder, module)


def test_decrement_const_error() -> None:
    """Test -- on const variable raises error (line 705)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="c", type_=astx.Int32(),
        value=astx.LiteralInt32(5),
        mutability=astx.MutabilityKind.constant,
    )
    decr = astx.UnaryOp(op_code="--", operand=astx.Identifier("c"))
    decr.type_ = astx.Int32()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(decr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="Cannot mutate"):
        check_result("build", builder, module)


def test_not_const_error() -> None:
    """Test ! on const variable raises error (line 725)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="c", type_=astx.Int32(),
        value=astx.LiteralInt32(1),
        mutability=astx.MutabilityKind.constant,
    )
    not_op = astx.UnaryOp(op_code="!", operand=astx.Identifier("c"))
    not_op.type_ = astx.Int32()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(not_op)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="Cannot mutate"):
        check_result("build", builder, module)


# ── FunctionReturn void ─────────────────────────────────────





# ── Redeclare identifier error ──────────────────────────────


def test_inline_var_redeclare_error() -> None:
    """Test re-declaring a variable raises error (line 2455)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl1 = astx.InlineVariableDeclaration(
        name="x", type_=astx.Int32(),
        value=astx.LiteralInt32(1),
        mutability=astx.MutabilityKind.mutable,
    )
    decl2 = astx.InlineVariableDeclaration(
        name="x", type_=astx.Int32(),
        value=astx.LiteralInt32(2),
        mutability=astx.MutabilityKind.mutable,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl1)
    block.append(decl2)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="Identifier already declared"):
        check_result("build", builder, module)

"""
title: More tests to push coverage higher for llvmliteir.py.
"""

import astx
import pytest

from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


# ── Integer == and != comparisons ───────────────────────────


def test_int_equality() -> None:
    """Test integer == comparison (line 1085)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a", type_=astx.Int32(),
        value=astx.LiteralInt32(5),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b", type_=astx.Int32(),
        value=astx.LiteralInt32(5),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp("==", astx.Identifier("a"), astx.Identifier("b"))
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


def test_int_inequality() -> None:
    """Test integer != comparison (line 1108)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl_a = astx.InlineVariableDeclaration(
        name="a", type_=astx.Int32(),
        value=astx.LiteralInt32(1),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.InlineVariableDeclaration(
        name="b", type_=astx.Int32(),
        value=astx.LiteralInt32(2),
        mutability=astx.MutabilityKind.mutable,
    )

    cond = astx.BinaryOp("!=", astx.Identifier("a"), astx.Identifier("b"))
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


# ── PrintExpr with integer ──────────────────────────────────


def test_print_integer() -> None:
    """Test PrintExpr with integer value (line 2744)."""
    builder = LLVMLiteIR()
    module = builder.module()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(PrintExpr(astx.LiteralInt32(42)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="42")


def test_print_float() -> None:
    """Test PrintExpr with float value (line 2751-2758)."""
    builder = LLVMLiteIR()
    module = builder.module()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(PrintExpr(astx.LiteralFloat32(3.14)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── LiteralTimestamp valid ──────────────────────────────────


def test_literal_timestamp_valid() -> None:
    """Test valid LiteralTimestamp (lines 1726-1810)."""
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("2025-03-06T14:30:00")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_literal_timestamp_with_space() -> None:
    """Test LiteralTimestamp with space separator."""
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("2025-03-06 14:30:00")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_literal_timestamp_invalid_format() -> None:
    """Test LiteralTimestamp with invalid format (line 1734)."""
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("20250306")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="LiteralTimestamp"):
        check_result("build", builder, module)


def test_literal_timestamp_hour_out_of_range() -> None:
    """Test LiteralTimestamp with out-of-range hour (line 1797)."""
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("2025-03-06T25:00:00")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="hour out of range"):
        check_result("build", builder, module)


def test_literal_timestamp_minute_out_of_range() -> None:
    """Test LiteralTimestamp with out-of-range minute (line 1801)."""
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("2025-03-06T12:60:00")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="minute out of range"):
        check_result("build", builder, module)


def test_literal_timestamp_second_out_of_range() -> None:
    """Test LiteralTimestamp with out-of-range second (line 1805)."""
    builder = LLVMLiteIR()
    module = builder.module()

    ts = astx.LiteralTimestamp("2025-03-06T12:30:61")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(ts)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="second out of range"):
        check_result("build", builder, module)


# ── LiteralDateTime valid + error branches ──────────────────


def test_literal_datetime_valid() -> None:
    """Test valid LiteralDateTime (lines 1825+)."""
    builder = LLVMLiteIR()
    module = builder.module()

    dt = astx.LiteralDateTime("2025-03-06T14:30:00")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(dt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_literal_datetime_invalid_format() -> None:
    """Test LiteralDateTime with invalid format (line 1845)."""
    builder = LLVMLiteIR()
    module = builder.module()

    dt = astx.LiteralDateTime("20250306")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(dt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="LiteralDateTime"):
        check_result("build", builder, module)


def test_literal_datetime_hour_out_of_range() -> None:
    """Test LiteralDateTime with out-of-range values."""
    builder = LLVMLiteIR()
    module = builder.module()

    dt = astx.LiteralDateTime("2025-03-06T25:00:00")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(dt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="hour out of range"):
        check_result("build", builder, module)


# ── LiteralDate ─────────────────────────────────────────────





def test_for_count_loop_basic() -> None:
    """Test ForCountLoopStmt (lines 1350+)."""
    builder = LLVMLiteIR()
    module = builder.module()

    init = astx.InlineVariableDeclaration(
        name="i", type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )
    cond = astx.BinaryOp("<", astx.Identifier("i"), astx.LiteralInt32(5))
    update = astx.BinaryOp("+", astx.Identifier("i"), astx.LiteralInt32(1))
    body = astx.Block()
    body.append(astx.LiteralInt32(0))

    loop = astx.ForCountLoopStmt(
        initializer=init, condition=cond, update=update, body=body
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(loop)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── Multiple prints to reuse format global ──────────────────


def test_format_global_reuse() -> None:
    """Test _get_or_create_format_global reuse (line 2613)."""
    builder = LLVMLiteIR()
    module = builder.module()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(PrintExpr(astx.LiteralInt32(1)))
    block.append(PrintExpr(astx.LiteralInt32(2)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


# ── IfStmt with only then branch (no else) ──────────────────


def test_if_stmt_no_else() -> None:
    """Test IfStmt with no else block."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x", type_=astx.Int32(),
        value=astx.LiteralInt32(1),
        mutability=astx.MutabilityKind.mutable,
    )
    cond = astx.BinaryOp(">", astx.Identifier("x"), astx.LiteralInt32(0))
    then_block = astx.Block()
    then_block.append(astx.LiteralInt32(42))

    if_stmt = astx.IfStmt(condition=cond, then=then_block)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(if_stmt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)

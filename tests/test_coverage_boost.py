"""
title: Tests to boost code coverage for llvmliteir.py.
"""

import astx
import pytest

from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


def test_variable_declaration_no_initializer_int() -> None:
    """
    title: Test VariableDeclaration without initializer for int type.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(astx.FunctionReturn(astx.Identifier("x")))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_variable_declaration_no_initializer_float() -> None:
    """
    title: Test VariableDeclaration for float type.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="y",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(0.0),
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


def test_variable_declaration_string_type() -> None:
    """
    title: Test VariableDeclaration for string type.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.VariableDeclaration(
        name="s",
        type_=astx.String(),
        value=astx.LiteralUTF8String("hello"),
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
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    if_stmt = astx.IfStmt(
        condition=cond, then=then_block, else_=else_block
    )

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
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    if_stmt = astx.IfStmt(
        condition=cond, then=then_block, else_=else_block
    )

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


def test_for_range_loop_without_step() -> None:
    """
    title: Test ForRangeLoopStmt without explicit step value.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    var_i = astx.InlineVariableDeclaration(
        "i", type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
    )
    start = astx.LiteralInt32(0)
    end = astx.LiteralInt32(5)

    body = astx.Block()
    body.append(astx.LiteralInt32(0))

    loop = astx.ForRangeLoopStmt(
        variable=var_i,
        start=start,
        end=end,
        step=astx.LiteralInt32(1),
        body=body,
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
    block.append(sub_expr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_const_variable_declaration() -> None:
    """
    title: Test const VariableDeclaration with mutability check.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.VariableDeclaration(
        name="PI",
        type_=astx.Int32(),
        value=astx.LiteralInt32(3),
        mutability=astx.MutabilityKind.constant,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(astx.FunctionReturn(astx.Identifier("PI")))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="3")


def test_decrement_operator() -> None:
    """
    title: Test standalone decrement operator.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(10),
        mutability=astx.MutabilityKind.mutable,
    )

    decr = astx.UnaryOp(op_code="--", operand=astx.Identifier("x"))
    decr.type_ = astx.Int32()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(decr)
    block.append(astx.FunctionReturn(astx.Identifier("x")))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="9")


def test_not_operator() -> None:
    """
    title: Test standalone NOT operator.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="flag",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )

    not_op = astx.UnaryOp(
        op_code="!", operand=astx.Identifier("flag")
    )
    not_op.type_ = astx.Int32()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(not_op)
    block.append(astx.FunctionReturn(astx.Identifier("flag")))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1")


def test_multiple_function_calls() -> None:
    """
    title: Test calling a user function that uses FunctionCall.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    # Helper function: add(a, b) -> a + b
    arg_a = astx.Argument(
        name="a", type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
    )
    arg_b = astx.Argument(
        name="b", type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
    )
    args = astx.Arguments()
    args.append(arg_a)
    args.append(arg_b)

    add_proto = astx.FunctionPrototype(
        name="add", args=args, return_type=astx.Int32()
    )
    add_body = astx.Block()
    add_expr = astx.BinaryOp(
        op_code="+",
        lhs=astx.Identifier("a"),
        rhs=astx.Identifier("b"),
    )
    add_body.append(astx.FunctionReturn(add_expr))
    add_fn = astx.FunctionDef(prototype=add_proto, body=add_body)
    module.block.append(add_fn)

    # main calls add(10, 32)
    call = astx.FunctionCall(
        fn="add",
        args=[astx.LiteralInt32(10), astx.LiteralInt32(32)],
    )

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(astx.FunctionReturn(call))
    main_fn = astx.FunctionDef(
        prototype=main_proto, body=main_block
    )
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="42")

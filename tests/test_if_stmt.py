"""
title: Test If statements with and without else.
"""

import pytest

from irx import astx
from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
        (astx.Float32, astx.LiteralFloat32),
        (astx.Float64, astx.LiteralFloat64),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_if_stmt.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_if_else_stmt(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test an If statement with an else branch.
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

    init_a = astx.InlineVariableDeclaration(
        "a", type_=int_type(), value=literal_type(10)
    )

    cond = astx.BinaryOp(
        op_code=">", lhs=astx.Identifier("a"), rhs=literal_type(5)
    )

    then_block = astx.Block()
    then_block.append(PrintExpr(astx.LiteralUTF8String("then branch")))

    else_block = astx.Block()
    else_block.append(PrintExpr(astx.LiteralUTF8String("else branch")))

    if_stmt = astx.IfStmt(condition=cond, then=then_block, else_=else_block)

    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_body = astx.Block()
    main_body.append(init_a)
    main_body.append(if_stmt)
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    main_fn = astx.FunctionDef(prototype=main_proto, body=main_body)
    module.block.append(main_fn)

    check_result(
        action, builder, module, expected_file, expected_output="then branch"
    )


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Float32, astx.LiteralFloat32),
        (astx.Float64, astx.LiteralFloat64),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_if_only_stmt(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test an If statement without an else branch.
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

    init_a = astx.InlineVariableDeclaration(
        "a", type_=int_type(), value=literal_type(3)
    )

    cond = astx.BinaryOp(
        op_code=">", lhs=astx.Identifier("a"), rhs=literal_type(5)
    )

    then_block = astx.Block()
    then_block.append(PrintExpr(astx.LiteralUTF8String("only then branch")))

    if_stmt = astx.IfStmt(condition=cond, then=then_block)

    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_body = astx.Block()
    main_body.append(init_a)
    main_body.append(if_stmt)
    main_body.append(PrintExpr(astx.LiteralUTF8String("done")))
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    main_fn = astx.FunctionDef(prototype=main_proto, body=main_body)
    module.block.append(main_fn)

    check_result(
        action, builder, module, expected_file, expected_output="done"
    )


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_if_both_branches_return(builder_class: type[Builder]) -> None:
    """
    title: Test IfStmt when both branches terminate with return.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    choose_proto = astx.FunctionPrototype(
        "choose",
        args=astx.Arguments(astx.Argument("x", astx.Int32())),
        return_type=astx.Int32(),
    )
    choose_body = astx.Block()

    cond = astx.BinaryOp(
        op_code="<",
        lhs=astx.Identifier("x"),
        rhs=astx.LiteralInt32(3),
    )
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(2)))
    choose_body.append(
        astx.IfStmt(condition=cond, then=then_block, else_=else_block)
    )

    choose_fn = astx.FunctionDef(prototype=choose_proto, body=choose_body)
    module.block.append(choose_fn)

    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_body = astx.Block()
    main_body.append(
        astx.FunctionReturn(
            astx.FunctionCall("choose", [astx.LiteralInt32(1)])
        )
    )
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_body)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="1")


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_if_one_branch_returns_other_falls_through(
    builder_class: type[Builder],
) -> None:
    """
    title: Test IfStmt when one branch returns and the other continues.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    branchy_proto = astx.FunctionPrototype(
        "branchy",
        args=astx.Arguments(astx.Argument("x", astx.Int32())),
        return_type=astx.Int32(),
    )
    branchy_body = astx.Block()

    cond = astx.BinaryOp(
        op_code="<",
        lhs=astx.Identifier("x"),
        rhs=astx.LiteralInt32(0),
    )
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(7)))
    else_block = astx.Block()
    else_block.append(
        astx.InlineVariableDeclaration(
            "y", type_=astx.Int32(), value=astx.LiteralInt32(5)
        )
    )
    branchy_body.append(
        astx.IfStmt(condition=cond, then=then_block, else_=else_block)
    )
    branchy_body.append(astx.FunctionReturn(astx.Identifier("y")))

    branchy_fn = astx.FunctionDef(prototype=branchy_proto, body=branchy_body)
    module.block.append(branchy_fn)

    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    call_neg = astx.FunctionCall("branchy", [astx.LiteralInt32(-1)])
    call_zero = astx.FunctionCall("branchy", [astx.LiteralInt32(0)])
    total = astx.BinaryOp(op_code="+", lhs=call_neg, rhs=call_zero)
    main_body = astx.Block()
    main_body.append(astx.FunctionReturn(total))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_body)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="12")


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_if_recursive_fibonacci_with_returning_branches(
    builder_class: type[Builder],
) -> None:
    """
    title: Test recursive fibonacci with IfStmt returning in both branches.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    fib_proto = astx.FunctionPrototype(
        "fib",
        args=astx.Arguments(astx.Argument("x", astx.Int32())),
        return_type=astx.Int32(),
    )
    fib_body = astx.Block()
    fib_fn = astx.FunctionDef(prototype=fib_proto, body=fib_body)

    cond = astx.BinaryOp(
        op_code="<",
        lhs=astx.Identifier("x"),
        rhs=astx.LiteralInt32(3),
    )
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))

    x_minus_one = astx.BinaryOp(
        op_code="-",
        lhs=astx.Identifier("x"),
        rhs=astx.LiteralInt32(1),
    )
    x_minus_two = astx.BinaryOp(
        op_code="-",
        lhs=astx.Identifier("x"),
        rhs=astx.LiteralInt32(2),
    )
    fib_n1 = astx.FunctionCall(fib_fn, [x_minus_one])
    fib_n2 = astx.FunctionCall(fib_fn, [x_minus_two])
    fib_sum = astx.BinaryOp(op_code="+", lhs=fib_n1, rhs=fib_n2)

    else_block = astx.Block()
    else_block.append(astx.FunctionReturn(fib_sum))
    fib_body.append(
        astx.IfStmt(condition=cond, then=then_block, else_=else_block)
    )
    module.block.append(fib_fn)

    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_body = astx.Block()
    main_body.append(
        astx.FunctionReturn(astx.FunctionCall("fib", [astx.LiteralInt32(10)]))
    )
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_body)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="55")

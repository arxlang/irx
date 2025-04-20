"""Test If statements with and without else."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


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
        LLVMLiteIR,
    ],
)
def test_if_else_stmt(
    action: str, expected_file: str, builder_class: Type[Builder]
) -> None:
    """Test an If statement with an else branch."""
    builder = builder_class()
    module = builder.module()

    init_a = astx.InlineVariableDeclaration(
        "a", type_=astx.Int32(), value=astx.LiteralInt32(10)
    )

    cond = astx.BinaryOp(
        op_code=">", lhs=astx.Variable("a"), rhs=astx.LiteralInt32(5)
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

    main_fn = astx.Function(prototype=main_proto, body=main_body)
    module.block.append(main_fn)

    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize(
    "action,expected_file",
    [
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_if_only_stmt(
    action: str, expected_file: str, builder_class: Type[Builder]
) -> None:
    """Test an If statement without an else branch."""
    builder = builder_class()
    module = builder.module()

    init_a = astx.InlineVariableDeclaration(
        "a", type_=astx.Int32(), value=astx.LiteralInt32(3)
    )

    cond = astx.BinaryOp(
        op_code=">", lhs=astx.Variable("a"), rhs=astx.LiteralInt32(5)
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
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    main_fn = astx.Function(prototype=main_proto, body=main_body)
    module.block.append(main_fn)

    check_result(action, builder, module, expected_file)

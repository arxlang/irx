"""Test If statements."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


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
def test_if_stmt(
    action: str, expected_file: str, builder_class: Type[Builder]
) -> None:
    """Test the If statement."""
    builder = builder_class()

    init_a = astx.InlineVariableDeclaration(
        "a", type_=astx.Int32(), value=astx.LiteralInt32(10)
    )

    var_a = astx.Variable("a")
    cond = astx.BinaryOp(op_code=">", lhs=var_a, rhs=astx.LiteralInt32(5))

    then_block = astx.Block()
    then_block.append(astx.LiteralInt32(1))

    else_block = astx.Block()
    else_block.append(astx.LiteralInt32(0))

    if_stmt = astx.IfStmt(condition=cond, then=then_block, else_=else_block)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )

    fn_block = astx.Block()
    fn_block.append(init_a)
    fn_block.append(if_stmt)

    fn_main = astx.Function(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)

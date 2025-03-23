"""Test For Loop statements."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_for_range.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_for_range(
    action: str, expected_file: str, builder_class: Type[Builder]
) -> None:
    """Test For Range statement."""
    builder = builder_class()

    # `for` statement
    var_a = astx.InlineVariableDeclaration(
        "a", type_=astx.Int32(), value=astx.LiteralInt32(-1)
    )
    start = astx.LiteralInt32(1)
    end = astx.LiteralInt32(10)
    step = astx.LiteralInt32(1)
    body = astx.Block()
    body.append(astx.LiteralInt32(2))
    for_loop = astx.ForRangeLoopStmt(
        variable=var_a,
        start=start,
        end=end,
        step=step,
        body=body,
    )

    # main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(for_loop)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn_main = astx.Function(prototype=proto, body=block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", ""),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_for_count(
    action: str, expected_file: str, builder_class: Type[Builder]
) -> None:
    """Test the For Count statement."""
    builder = builder_class()

    # NOTE: it seems that the systable in the tests is not correctly
    # sanitized, the variable `a` was renamed to `a2`
    init_a = astx.InlineVariableDeclaration(
        "a2", type_=astx.Int32(), value=astx.LiteralInt32(0)
    )
    var_a = astx.Variable("a2")
    cond = astx.BinaryOp(op_code="<", lhs=var_a, rhs=astx.LiteralInt32(10))
    update = astx.UnaryOp(op_code="++", operand=var_a)
    step = astx.LiteralInt32(1)  #example step value 

    for_body = astx.Block()
    for_body.append(astx.LiteralInt32(2))
    for_loop = astx.ForCountLoopStmt(
        initializer=init_a,
        condition=cond,
        update=update,
        body=for_body,
        step = step
    )

    # main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn_block = astx.Block()
    fn_block.append(for_loop)
    fn_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn_main = astx.Function(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    # note: not yet fully implemented
    check_result(action, builder, module, expected_file)

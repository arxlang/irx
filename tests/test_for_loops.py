"""
title: Test For Loop statements.
"""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
    ],
)
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
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test For Range statement.
    parameters:
      action:
        type: str
      expected_file:
        type: str
      builder_class:
        type: Type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()

    # `for` statement
    var_a = astx.InlineVariableDeclaration(
        "a", type_=int_type(), mutability=astx.MutabilityKind.mutable
    )
    start = literal_type(1)
    end = literal_type(10)
    step = literal_type(1)
    body = astx.Block()
    body.append(literal_type(2))
    for_loop = astx.ForRangeLoopStmt(
        variable=var_a,
        start=start,
        end=end,
        step=step,
        body=body,
    )

    # main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    block = astx.Block()
    block.append(for_loop)
    block.append(astx.FunctionReturn(literal_type(0)))
    fn_main = astx.FunctionDef(prototype=proto, body=block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize(
    "float_type, literal_type",
    [
        (astx.Float32, astx.LiteralFloat32),
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
        LLVMLiteIR,
    ],
)
def test_for_range_float(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    float_type: type,
    literal_type: type,
) -> None:
    """
    title: Test For Range statement with floating-point types.
    """
    builder = builder_class()

    # `for` statement
    var_a = astx.InlineVariableDeclaration(
        "a", type_=float_type(), mutability=astx.MutabilityKind.mutable
    )
    start = literal_type(1.0)
    end = literal_type(10.0)
    step = literal_type(1.0)
    body = astx.Block()
    body.append(literal_type(2.0))
    for_loop = astx.ForRangeLoopStmt(
        variable=var_a,
        start=start,
        end=end,
        step=step,
        body=body,
    )

    # main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=float_type()
    )
    block = astx.Block()
    block.append(for_loop)
    block.append(astx.FunctionReturn(literal_type(0.0)))
    fn_main = astx.FunctionDef(prototype=proto, body=block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int64, astx.LiteralInt64),
        (astx.Int8, astx.LiteralInt8),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_for_loops.ll"),
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
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test the For Count statement.
    parameters:
      action:
        type: str
      expected_file:
        type: str
      builder_class:
        type: Type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()

    init_a = astx.InlineVariableDeclaration(
        "a2",
        type_=int_type(),
        value=literal_type(0),
        mutability=astx.MutabilityKind.mutable,
    )
    var_a = astx.Identifier("a2")
    cond = astx.BinaryOp(op_code="<", lhs=var_a, rhs=literal_type(10))
    update = astx.UnaryOp(op_code="++", operand=var_a)

    for_body = astx.Block()
    for_body.append(literal_type(2))
    for_loop = astx.ForCountLoopStmt(
        initializer=init_a,
        condition=cond,
        update=update,
        body=for_body,
    )

    # main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    fn_block = astx.Block()
    fn_block.append(for_loop)
    fn_block.append(astx.FunctionReturn(literal_type(0)))
    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)

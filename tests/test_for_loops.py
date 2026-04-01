"""
title: Test For Loop statements.
"""

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
        (astx.Float32, astx.LiteralFloat32),
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
    builder_class: type[Builder],
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
        type: type[Builder]
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
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_for_range_empty_body_no_crash(
    builder_class: type[Builder],
) -> None:
    """
    title: ForRangeLoopStmt with empty body must not crash on empty stack.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    variable = astx.InlineVariableDeclaration(
        "a",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
    )
    for_loop = astx.ForRangeLoopStmt(
        variable=variable,
        start=astx.LiteralInt32(1),
        end=astx.LiteralInt32(10),
        step=astx.LiteralInt32(1),
        body=astx.Block(),
    )

    proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    block = astx.Block()
    block.append(for_loop)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn_main = astx.FunctionDef(prototype=proto, body=block)

    module = builder.module()
    module.block.append(fn_main)

    check_result("build", builder, module, "")


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
    builder_class: type[Builder],
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
        type: type[Builder]
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


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_for_count_empty_body_no_crash(
    builder_class: type[Builder],
) -> None:
    """
    title: ForCountLoopStmt with empty body must not crash on empty stack.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    initializer = astx.InlineVariableDeclaration(
        "a2",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )
    variable = astx.Identifier("a2")
    for_loop = astx.ForCountLoopStmt(
        initializer=initializer,
        condition=astx.BinaryOp(
            op_code="<",
            lhs=variable,
            rhs=astx.LiteralInt32(10),
        ),
        update=astx.UnaryOp(op_code="++", operand=variable),
        body=astx.Block(),
    )

    proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    fn_block = astx.Block()
    fn_block.append(for_loop)
    fn_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    check_result("build", builder, module, "")

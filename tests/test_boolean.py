"""
title: Tests for Boolean logic and comparisons.
"""

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder

from .conftest import check_result


@pytest.mark.parametrize(
    "lhs,op,rhs,expected",
    [
        (True, "&&", True, "1"),
        (True, "||", False, "1"),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_boolean_operations(
    builder_class: type[Builder],
    lhs: bool,
    op: str,
    rhs: bool,
    expected: str,
) -> None:
    """
    title: Test literal Boolean AND/OR operations.
    parameters:
      builder_class:
        type: type[Builder]
      lhs:
        type: bool
      op:
        type: str
      rhs:
        type: bool
      expected:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    # build the ASTx expression: (lhs && rhs) or (lhs || rhs)
    left = astx.LiteralBoolean(lhs)
    right = astx.LiteralBoolean(rhs)
    expr = astx.BinaryOp(op, left, right)

    proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Boolean(),
    )
    block = astx.Block()
    block.append(astx.FunctionReturn(expr))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)


@pytest.mark.parametrize(
    "num_type,literal_type",
    [
        (astx.Int8, astx.LiteralInt8),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int32, astx.LiteralInt32),
        (astx.Int64, astx.LiteralInt64),
        (astx.Float32, astx.LiteralFloat32),
        (astx.Float64, astx.LiteralFloat64),
    ],
)
@pytest.mark.parametrize(
    "lhs,op,rhs,expected",
    [
        (1, "<", 2, "1"),
        (6, ">=", 6, "1"),
        (1, "==", 1, "1"),
        (1, "!=", 2, "1"),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_boolean_comparison(
    builder_class: type[Builder],
    num_type: type,
    literal_type: type,
    lhs: int,
    op: str,
    rhs: int,
    expected: str,
) -> None:
    """
    title: Test numeric comparisons for integers and floats.
    parameters:
      builder_class:
        type: type[Builder]
      num_type:
        type: type
      literal_type:
        type: type
      lhs:
        type: int
      op:
        type: str
      rhs:
        type: int
      expected:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    # build e.g. LiteralInt32(1) < LiteralInt32(2)
    left = literal_type(lhs)
    right = literal_type(rhs)
    expr = astx.BinaryOp(op, left, right)

    proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Boolean(),
    )
    block = astx.Block()
    block.append(astx.FunctionReturn(expr))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)

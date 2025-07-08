"""Tests for Boolean logic and comparisons."""

import subprocess

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

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
        LLVMLiteIR,
    ],
)
def test_boolean_operations(
    builder_class: Type[Builder],
    lhs: bool,
    op: str,
    rhs: bool,
    expected: str,
) -> None:
    """Test literal Boolean AND/OR operations."""
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

    success = True
    try:
        check_result("build", builder, module, expected_output=expected)
    except subprocess.CalledProcessError as e:
        success = False
        assert e.returncode == int(expected)
    assert not success


@pytest.mark.parametrize(
    "int_type,literal_type",
    [
        (astx.Int8, astx.LiteralInt8),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int32, astx.LiteralInt32),
        (astx.Int64, astx.LiteralInt64),
    ],
)
@pytest.mark.parametrize(
    "lhs,op,rhs,expected",
    [
        (1, "<", 2, "1"),
        (6, ">=", 6, "1"),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_boolean_comparison(
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
    lhs: int,
    op: str,
    rhs: int,
    expected: str,
) -> None:
    """Test integer comparisons."""
    builder = builder_class()
    module = builder.module()

    # build e.g. LiteralInt32(1) < LiteralInt32(2)
    left = literal_type(lhs)
    right = literal_type(rhs)
    expr = astx.BinaryOp(op, left, right)

    # wrap in a main() returning a  Boolean
    proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Boolean(),
    )
    block = astx.Block()
    block.append(astx.FunctionReturn(expr))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    success = True
    try:
        check_result("build", builder, module, expected_output=expected)
    except subprocess.CalledProcessError as e:
        success = False
        assert e.returncode == int(expected)
    assert not success

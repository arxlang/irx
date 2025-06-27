"""Test For VariableAssignment."""

import subprocess

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
    "builder_class",
    [
        # ("translate", "test_variable_assignment.ll"),
        LLVMLiteIR,
    ],
)
def test_variable_assignment(
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """Test VariableAssignment by reassigning and returning."""
    builder = builder_class()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x", type_=int_type(), value=literal_type(10)
    )
    assignment = astx.VariableAssignment(name="x", value=literal_type(42))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    fn_block = astx.Block()
    fn_block.append(decl)
    fn_block.append(assignment)
    fn_block.append(astx.FunctionReturn(astx.Variable("x")))
    fn_main = astx.Function(prototype=proto, body=fn_block)

    module.block.append(fn_main)

    expected_output = "42"
    success = True

    try:
        check_result("build", builder, module, expected_output=expected_output)
    except subprocess.CalledProcessError as e:
        success = False
        assert e.returncode == int(expected_output)

    assert not success

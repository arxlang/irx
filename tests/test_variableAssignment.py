"""Test For variableAssignemnt."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [(astx.Int32, astx.LiteralInt32), (astx.Int16, astx.LiteralInt16)],
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
def test_variable_assignment(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """Test VariableAssignment."""
    builder = builder_class()

    # Declare a variable and assign it later
    decl = astx.InlineVariableDeclaration(
        name="x", type_=int_type(), value=literal_type(10)
    )
    assignment = astx.VariableAssignment(name="x", value=literal_type(42))

    # main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    fn_block = astx.Block()
    fn_block.append(decl)
    fn_block.append(assignment)
    fn_block.append(astx.FunctionReturn(astx.Variable("x")))
    fn_main = astx.Function(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)

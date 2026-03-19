"""
title: Test For VariableAssignment.
"""

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "var_type, literal_type, expected_output",
    [
        (astx.Int32, astx.LiteralInt32, "42"),
        (astx.Int16, astx.LiteralInt16, "42"),
        (astx.Int8, astx.LiteralInt8, "42"),
        (astx.Int64, astx.LiteralInt64, "42"),
        (astx.Float32, astx.LiteralFloat32, "42.000000"),
        (astx.Float64, astx.LiteralFloat64, "42.000000"),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_variable_assignment(
    builder_class: type[Builder],
    var_type: type,
    literal_type: type,
    expected_output: str,
) -> None:
    """
    title: Test VariableAssignment by reassigning and returning.
    parameters:
      builder_class:
        type: type[Builder]
      var_type:
        type: type
      literal_type:
        type: type
      expected_output:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=var_type(),
        value=literal_type(10),
        mutability=astx.MutabilityKind.mutable,
    )
    assignment = astx.VariableAssignment(name="x", value=literal_type(42))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn_block = astx.Block()
    fn_block.append(decl)
    fn_block.append(assignment)
    fn_block.append(PrintExpr(astx.Identifier("x")))
    fn_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module.block.append(fn_main)

    check_result("build", builder, module, expected_output=expected_output)


@pytest.mark.parametrize(
    "var_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Float32, astx.LiteralFloat32),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_variable_declaration_no_initializer(
    builder_class: type[Builder],
    var_type: type,
    literal_type: type,
) -> None:
    """
    title: Test VariableDeclaration without an initializer defaults to zero.
    parameters:
      builder_class:
        type: type[Builder]
      var_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()
    module = builder.module()

    decl = astx.VariableDeclaration(
        name="x",
        type_=var_type(),
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=var_type()
    )
    fn_block = astx.Block()
    fn_block.append(decl)
    fn_block.append(astx.FunctionReturn(astx.Identifier("x")))
    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module.block.append(fn_main)

    check_result("build", builder, module)

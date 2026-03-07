"""Tests for StructType variable declarations."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import StructType

from .conftest import check_result


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_struct_variable_declaration(
    builder_class: Type[Builder],
) -> None:
    """Test struct variable declaration with defined struct."""
    builder = builder_class()
    module = builder.module()

    struct_def = astx.StructDefStmt(
        name="Point",
        attributes=[
            astx.VariableDeclaration("x", astx.Int32()),
            astx.VariableDeclaration("y", astx.Int32()),
        ],
    )

    struct_var = astx.VariableDeclaration(
        name="p",
        type_=StructType(struct_name="Point"),
    )

    block = astx.Block()
    block.append(struct_def)
    block.append(struct_var)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="0")


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_struct_variable_undefined_error(
    builder_class: Type[Builder],
) -> None:
    """Test error when struct variable references undefined struct."""
    builder = builder_class()
    module = builder.module()

    struct_var = astx.VariableDeclaration(
        name="p",
        type_=StructType(struct_name="Point"),
    )

    block = astx.Block()
    block.append(struct_var)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match=r"Struct 'Point' not defined"):
        builder.build(module, output_file="/tmp/test")


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_struct_variable_with_value_error(
    builder_class: Type[Builder],
) -> None:
    """Test error when struct variable has value initialization."""
    builder = builder_class()
    module = builder.module()

    struct_def = astx.StructDefStmt(
        name="Point",
        attributes=[
            astx.VariableDeclaration("x", astx.Int32()),
            astx.VariableDeclaration("y", astx.Int32()),
        ],
    )

    struct_var = astx.VariableDeclaration(
        name="p",
        type_=StructType(struct_name="Point"),
        value=astx.LiteralInt32(42),
    )

    block = astx.Block()
    block.append(struct_def)
    block.append(struct_var)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(
        Exception, match=r"Struct initialization with values not yet supported"
    ):
        builder.build(module, output_file="/tmp/test")

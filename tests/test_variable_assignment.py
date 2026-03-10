"""
title: Test For VariableAssignment.
"""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

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
    """
    title: Test VariableAssignment by reassigning and returning.
    parameters:
      builder_class:
        type: Type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=int_type(),
        value=literal_type(10),
        mutability=astx.MutabilityKind.mutable,
    )
    assignment = astx.VariableAssignment(name="x", value=literal_type(42))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    fn_block = astx.Block()
    fn_block.append(decl)
    fn_block.append(assignment)
    fn_block.append(astx.FunctionReturn(astx.Identifier("x")))
    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module.block.append(fn_main)

    expected_output = "42"
    check_result("build", builder, module, expected_output=expected_output)


def test_variable_declaration_no_initializer_int() -> None:
    """
    title: Test VariableDeclaration without initializer for int type.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(PrintExpr(astx.LiteralUTF8String("0")))
    block.append(astx.FunctionReturn(astx.Identifier("x")))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="0")


def test_variable_declaration_no_initializer_float() -> None:
    """
    title: Test VariableDeclaration for float type.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="y",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(0.0),
        mutability=astx.MutabilityKind.mutable,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(PrintExpr(astx.LiteralUTF8String("0.0")))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="0.0")


def test_variable_declaration_string_type() -> None:
    """
    title: Test VariableDeclaration for string type.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.VariableDeclaration(
        name="s",
        type_=astx.String(),
        value=astx.LiteralUTF8String("hello"),
        mutability=astx.MutabilityKind.mutable,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(PrintExpr(astx.LiteralUTF8String("hello")))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="hello")


def test_const_variable_declaration() -> None:
    """
    title: Test const VariableDeclaration with mutability check.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.VariableDeclaration(
        name="PI",
        type_=astx.Int32(),
        value=astx.LiteralInt32(3),
        mutability=astx.MutabilityKind.constant,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(astx.FunctionReturn(astx.Identifier("PI")))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="3")


def test_variable_assignment_extra() -> None:
    """
    title: Test VariableAssignment visitor (line 1314+).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(10),
        mutability=astx.MutabilityKind.mutable,
    )

    assign = astx.VariableAssignment(name="x", value=astx.LiteralInt32(42))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(assign)
    block.append(astx.FunctionReturn(astx.Identifier("x")))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="42")


def test_variable_assignment_const_error() -> None:
    """
    title: Test VariableAssignment to const raises error (line 1325).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="c",
        type_=astx.Int32(),
        value=astx.LiteralInt32(10),
        mutability=astx.MutabilityKind.constant,
    )

    assign = astx.VariableAssignment(name="c", value=astx.LiteralInt32(42))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(assign)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="Cannot assign"):
        check_result("build", builder, module)


def test_inline_var_redeclare_error() -> None:
    """
    title: Test re-declaring a variable raises error (line 2455).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl1 = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(1),
        mutability=astx.MutabilityKind.mutable,
    )
    decl2 = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(2),
        mutability=astx.MutabilityKind.mutable,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl1)
    block.append(decl2)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="Identifier already declared"):
        check_result("build", builder, module)

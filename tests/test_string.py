"""
title: Tests for string operations.
"""

from typing import Type, Any
from unittest.mock import MagicMock

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr
from llvmlite import ir

from .conftest import check_result


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_literal_utf8_with_print(
    builder_class: Type[Builder],
) -> None:
    """
    title: Test UTF-8 string literal by printing to stdout.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    expected = "Hello, World!"

    string_literal = astx.LiteralUTF8String(expected)

    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=string_literal
    )

    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_literal_utf8_char_with_print(
    builder_class: Type[Builder],
) -> None:
    """
    title: Test UTF-8 char literal by printing to stdout.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    expected = "A"

    char_literal = astx.LiteralUTF8Char(expected)

    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=char_literal
    )

    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_literal_generic_with_print(
    builder_class: Type[Builder],
) -> None:
    """
    title: Test generic string literal by printing to stdout.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    expected = "Generic String"

    string_literal = astx.LiteralString(expected)

    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=string_literal
    )

    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)


@pytest.mark.parametrize(
    "lhs_str, rhs_str, expected",
    [
        ("Hello, ", "World!", "Hello, World!"),
        ("", "Empty", "Empty"),
        ("123", "456", "123456"),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_concatenation_with_print(
    builder_class: Type[Builder],
    lhs_str: str,
    rhs_str: str,
    expected: str,
) -> None:
    """
    title: Test string concatenation by printing result to stdout.
    parameters:
      builder_class:
        type: Type[Builder]
      lhs_str:
        type: str
      rhs_str:
        type: str
      expected:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    left = astx.LiteralUTF8Char(lhs_str)
    right = astx.LiteralUTF8Char(rhs_str)
    expr = astx.BinaryOp("+", left, right)

    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=expr
    )

    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)


@pytest.mark.parametrize(
    "lhs_str, op, rhs_str, expected_result",
    [
        ("hello", "==", "hello", True),
        ("hello", "==", "world", False),
        ("test", "!=", "different", True),
        ("", "==", "", True),
        ("", "!=", "nonempty", True),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_comparison_with_print(
    builder_class: Type[Builder],
    lhs_str: str,
    op: str,
    rhs_str: str,
    expected_result: bool,
) -> None:
    """
    title: Test string comparison operations by printing result to stdout.
    parameters:
      builder_class:
        type: Type[Builder]
      lhs_str:
        type: str
      op:
        type: str
      rhs_str:
        type: str
      expected_result:
        type: bool
    """
    builder = builder_class()
    module = builder.module()

    left = astx.LiteralUTF8Char(lhs_str)
    right = astx.LiteralUTF8Char(rhs_str)
    expr = astx.BinaryOp(op, left, right)

    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.Boolean(), value=expr
    )

    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(str(expected_result))))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result(
        "build", builder, module, expected_output=str(expected_result)
    )


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_empty_string_with_print(
    builder_class: Type[Builder],
) -> None:
    """
    title: Test empty string by printing to stdout.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    expected = ""

    string_literal = astx.LiteralUTF8String(expected)

    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=string_literal
    )

    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String("EMPTY")))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="EMPTY")


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_with_special_characters_with_print(
    builder_class: Type[Builder],
) -> None:
    """
    title: Test string with special characters by printing to stdout.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    expected = 'Special: \\n\\t\\r"'

    string_literal = astx.LiteralUTF8String(expected)

    # Declare tmp: string = "Special: \\n\\t\\r\""
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=string_literal
    )

    # Return block that prints string with special chars then returns 0
    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)

def setup_builder() -> Any:
    main_builder = LLVMLiteIR()
    visitor = main_builder.translator
    func_type = ir.FunctionType(visitor._llvm.INT32_TYPE, [])
    fn = ir.Function(visitor._llvm.module, func_type, name="main")
    bb = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(bb)
    return visitor


def test_string_helper_functions() -> None:
    builder = setup_builder()

    # Call the creators
    concat_fn = builder._create_string_concat_function()
    assert concat_fn.name == "string_concat"
    # Call again to hit the cached branch
    assert builder._create_string_concat_function() is concat_fn

    len_fn = builder._create_string_length_function()
    assert len_fn.name == "string_length"
    assert builder._create_string_length_function() is len_fn

    eq_fn = builder._create_string_equals_function()
    assert eq_fn.name == "string_equals"
    assert builder._create_string_equals_function() is eq_fn

    sub_fn = builder._create_string_substring_function()
    assert sub_fn.name == "string_substring"
    assert builder._create_string_substring_function() is sub_fn


def test_handle_string_operations() -> None:
    builder = setup_builder()

    str1 = ir.Constant(builder._llvm.ASCII_STRING_TYPE, None)
    str2 = ir.Constant(builder._llvm.ASCII_STRING_TYPE, None)

    # This will insert the call in the current block
    res_concat = builder._handle_string_concatenation(str1, str2)
    assert res_concat is not None

    res_cmp_eq = builder._handle_string_comparison(str1, str2, "==")
    assert res_cmp_eq is not None

    res_cmp_neq = builder._handle_string_comparison(str1, str2, "!=")
    assert res_cmp_neq is not None

    with pytest.raises(Exception):
        builder._handle_string_comparison(str1, str2, "<")

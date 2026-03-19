"""
title: Tests for string operations.
"""

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_literal_utf8_with_print(
    builder_class: type[Builder],
) -> None:
    """
    title: Test UTF-8 string literal by printing to stdout.
    parameters:
      builder_class:
        type: type[Builder]
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
    builder_class: type[Builder],
) -> None:
    """
    title: Test UTF-8 char literal by printing to stdout.
    parameters:
      builder_class:
        type: type[Builder]
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
    builder_class: type[Builder],
) -> None:
    """
    title: Test generic string literal by printing to stdout.
    parameters:
      builder_class:
        type: type[Builder]
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
    builder_class: type[Builder],
    lhs_str: str,
    rhs_str: str,
    expected: str,
) -> None:
    """
    title: Test string concatenation by printing result to stdout.
    parameters:
      builder_class:
        type: type[Builder]
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
    builder_class: type[Builder],
    lhs_str: str,
    op: str,
    rhs_str: str,
    expected_result: bool,
) -> None:
    """
    title: Test string comparison operations by printing result to stdout.
    parameters:
      builder_class:
        type: type[Builder]
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
    builder_class: type[Builder],
) -> None:
    """
    title: Test empty string by printing to stdout.
    parameters:
      builder_class:
        type: type[Builder]
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
    builder_class: type[Builder],
) -> None:
    """
    title: Test string with special characters by printing to stdout.
    parameters:
      builder_class:
        type: type[Builder]
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


def test_utf8_char_lowering_correctness() -> None:
    """Verify LiteralUTF8Char correctly lowers to UTF-8 hex in IR."""

    builder = LLVMLiteIR()
    module = builder.module()

    # 'é' is represented as \xc3\xa9 in UTF-8
    char_node = astx.LiteralUTF8Char("é")

    block = astx.Block()
    block.append(
        astx.VariableDeclaration(
            name="tmp", type_=astx.String(), value=char_node
        )
    )
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    module.block.append(astx.FunctionDef(prototype=proto, body=block))

    ir_output = builder.translate(module)

    # Verify the UTF-8 hex sequence exists in the generated IR constant
    assert "\\c3\\a9" in ir_output.lower()

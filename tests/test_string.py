"""Tests for string operations."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_literal_utf8_with_print(
    builder_class: Type[Builder],
) -> None:
    """Test UTF-8 string literal by printing to stdout."""
    builder = builder_class()
    module = builder.module()

    expected = "Hello, World!"

    string_literal = astx.LiteralUTF8String(expected)

    # Declare tmp: string = "Hello, World!"
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=string_literal
    )

    # Return block that prints string then returns 0
    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: int main() -> returns 0
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
    """Test UTF-8 char literal by printing to stdout."""
    builder = builder_class()
    module = builder.module()

    expected = "A"

    # Create UTF-8 char literal
    char_literal = astx.LiteralUTF8Char(expected)

    # Declare tmp: string = "A"
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=char_literal
    )

    # Return block that prints char then returns 0
    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: int main() -> returns 0
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
    """Test generic string literal by printing to stdout."""
    builder = builder_class()
    module = builder.module()

    expected = "Generic String"

    # Create generic string literal
    string_literal = astx.LiteralString(expected)

    # Declare tmp: string = "Generic String"
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=string_literal
    )

    # Return block that prints string then returns 0
    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: int main() -> returns 0
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
        ("Test", "", "Test"),
        ("AB", "CD", "ABCD"),
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
    """Test string concatenation by printing result to stdout."""
    builder = builder_class()
    module = builder.module()

    # Build expression: lhs_str + rhs_str
    left = astx.LiteralUTF8Char(lhs_str)
    right = astx.LiteralUTF8Char(rhs_str)
    expr = astx.BinaryOp("+", left, right)

    # Declare tmp: string = expr
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=expr
    )

    # Return block that prints concatenated string then returns 0
    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: int main() -> returns 0
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
        ("test", "!=", "test", False),
        ("test", "!=", "different", True),
        ("ABC", "==", "ABC", True),
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
    """Test string comparison operations by printing result to stdout."""
    builder = builder_class()
    module = builder.module()

    # Build expression: lhs_str <op> rhs_str
    left = astx.LiteralUTF8Char(lhs_str)
    right = astx.LiteralUTF8Char(rhs_str)
    expr = astx.BinaryOp(op, left, right)

    # Declare tmp: boolean = expr
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.Boolean(), value=expr
    )

    # Return block that prints boolean result then returns 0
    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(str(expected_result))))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: int main() -> returns 0
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result(
        "build", builder, module, expected_output=str(expected_result)
    )


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_variable_assignment_with_print(
    builder_class: Type[Builder],
) -> None:
    """Test string variable assignment by printing to stdout."""
    builder = builder_class()
    module = builder.module()

    expected = "Assigned String"

    # Create string literal
    string_literal = astx.LiteralUTF8String(expected)

    # Declare variable: string my_string
    decl_var = astx.VariableDeclaration(name="my_string", type_=astx.String())

    # Assign: my_string = "Assigned String"
    assign_stmt = astx.VariableAssignment(
        name="my_string", value=string_literal
    )

    # Return block
    block = astx.Block()
    block.append(decl_var)
    block.append(assign_stmt)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: int main() -> returns 0
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_with_unicode_characters_with_print(
    builder_class: Type[Builder],
) -> None:
    """Test string with Unicode characters by printing to stdout."""
    builder = builder_class()
    module = builder.module()

    expected = "Hello 世界 🌍"

    # Create UTF-8 string with Unicode characters
    string_literal = astx.LiteralUTF8String(expected)

    # Declare tmp: string = "Hello 世界 🌍"
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=string_literal
    )

    # Return block that prints Unicode string then returns 0
    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.LiteralUTF8String(expected)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: int main() -> returns 0
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_empty_string_with_print(
    builder_class: Type[Builder],
) -> None:
    """Test empty string by printing to stdout."""
    builder = builder_class()
    module = builder.module()

    expected = ""

    # Create empty string literal
    string_literal = astx.LiteralUTF8String(expected)

    # Declare tmp: string = ""
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.String(), value=string_literal
    )

    # Return block that prints empty string then returns 0
    # Note: for empty string, we print a placeholder to verify the test
    block = astx.Block()
    block.append(decl_tmp)
    block.append(
        PrintExpr(astx.LiteralUTF8String("EMPTY"))
    )  # Placeholder for empty string
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: int main() -> returns 0
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
    """Test string with special characters by printing to stdout."""
    builder = builder_class()
    module = builder.module()

    expected = 'Special: \\n\\t\\r"'

    # Create string literal with special characters
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

    # Define: int main() -> returns 0
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_function_parameter_with_print(
    builder_class: Type[Builder],
) -> None:
    """Test string as function parameter by printing to stdout."""
    builder = builder_class()
    module = builder.module()

    expected = "Parameter String"

    # Define function that takes string parameter and prints it
    # string test_function(string msg) { print(msg); return msg; }
    args = astx.Arguments()
    args.append(astx.Argument(name="msg", type_=astx.String()))

    func_block = astx.Block()
    func_block.append(PrintExpr(astx.LiteralUTF8String("msg")))
    func_block.append(astx.FunctionReturn(astx.Variable("msg")))

    test_func_proto = astx.FunctionPrototype(
        name="test_function", args=args, return_type=astx.String()
    )
    test_func = astx.FunctionDef(prototype=test_func_proto, body=func_block)

    # Main function calls test_function
    string_literal = astx.LiteralUTF8String(expected)
    func_call = astx.FunctionCall(fn=test_func, args=[string_literal])

    decl_tmp = astx.VariableDeclaration(
        name="result", type_=astx.String(), value=func_call
    )

    main_block = astx.Block()
    main_block.append(decl_tmp)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: int main() -> returns 0
    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(test_func)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output=expected)

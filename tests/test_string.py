"""Tests for string."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "string_value",
    [
        "hello",
        "world",
        "LLVM test",
        "utf8✓",
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_string_literals_with_print(
    builder_class: Type[Builder],
    string_value: str,
) -> None:
    """Test printing UTF8 string literals."""
    builder = builder_class()
    module = builder.module()

    # Create a literal UTF-8 string
    literal_str = astx.LiteralUTF8String(string_value)

    # Declare tmp: string = literal_str
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.UTF8String(), value=literal_str
    )

    # Block: declare string, print, return 0
    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(literal_str))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: int main()
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    # The expected output is the string literal itself
    check_result("build", builder, module, expected_output=string_value)

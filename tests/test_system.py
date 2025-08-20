"""Tests for the PrintExpr node."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import Cast, PrintExpr

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
def test_print_expr(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """Test the PrintExpr node."""
    builder = builder_class()
    module = builder.module()

    print_node = PrintExpr(astx.LiteralUTF8String("Hello, world!"))

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(print_node)
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    check_result(action, builder, module, expected_file)


def test_printexpr_init_and_struct_direct():
    msg = astx.LiteralUTF8String("Direct test")
    node = PrintExpr(msg)

    struct = node.get_struct(simplified=True)
    assert "FunctionCall[PrintExpr]" in struct
    assert list(struct.values())[0] == {
        "LiteralUTF8String: Direct test": "Direct test"
    }


def test_cast_init_and_struct():
    val = astx.LiteralUTF8String("42")
    cast = Cast(val, target_type="int32")

    struct = cast.get_struct(simplified=True)
    assert "Cast[int32]" in struct
    assert list(struct.values())[0] == {"LiteralUTF8String: 42": "42"}

"""Tests for the PrintExpr node."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


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
    action: str, expected_file: str, builder_class: Type[Builder]
) -> None:
    """Test the PrintExpr node."""
    builder = builder_class()
    module = builder.module()

    print_node = PrintExpr(astx.LiteralUTF8String("Hello, world!"))

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(print_node)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.Function(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    check_result(action, builder, module, expected_file)

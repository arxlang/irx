"""Tests for None / void type."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_none_as_placeholder_in_expression(
    builder_class: Type[Builder],
) -> None:
    """Ensure LiteralNone pushes None on result stack without emitting IR."""
    builder = builder_class()
    module = builder.module()

    block_fn_void = astx.Block()
    block_fn_void.append(PrintExpr(astx.LiteralUTF8String("done")))
    block_fn_void.append(astx.FunctionReturn(astx.LiteralNone()))

    fn_void_proto = astx.FunctionPrototype(
        name="fn_void", args=astx.Arguments(), return_type=astx.NoneType()
    )
    fn_void = astx.FunctionDef(prototype=fn_void_proto, body=block_fn_void)

    block_fn_main = astx.Block()
    block_fn_main.append(astx.FunctionCall(fn_void, args=[]))
    block_fn_main.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    fn_main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn_main = astx.FunctionDef(prototype=fn_main_proto, body=block_fn_main)

    module.block.append(fn_void)
    module.block.append(fn_main)

    check_result("build", builder, module, expected_output="done")

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

    # tmp: int32 = 5
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.Int32(), value=astx.LiteralInt32(5)
    )

    # Just evaluate LiteralNone (no IR emitted)
    none_expr = astx.LiteralNone()

    block = astx.Block()
    block.append(decl_tmp)
    block.append(none_expr)
    block.append(PrintExpr(astx.LiteralUTF8String("done")))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="done")

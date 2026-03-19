"""
title: Dict key lookup tests (SubscriptExpr on LiteralDict).
"""

from __future__ import annotations

from typing import cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from irx.system import PrintExpr
from llvmlite import ir

from .conftest import check_result


def _make_int_dict() -> astx.LiteralDict:
    return astx.LiteralDict(
        elements={
            astx.LiteralInt32(1): astx.LiteralInt32(10),
            astx.LiteralInt32(2): astx.LiteralInt32(20),
        }
    )


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_dict_lookup_hit(builder_class: type[Builder]) -> None:
    """
    title: SubscriptExpr constant key returns the correct value.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.LiteralInt32(2)
    )
    visitor.visit(expr)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)
    assert result.constant == 20


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_dict_lookup_miss(builder_class: type[Builder]) -> None:
    """
    title: SubscriptExpr constant key raises KeyError for a missing key.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.LiteralInt32(99)
    )
    with pytest.raises(KeyError, match="not found in dict"):
        visitor.visit(expr)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_dict_lookup_build(builder_class: type[Builder]) -> None:
    """
    title: SubscriptExpr constant key compiles and prints the correct value.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    lookup = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.LiteralInt32(1)
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(PrintExpr(lookup))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="10")


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_dict_lookup_runtime_variable_key(
    builder_class: type[Builder],
) -> None:
    """
    title: SubscriptExpr variable key does runtime linear scan and returns
      the correct value.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    key_decl = astx.InlineVariableDeclaration(
        name="k",
        type_=astx.Int32(),
        value=astx.LiteralInt32(2),
    )
    lookup = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.Identifier("k")
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(key_decl)
    block.append(PrintExpr(lookup))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="20")

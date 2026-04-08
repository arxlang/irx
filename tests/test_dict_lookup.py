"""
title: Dict key lookup tests (SubscriptExpr on LiteralDict).
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.analysis import SemanticError
from irx.builder import Builder as LLVMBuilder
from irx.builder import Visitor as LLVMVisitor
from irx.builder.base import Builder
from irx.system import PrintExpr
from llvmlite import ir

from .conftest import check_result

HAS_LITERAL_TIME = hasattr(astx, "LiteralTime")


def _make_int_dict() -> astx.LiteralDict:
    """
    title: Make int dict.
    returns:
      type: astx.LiteralDict
    """
    return astx.LiteralDict(
        elements={
            astx.LiteralInt32(1): astx.LiteralInt32(10),
            astx.LiteralInt32(2): astx.LiteralInt32(20),
        }
    )


def _make_float_dict() -> astx.LiteralDict:
    """
    title: Make float dict.
    returns:
      type: astx.LiteralDict
    """
    return astx.LiteralDict(
        elements={
            astx.LiteralFloat32(1.5): astx.LiteralInt32(10),
            astx.LiteralFloat32(2.5): astx.LiteralInt32(20),
        }
    )


def _make_lookup_module(
    lookup: astx.SubscriptExpr, *setup_nodes: astx.AST
) -> astx.Module:
    """
    title: Make lookup module.
    parameters:
      lookup:
        type: astx.SubscriptExpr
      setup_nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    module = astx.Module()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    for setup_node in setup_nodes:
        block.append(setup_node)
    block.append(PrintExpr(lookup))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    return module


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_dict_lookup_hit(builder_class: type[Builder]) -> None:
    """
    title: SubscriptExpr constant key returns the correct value.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.LiteralInt32(2)
    )
    visitor.visit(expr)
    result = visitor.result_stack.pop()

    assert isinstance(result, ir.Constant)
    EXPECTED_VAL_FOR_KEY_2 = 20
    assert result.constant == EXPECTED_VAL_FOR_KEY_2


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_dict_lookup_miss(builder_class: type[Builder]) -> None:
    """
    title: SubscriptExpr constant key raises KeyError for a missing key.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.LiteralInt32(99)
    )
    with pytest.raises(KeyError, match="not found in dict"):
        visitor.visit(expr)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_dict_lookup_build(builder_class: type[Builder]) -> None:
    """
    title: SubscriptExpr constant key compiles and prints the correct value.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    lookup = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.LiteralInt32(1)
    )
    module = _make_lookup_module(lookup)

    check_result("build", builder, module, expected_output="10")


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_dict_lookup_runtime_variable_key(
    builder_class: type[Builder],
) -> None:
    """
    title: >-
      SubscriptExpr variable integer key does runtime lookup and returns the
      correct value.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    key_decl = astx.InlineVariableDeclaration(
        name="k",
        type_=astx.Int32(),
        value=astx.LiteralInt32(2),
    )
    lookup = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.Identifier("k")
    )
    module = _make_lookup_module(lookup, key_decl)

    check_result("build", builder, module, expected_output="20")


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_dict_lookup_runtime_variable_key_miss_exits(
    builder_class: type[Builder],
) -> None:
    """
    title: SubscriptExpr runtime miss lowers to an explicit exit path.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    key_decl = astx.InlineVariableDeclaration(
        name="k",
        type_=astx.Int32(),
        value=astx.LiteralInt32(99),
    )
    lookup = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.Identifier("k")
    )
    module = _make_lookup_module(lookup, key_decl)

    ir_text = builder.translate(module)

    assert 'call void @"exit"(i32 1)' in ir_text
    assert "switch i32" in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_dict_lookup_empty_dict_runtime_key_raises_keyerror(
    builder_class: type[Builder],
) -> None:
    """
    title: Empty dict lookup with a runtime key fails during compilation.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    key_decl = astx.InlineVariableDeclaration(
        name="k",
        type_=astx.Int32(),
        value=astx.LiteralInt32(1),
    )
    lookup = astx.SubscriptExpr(
        value=astx.LiteralDict(elements={}),
        index=astx.Identifier("k"),
    )
    module = _make_lookup_module(lookup, key_decl)

    with pytest.raises(SemanticError, match="empty dict"):
        builder.translate(module)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_dict_lookup_runtime_variable_key_mismatched_integer_widths(
    builder_class: type[Builder],
) -> None:
    """
    title: SubscriptExpr widens runtime integer keys before switch lookup.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    key_decl = astx.InlineVariableDeclaration(
        name="k",
        type_=astx.Int16(),
        value=astx.LiteralInt16(2),
    )
    lookup = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.Identifier("k")
    )
    module = _make_lookup_module(lookup, key_decl)

    ir_text = builder.translate(module)

    assert "sext i16" in ir_text
    assert "switch i32" in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_dict_lookup_rejects_incompatible_constant_key_type(
    builder_class: type[Builder],
) -> None:
    """
    title: SubscriptExpr rejects incompatible constant key categories.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.SubscriptExpr(
        value=_make_int_dict(), index=astx.LiteralFloat32(2.0)
    )
    with pytest.raises(TypeError, match="incompatible with dict key type"):
        visitor.visit(expr)


@pytest.mark.skipif(
    not HAS_LITERAL_TIME, reason="astx.LiteralTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_dict_lookup_rejects_unsupported_constant_time_keys(
    builder_class: type[Builder],
) -> None:
    """
    title: SubscriptExpr rejects unsupported non-numeric constant key types.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.SubscriptExpr(
        value=astx.LiteralDict(
            elements={
                astx.LiteralTime("12:34:56"): astx.LiteralInt32(7),
            }
        ),
        index=astx.LiteralTime("12:34:56"),
    )
    with pytest.raises(
        TypeError,
        match="only integer and floating-point dict keys are supported",
    ):
        visitor.visit(expr)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_dict_lookup_runtime_float_variable_key(
    builder_class: type[Builder],
) -> None:
    """
    title: SubscriptExpr runtime float key lowers to float comparisons.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    key_decl = astx.InlineVariableDeclaration(
        name="k",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(2.5),
    )
    lookup = astx.SubscriptExpr(
        value=_make_float_dict(), index=astx.Identifier("k")
    )
    module = _make_lookup_module(lookup, key_decl)

    ir_text = builder.translate(module)

    assert "fcmp oeq float" in ir_text
    assert 'call void @"exit"(i32 1)' in ir_text

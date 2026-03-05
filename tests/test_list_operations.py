"""
title: Tests for list operations lowering.
"""

from __future__ import annotations

from typing import Type, cast

import astx
import pytest

from irx import system
from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir


def _lower(node: astx.AST, builder_class: Type[Builder]) -> ir.Value:
    """
    title: Lower one AST node and return resulting value.
    parameters:
      node:
        type: astx.AST
      builder_class:
        type: Type[Builder]
    returns:
      type: ir.Value
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()
    visitor.visit(node)
    return cast(ir.Value, visitor.result_stack.pop())


def _array_values(const: ir.Constant) -> list[int]:
    """
    title: Extract integer values from a constant array.
    parameters:
      const:
        type: ir.Constant
    returns:
      type: list[int]
    """
    assert isinstance(const.type, ir.ArrayType)
    assert isinstance(const.type.element, ir.IntType)
    return [int(elem.constant) for elem in const.constant]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_list_insert(builder_class: Type[Builder]) -> None:
    """
    title: Insert updates list content and size.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    list_expr = astx.LiteralList(
        elements=[
            astx.LiteralInt32(1),
            astx.LiteralInt32(3),
        ]
    )
    node = system.ListInsertExpr(
        list_expr=list_expr,
        index=astx.LiteralInt32(1),
        value=astx.LiteralInt32(2),
    )
    result = _lower(node, builder_class)
    assert isinstance(result, ir.Constant)
    assert isinstance(result.type, ir.ArrayType)
    assert result.type.count == 3  # noqa: PLR2004
    assert _array_values(result) == [1, 2, 3]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_list_remove(builder_class: Type[Builder]) -> None:
    """
    title: Remove deletes first matching value.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    list_expr = astx.LiteralList(
        elements=[
            astx.LiteralInt32(1),
            astx.LiteralInt32(2),
            astx.LiteralInt32(2),
            astx.LiteralInt32(3),
        ]
    )
    node = system.ListRemoveExpr(
        list_expr=list_expr,
        value=astx.LiteralInt32(2),
    )
    result = _lower(node, builder_class)
    assert isinstance(result, ir.Constant)
    assert isinstance(result.type, ir.ArrayType)
    assert result.type.count == 3  # noqa: PLR2004
    assert _array_values(result) == [1, 2, 3]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_list_search_found_and_missing(builder_class: Type[Builder]) -> None:
    """
    title: Search returns index or -1 when missing.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    list_expr = astx.LiteralList(
        elements=[
            astx.LiteralInt32(10),
            astx.LiteralInt32(20),
            astx.LiteralInt32(30),
        ]
    )
    found = _lower(
        system.ListSearchExpr(
            list_expr=list_expr,
            value=astx.LiteralInt32(20),
        ),
        builder_class,
    )
    missing = _lower(
        system.ListSearchExpr(
            list_expr=list_expr,
            value=astx.LiteralInt32(999),
        ),
        builder_class,
    )
    assert isinstance(found, ir.Constant)
    assert isinstance(missing, ir.Constant)
    assert int(found.constant) == 1
    assert int(missing.constant) == -1


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_list_count(builder_class: Type[Builder]) -> None:
    """
    title: Count returns number of matching entries.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    list_expr = astx.LiteralList(
        elements=[
            astx.LiteralInt32(5),
            astx.LiteralInt32(5),
            astx.LiteralInt32(6),
            astx.LiteralInt32(5),
        ]
    )
    result = _lower(
        system.ListCountExpr(
            list_expr=list_expr,
            value=astx.LiteralInt32(5),
        ),
        builder_class,
    )
    assert isinstance(result, ir.Constant)
    assert int(result.constant) == 3  # noqa: PLR2004


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_list_slice(builder_class: Type[Builder]) -> None:
    """
    title: Slice returns a new list for start:end bounds.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    list_expr = astx.LiteralList(
        elements=[
            astx.LiteralInt32(0),
            astx.LiteralInt32(1),
            astx.LiteralInt32(2),
            astx.LiteralInt32(3),
        ]
    )
    result = _lower(
        system.ListSliceExpr(
            list_expr=list_expr,
            start=astx.LiteralInt32(1),
            end=astx.LiteralInt32(3),
        ),
        builder_class,
    )
    assert isinstance(result, ir.Constant)
    assert isinstance(result.type, ir.ArrayType)
    assert result.type.count == 2  # noqa: PLR2004
    assert _array_values(result) == [1, 2]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_list_remove_raises_when_value_missing(
    builder_class: Type[Builder],
) -> None:
    """
    title: Remove raises if value does not exist.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    list_expr = astx.LiteralList(
        elements=[astx.LiteralInt32(1), astx.LiteralInt32(2)]
    )
    with pytest.raises(ValueError, match="not found"):
        _lower(
            system.ListRemoveExpr(
                list_expr=list_expr,
                value=astx.LiteralInt32(42),
            ),
            builder_class,
        )

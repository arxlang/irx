"""
title: Structural tests for system AST nodes.
"""

from typing import Any, cast

import astx

from irx.system import Cast, PrintExpr


def test_print_expr_get_struct_shapes() -> None:
    """
    title: PrintExpr get_struct should work for full and simplified output.
    """
    message = astx.LiteralUTF8String("hello")
    node = PrintExpr(message)

    full = node.get_struct()
    full_key = f"FunctionCall[{node}]"
    assert isinstance(full, dict)
    assert full_key in full
    full_entry = cast(dict[str, Any], full[full_key])
    assert full_entry["content"] == message.get_struct(False)
    assert "metadata" in full_entry

    simplified = node.get_struct(simplified=True)
    assert isinstance(simplified, dict)
    assert simplified[full_key] == message.get_struct(True)


def test_cast_get_struct_shapes() -> None:
    """
    title: Cast get_struct should include target type in its key.
    """
    value = astx.LiteralInt32(7)
    node = Cast(value=value, target_type=astx.Float32())
    key = f"Cast[{node.target_type}]"

    full = node.get_struct()
    assert isinstance(full, dict)
    assert key in full
    full_entry = cast(dict[str, Any], full[key])
    assert full_entry["content"] == value.get_struct(False)

    simplified = node.get_struct(simplified=True)
    assert isinstance(simplified, dict)
    assert simplified[key] == value.get_struct(True)


def test_print_expr_names_are_unique() -> None:
    """
    title: PrintExpr should allocate monotonically unique symbol names.
    """
    first = PrintExpr(astx.LiteralUTF8String("a"))
    second = PrintExpr(astx.LiteralUTF8String("b"))
    assert first._name != second._name

"""
title: Structural tests for Arrow AST nodes.
"""

from typing import Any, cast

import astx

from irx.arrow import ArrowInt32ArrayLength


def test_arrow_int32_array_length_get_struct_shapes() -> None:
    """
    title: Arrow helper get_struct should work for full and simplified output.
    """
    first = astx.LiteralInt32(1)
    second = astx.LiteralInt32(2)
    node = ArrowInt32ArrayLength([first, second])

    full = node.get_struct()
    assert isinstance(full, dict)
    assert "ArrowInt32ArrayLength" in full
    full_entry = cast(dict[str, Any], full["ArrowInt32ArrayLength"])
    assert full_entry["content"] == [
        first.get_struct(False),
        second.get_struct(False),
    ]

    simplified = node.get_struct(simplified=True)
    assert isinstance(simplified, dict)
    assert simplified["ArrowInt32ArrayLength"] == [
        first.get_struct(True),
        second.get_struct(True),
    ]

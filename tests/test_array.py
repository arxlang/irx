"""
title: Structural tests for array AST nodes.
"""

from typing import Any, cast

from irx import astx
from irx.builder import Builder


def test_array_int32_array_length_get_struct_shapes() -> None:
    """
    title: Array helper get_struct should work for full and simplified output.
    """
    first = astx.LiteralInt32(1)
    second = astx.LiteralInt32(2)
    node = astx.ArrayInt32ArrayLength([first, second])

    full = node.get_struct()
    assert isinstance(full, dict)
    assert "ArrayInt32ArrayLength" in full
    full_entry = cast(dict[str, Any], full["ArrayInt32ArrayLength"])
    assert full_entry["content"] == [
        first.get_struct(False),
        second.get_struct(False),
    ]

    simplified = node.get_struct(simplified=True)
    assert isinstance(simplified, dict)
    assert simplified["ArrayInt32ArrayLength"] == [
        first.get_struct(True),
        second.get_struct(True),
    ]


def test_arrow_alias_preserves_legacy_struct_key() -> None:
    """
    title: Legacy Arrow helper aliases should remain available.
    """
    node = astx.ArrowInt32ArrayLength([astx.LiteralInt32(1)])

    full = node.get_struct()

    assert isinstance(full, dict)
    assert "ArrowInt32ArrayLength" in full


def test_arrow_alias_still_lowers_through_array_runtime() -> None:
    """
    title: >-
      Legacy Arrow helper aliases should still lower through the array runtime.
    """
    builder = Builder()
    module = astx.Module()
    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    body = astx.Block()
    body.append(
        astx.FunctionReturn(astx.ArrowInt32ArrayLength([astx.LiteralInt32(1)]))
    )
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))

    ir_text = builder.translate(module)

    assert '@"irx_arrow_array_length"' in ir_text
    assert (
        "array" in builder.translator.runtime_features.active_feature_names()
    )

"""
title: Structural tests for list helper AST nodes.
"""

from __future__ import annotations

import importlib

from typing import Any, cast

from irx import astx


def test_collections_package_reexports_list_helpers() -> None:
    """
    title: Collections package should preserve the list helper import surface.
    """
    package = importlib.import_module("irx.astx.collections")
    list_module = importlib.import_module("irx.astx.collections.list")

    assert package.ListCreate is list_module.ListCreate is astx.ListCreate
    assert package.ListIndex is list_module.ListIndex is astx.ListIndex
    assert package.ListAppend is list_module.ListAppend is astx.ListAppend
    assert package.ListLength is list_module.ListLength is astx.ListLength


def test_list_create_get_struct_shapes() -> None:
    """
    title: ListCreate get_struct should expose the element type.
    """
    element_type = astx.Int32()
    node = astx.ListCreate(element_type)

    full = node.get_struct()
    assert isinstance(full, dict)
    full_entry = cast(dict[str, Any], full["ListCreate"])
    assert full_entry["content"] == {
        "element_type": element_type.get_struct(False)
    }

    simplified = node.get_struct(simplified=True)
    assert isinstance(simplified, dict)
    assert simplified["ListCreate"] == {
        "element_type": element_type.get_struct(True)
    }


def test_list_index_get_struct_shapes() -> None:
    """
    title: ListIndex get_struct should expose the base and index nodes.
    """
    base = astx.Identifier("items")
    index = astx.LiteralInt32(3)
    node = astx.ListIndex(base, index)

    full = node.get_struct()
    assert isinstance(full, dict)
    full_entry = cast(dict[str, Any], full["ListIndex"])
    assert full_entry["content"] == {
        "base": base.get_struct(False),
        "index": index.get_struct(False),
    }

    simplified = node.get_struct(simplified=True)
    assert isinstance(simplified, dict)
    assert simplified["ListIndex"] == {
        "base": base.get_struct(True),
        "index": index.get_struct(True),
    }


def test_list_append_get_struct_shapes() -> None:
    """
    title: ListAppend get_struct should expose the base and appended value.
    """
    base = astx.Identifier("items")
    value = astx.LiteralInt32(7)
    node = astx.ListAppend(base, value)

    full = node.get_struct()
    assert isinstance(full, dict)
    full_entry = cast(dict[str, Any], full["ListAppend"])
    assert full_entry["content"] == {
        "base": base.get_struct(False),
        "value": value.get_struct(False),
    }

    simplified = node.get_struct(simplified=True)
    assert isinstance(simplified, dict)
    assert simplified["ListAppend"] == {
        "base": base.get_struct(True),
        "value": value.get_struct(True),
    }


def test_list_length_get_struct_shapes() -> None:
    """
    title: ListLength get_struct should expose the base node.
    """
    base = astx.Identifier("items")
    node = astx.ListLength(base)

    full = node.get_struct()
    assert isinstance(full, dict)
    full_entry = cast(dict[str, Any], full["ListLength"])
    assert full_entry["content"] == {"base": base.get_struct(False)}

    simplified = node.get_struct(simplified=True)
    assert isinstance(simplified, dict)
    assert simplified["ListLength"] == {"base": base.get_struct(True)}

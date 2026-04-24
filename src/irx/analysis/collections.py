"""
title: Collection capability helpers for semantic analysis.
summary: >-
  Centralize collection-kind detection, item-type extraction, and method
  capability records for list, tuple, set, and dictionary operations.
"""

from __future__ import annotations

from enum import Enum
from typing import cast

from public import public

from irx import astx
from irx.builtins.collections.list import list_element_type
from irx.typecheck import typechecked


@public
@typechecked
class CollectionKind(str, Enum):
    """
    title: Stable semantic collection-kind names.
    """

    LIST = "list"
    TUPLE = "tuple"
    SET = "set"
    DICT = "dict"


@public
@typechecked
def collection_kind(
    type_: astx.DataType | None,
) -> CollectionKind | None:
    """
    title: Return the collection kind represented by one type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: CollectionKind | None
    """
    if isinstance(type_, astx.ListType):
        return CollectionKind.LIST
    if isinstance(type_, astx.TupleType):
        return CollectionKind.TUPLE
    if isinstance(type_, astx.SetType):
        return CollectionKind.SET
    if isinstance(type_, astx.DictType):
        return CollectionKind.DICT
    return None


@public
@typechecked
def is_collection_type(type_: astx.DataType | None) -> bool:
    """
    title: Return whether one type is a supported collection type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return collection_kind(type_) is not None


@public
@typechecked
def collection_value_types(
    type_: astx.DataType | None,
) -> tuple[astx.DataType, ...]:
    """
    title: Return value-like element types for one collection.
    summary: >-
      Lists and sets expose their element type, tuples expose all positional
      member types, and dictionaries expose their value type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: tuple[astx.DataType, Ellipsis]
    """
    list_type = list_element_type(type_)
    if list_type is not None:
        return (list_type,)
    if isinstance(type_, astx.TupleType):
        return tuple(
            member
            for member in cast(list[object], type_.element_types)
            if isinstance(member, astx.DataType)
        )
    if isinstance(type_, astx.SetType) and isinstance(
        type_.element_type,
        astx.DataType,
    ):
        return (type_.element_type,)
    if isinstance(type_, astx.DictType) and isinstance(
        type_.value_type,
        astx.DataType,
    ):
        return (type_.value_type,)
    return ()


@public
@typechecked
def collection_contains_types(
    type_: astx.DataType | None,
) -> tuple[astx.DataType, ...]:
    """
    title: Return accepted containment probe types for one collection.
    summary: >-
      Dictionaries use key containment; other collections use value/element
      containment.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: tuple[astx.DataType, Ellipsis]
    """
    if isinstance(type_, astx.DictType) and isinstance(
        type_.key_type,
        astx.DataType,
    ):
        return (type_.key_type,)
    return collection_value_types(type_)


@public
@typechecked
def collection_supports_sequence_search(
    type_: astx.DataType | None,
) -> bool:
    """
    title: Return whether a collection supports index/count queries.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return isinstance(type_, (astx.ListType, astx.TupleType))


__all__ = [
    "CollectionKind",
    "collection_contains_types",
    "collection_kind",
    "collection_supports_sequence_search",
    "collection_value_types",
    "is_collection_type",
]

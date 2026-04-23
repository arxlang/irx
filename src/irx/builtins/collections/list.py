"""
title: Minimal dynamic-list metadata and type helpers.
summary: >-
  Centralize the small IRX-side list layout contract plus the AST-type helpers
  shared by semantic analysis and lowering.
"""

from __future__ import annotations

import astx

from irx.typecheck import typechecked

LIST_FIELD_INDICES: dict[str, int] = {
    "data": 0,
    "length": 1,
    "capacity": 2,
    "element_size": 3,
}
LIST_RUNTIME_FEATURE = "list"
LIST_APPEND_SYMBOL = "irx_list_append"
LIST_AT_SYMBOL = "irx_list_at"


@typechecked
def list_element_type(type_: astx.DataType | None) -> astx.DataType | None:
    """
    title: Return one concrete list element type when available.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: astx.DataType | None
    """
    if not isinstance(type_, astx.ListType):
        return None
    if len(type_.element_types) != 1:
        return None
    element_type = type_.element_types[0]
    return element_type if isinstance(element_type, astx.DataType) else None


@typechecked
def list_has_concrete_element_type(type_: astx.DataType | None) -> bool:
    """
    title: Return whether one list type has a single concrete element type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return list_element_type(type_) is not None

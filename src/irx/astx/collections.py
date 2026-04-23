"""
title: IRX-owned list builder AST nodes.
summary: >-
  Provide the smallest explicit list-construction API that host frontends can
  target when they need to build list values incrementally in non-literal
  control-flow contexts.
"""

from __future__ import annotations

from typing import cast

import astx

from irx.typecheck import typechecked


@typechecked
class ListCreate(astx.base.DataType):
    """
    title: Internal empty-list constructor node.
    attributes:
      element_type:
        type: astx.DataType
      type_:
        type: astx.ListType
    """

    element_type: astx.DataType
    type_: astx.ListType

    def __init__(self, element_type: astx.DataType) -> None:
        """
        title: Initialize one empty-list constructor.
        parameters:
          element_type:
            type: astx.DataType
        """
        super().__init__()
        self.element_type = element_type
        self.type_ = astx.ListType([element_type])

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        return self._prepare_struct(
            "ListCreate",
            cast(
                astx.base.ReprStruct,
                {"element_type": self.element_type.get_struct(simplified)},
            ),
            simplified,
        )


@typechecked
class ListAppend(astx.base.DataType):
    """
    title: Internal list append node.
    summary: >-
      Append one value to an existing mutable list variable or field. This node
      models incremental list growth and is not a user-facing collection API by
      itself.
    attributes:
      base:
        type: astx.AST
      value:
        type: astx.AST
      type_:
        type: astx.Int32
    """

    base: astx.AST
    value: astx.AST
    type_: astx.Int32

    def __init__(self, base: astx.AST, value: astx.AST) -> None:
        """
        title: Initialize one list append node.
        parameters:
          base:
            type: astx.AST
          value:
            type: astx.AST
        """
        super().__init__()
        self.base = base
        self.value = value
        self.type_ = astx.Int32()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        return self._prepare_struct(
            "ListAppend",
            cast(
                astx.base.ReprStruct,
                {
                    "base": self.base.get_struct(simplified),
                    "value": self.value.get_struct(simplified),
                },
            ),
            simplified,
        )


__all__ = ["ListAppend", "ListCreate"]

"""
title: IRX-owned list helper AST nodes.
summary: >-
  Provide the smallest explicit list-construction and query API that host
  frontends can target when they need to build list values incrementally or
  reason about list metadata in lowered control-flow contexts.
"""

from __future__ import annotations

from typing import cast

import astx

from astx.types import AnyType

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
class ListIndex(astx.base.DataType):
    """
    title: Internal list indexing node.
    summary: >-
      Read one element from a list-valued expression using one integer index.
    attributes:
      base:
        type: astx.AST
      index:
        type: astx.AST
      type_:
        type: astx.DataType
    """

    base: astx.AST
    index: astx.AST
    type_: astx.DataType

    def __init__(self, base: astx.AST, index: astx.AST) -> None:
        """
        title: Initialize one list indexing expression.
        parameters:
          base:
            type: astx.AST
          index:
            type: astx.AST
        """
        super().__init__()
        self.base = base
        self.index = index
        self.type_ = AnyType()

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
            "ListIndex",
            cast(
                astx.base.ReprStruct,
                {
                    "base": self.base.get_struct(simplified),
                    "index": self.index.get_struct(simplified),
                },
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


@typechecked
class ListLength(astx.base.DataType):
    """
    title: Internal list length node.
    summary: >-
      Return the current logical length of one list-valued expression as an
      int32 value. The runtime stores list lengths as int64, but the current
      IRx language-level contract intentionally truncates that representation
      to Int32.
    attributes:
      base:
        type: astx.AST
      type_:
        type: astx.Int32
    """

    base: astx.AST
    type_: astx.Int32

    def __init__(self, base: astx.AST) -> None:
        """
        title: Initialize one list length query.
        parameters:
          base:
            type: astx.AST
        """
        super().__init__()
        self.base = base
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
            "ListLength",
            cast(
                astx.base.ReprStruct,
                {"base": self.base.get_struct(simplified)},
            ),
            simplified,
        )


__all__ = ["ListAppend", "ListCreate", "ListIndex", "ListLength"]

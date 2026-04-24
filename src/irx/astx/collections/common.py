"""
title: Shared collection helper AST nodes.
summary: >-
  Provide backend-neutral collection query nodes that frontends can target when
  they need common list, tuple, set, or dictionary operations without binding
  directly to one runtime representation.
"""

from __future__ import annotations

from typing import cast

import astx

from irx.typecheck import typechecked


@typechecked
class CollectionLength(astx.base.DataType):
    """
    title: Common collection length query node.
    summary: >-
      Return the logical length of a list, tuple, set, or dictionary as an
      Int32 value.
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
        title: Initialize one collection length query.
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
            "CollectionLength",
            cast(
                astx.base.ReprStruct,
                {"base": self.base.get_struct(simplified)},
            ),
            simplified,
        )


@typechecked
class CollectionIsEmpty(astx.base.DataType):
    """
    title: Common collection emptiness query node.
    summary: >-
      Return whether a list, tuple, set, or dictionary has no logical entries.
    attributes:
      base:
        type: astx.AST
      type_:
        type: astx.Boolean
    """

    base: astx.AST
    type_: astx.Boolean

    def __init__(self, base: astx.AST) -> None:
        """
        title: Initialize one collection emptiness query.
        parameters:
          base:
            type: astx.AST
        """
        super().__init__()
        self.base = base
        self.type_ = astx.Boolean()

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
            "CollectionIsEmpty",
            cast(
                astx.base.ReprStruct,
                {"base": self.base.get_struct(simplified)},
            ),
            simplified,
        )


@typechecked
class CollectionContains(astx.base.DataType):
    """
    title: Common collection containment query node.
    summary: >-
      Return whether a list, tuple, or set contains a value, or whether a
      dictionary contains a key.
    attributes:
      base:
        type: astx.AST
      value:
        type: astx.AST
      type_:
        type: astx.Boolean
    """

    base: astx.AST
    value: astx.AST
    type_: astx.Boolean

    def __init__(self, base: astx.AST, value: astx.AST) -> None:
        """
        title: Initialize one collection containment query.
        parameters:
          base:
            type: astx.AST
          value:
            type: astx.AST
        """
        super().__init__()
        self.base = base
        self.value = value
        self.type_ = astx.Boolean()

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
            "CollectionContains",
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
class CollectionIndex(astx.base.DataType):
    """
    title: Common sequence index query node.
    summary: >-
      Return the first zero-based index of a value in a list or tuple. The
      initial IRx contract returns -1 when the value is not found.
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
        title: Initialize one sequence index query.
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
            "CollectionIndex",
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
class CollectionCount(astx.base.DataType):
    """
    title: Common sequence count query node.
    summary: >-
      Return how many times a value appears in a list or tuple as an Int32
      value.
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
        title: Initialize one sequence count query.
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
            "CollectionCount",
            cast(
                astx.base.ReprStruct,
                {
                    "base": self.base.get_struct(simplified),
                    "value": self.value.get_struct(simplified),
                },
            ),
            simplified,
        )


__all__ = [
    "CollectionContains",
    "CollectionCount",
    "CollectionIndex",
    "CollectionIsEmpty",
    "CollectionLength",
]

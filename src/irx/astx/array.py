"""
title: IRx-owned array AST nodes.
"""

from typing import cast

import astx

from irx.typecheck import typechecked


@typechecked
class ArrayInt32ArrayLength(astx.base.DataType):
    """
    title: Internal array helper AST node.
    summary: >-
      Build an int32 array using the IRx builtin array runtime, then return its
      length.
    attributes:
      values:
        type: list[astx.AST]
      type_:
        type: astx.Int32
    """

    values: list[astx.AST]
    type_: astx.Int32

    def __init__(self, values: list[astx.AST]) -> None:
        """
        title: Initialize ArrayInt32ArrayLength.
        parameters:
          values:
            type: list[astx.AST]
        """
        super().__init__()
        self.values = values
        self.type_ = astx.Int32()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the array helper.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        value = cast(
            astx.base.ReprStruct,
            [item.get_struct(simplified) for item in self.values],
        )
        return self._prepare_struct(
            "ArrayInt32ArrayLength",
            value,
            simplified,
        )


__all__ = ["ArrayInt32ArrayLength"]

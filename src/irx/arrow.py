"""
title: Arrow-specific AST helpers.
"""

from typing import cast

import astx


class ArrowInt32ArrayLength(astx.base.DataType):
    """
    title: Internal Arrow helper AST node.
    summary: >-
      Build an Arrow int32 array using the IRx runtime, then return its length.
    attributes:
      values:
        type: list[astx.AST]
      type_:
        type: astx.Int32
    """

    values: list[astx.AST]
    type_: astx.Int32

    def __init__(self, values: list[astx.AST]) -> None:
        super().__init__()
        self.values = values
        self.type_ = astx.Int32()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the Arrow helper.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = "ArrowInt32ArrayLength"
        value = cast(
            astx.base.ReprStruct,
            [item.get_struct(simplified) for item in self.values],
        )
        return self._prepare_struct(key, value, simplified)

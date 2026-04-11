"""
title: IRx-owned public FFI AST types.
summary: >-
  Provide the minimal pointer and opaque-handle type nodes that hosts can use
  when targeting IRx's stable public FFI layer.
"""

from __future__ import annotations

from typing import cast

import astx

from astx.types import AnyType

from irx.typecheck import typechecked


@typechecked
class PointerType(AnyType):
    """
    title: Stable public FFI pointer type.
    attributes:
      pointee_type:
        type: astx.DataType | None
    """

    pointee_type: astx.DataType | None

    def __init__(self, pointee_type: astx.DataType | None = None) -> None:
        """
        title: Initialize one pointer type.
        parameters:
          pointee_type:
            type: astx.DataType | None
        """
        super().__init__()
        self.pointee_type = pointee_type

    def __str__(self) -> str:
        """
        title: Render one pointer type.
        returns:
          type: str
        """
        if self.pointee_type is None:
            return "PointerType"
        return f"PointerType[{self.pointee_type}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a pointer type.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = "POINTER-TYPE"
        value = (
            None
            if self.pointee_type is None
            else self.pointee_type.get_struct(simplified)
        )
        return self._prepare_struct(
            key,
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class OpaqueHandleType(AnyType):
    """
    title: Stable public FFI opaque-handle type.
    attributes:
      handle_name:
        type: str
    """

    handle_name: str

    def __init__(self, handle_name: str) -> None:
        """
        title: Initialize one opaque-handle type.
        parameters:
          handle_name:
            type: str
        """
        super().__init__()
        self.handle_name = handle_name

    def __str__(self) -> str:
        """
        title: Render one opaque-handle type.
        returns:
          type: str
        """
        return f"OpaqueHandleType[{self.handle_name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for an opaque-handle type.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        return self._prepare_struct(
            f"OPAQUE-HANDLE[{self.handle_name}]",
            self.handle_name,
            simplified,
        )


__all__ = ["OpaqueHandleType", "PointerType"]

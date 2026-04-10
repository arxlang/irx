"""
title: IRx-owned low-level buffer AST nodes.
summary: >-
  Provide internal nodes that host compilers can target for the buffer/view
  substrate without defining a user-facing array API.
"""

from __future__ import annotations

from typing import cast

import astx

from astx.types import AnyType

from irx.buffer import BufferViewMetadata
from irx.typecheck import typechecked


@typechecked
class BufferOwnerType(AnyType):
    """
    title: Internal opaque buffer owner handle type.
    """

    def __str__(self) -> str:
        """
        title: Render the buffer owner type.
        returns:
          type: str
        """
        return "BufferOwnerType"


@typechecked
class BufferViewType(AnyType):
    """
    title: Internal canonical low-level buffer view descriptor type.
    """

    def __str__(self) -> str:
        """
        title: Render the buffer view type.
        returns:
          type: str
        """
        return "BufferViewType"


@typechecked
class BufferViewDescriptor(astx.base.DataType):
    """
    title: Internal low-level buffer view descriptor literal.
    attributes:
      metadata:
        type: BufferViewMetadata
      type_:
        type: BufferViewType
    """

    metadata: BufferViewMetadata
    type_: BufferViewType

    def __init__(self, metadata: BufferViewMetadata) -> None:
        """
        title: Initialize one buffer view descriptor.
        parameters:
          metadata:
            type: BufferViewMetadata
        """
        super().__init__()
        self.metadata = metadata
        self.type_ = BufferViewType()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        _ = simplified
        return self._prepare_struct(
            "BufferViewDescriptor",
            cast(astx.base.ReprStruct, repr(self.metadata)),
            simplified,
        )


@typechecked
class BufferViewWrite(astx.base.DataType):
    """
    title: Internal low-level raw byte write through a buffer view.
    summary: >-
      Writes one 8-bit integer at offset_bytes + byte_offset. This is not a
      generic typed element store or user-facing array mutation API.
    attributes:
      view:
        type: astx.AST
      value:
        type: astx.AST
      byte_offset:
        type: int
      type_:
        type: astx.Int32
    """

    view: astx.AST
    value: astx.AST
    byte_offset: int
    type_: astx.Int32

    def __init__(
        self,
        view: astx.AST,
        value: astx.AST,
        *,
        byte_offset: int = 0,
    ) -> None:
        """
        title: Initialize one low-level view write.
        parameters:
          view:
            type: astx.AST
          value:
            type: astx.AST
          byte_offset:
            type: int
        """
        super().__init__()
        self.view = view
        self.value = value
        self.byte_offset = byte_offset
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
        value = {
            "view": self.view.get_struct(simplified),
            "value": self.value.get_struct(simplified),
            "byte_offset": self.byte_offset,
        }
        return self._prepare_struct(
            "BufferViewWrite",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class BufferViewRetain(astx.base.DataType):
    """
    title: Internal explicit runtime retain for a buffer view owner.
    attributes:
      view:
        type: astx.AST
      type_:
        type: astx.Int32
    """

    view: astx.AST
    type_: astx.Int32

    def __init__(self, view: astx.AST) -> None:
        """
        title: Initialize one buffer retain helper call.
        parameters:
          view:
            type: astx.AST
        """
        super().__init__()
        self.view = view
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
            "BufferViewRetain",
            self.view.get_struct(simplified),
            simplified,
        )


@typechecked
class BufferViewRelease(astx.base.DataType):
    """
    title: Internal explicit runtime release for a buffer view owner.
    attributes:
      view:
        type: astx.AST
      type_:
        type: astx.Int32
    """

    view: astx.AST
    type_: astx.Int32

    def __init__(self, view: astx.AST) -> None:
        """
        title: Initialize one buffer release helper call.
        parameters:
          view:
            type: astx.AST
        """
        super().__init__()
        self.view = view
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
            "BufferViewRelease",
            self.view.get_struct(simplified),
            simplified,
        )


__all__ = [
    "BufferOwnerType",
    "BufferViewDescriptor",
    "BufferViewRelease",
    "BufferViewRetain",
    "BufferViewType",
    "BufferViewWrite",
]

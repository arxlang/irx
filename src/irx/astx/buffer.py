"""
title: IRx-owned low-level buffer AST nodes.
summary: >-
  Provide internal nodes that host compilers can target for the buffer/view
  substrate without defining a user-facing array API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import astx

from astx.types import AnyType

from irx.astx.ffi import OpaqueHandleType
from irx.buffer import BufferViewMetadata
from irx.typecheck import typechecked


@typechecked
class BufferOwnerType(OpaqueHandleType):
    """
    title: Internal opaque buffer owner handle type.
    """

    def __init__(self) -> None:
        """
        title: Initialize the buffer owner handle type.
        """
        super().__init__("irx_buffer_owner_handle")

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
    attributes:
      element_type:
        type: astx.DataType | None
    """

    element_type: astx.DataType | None

    def __init__(self, element_type: astx.DataType | None = None) -> None:
        """
        title: Initialize a buffer view descriptor type.
        parameters:
          element_type:
            type: astx.DataType | None
        """
        super().__init__()
        self.element_type = element_type

    def __str__(self) -> str:
        """
        title: Render the buffer view type.
        returns:
          type: str
        """
        if self.element_type is None:
            return "BufferViewType"
        return f"BufferViewType[{self.element_type}]"


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

    def __init__(
        self,
        metadata: BufferViewMetadata,
        element_type: astx.DataType | None = None,
    ) -> None:
        """
        title: Initialize one buffer view descriptor.
        parameters:
          metadata:
            type: BufferViewMetadata
          element_type:
            type: astx.DataType | None
        """
        super().__init__()
        self.metadata = metadata
        self.type_ = BufferViewType(element_type)

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
class BufferViewIndex(astx.base.DataType):
    """
    title: Internal low-level indexed read through a buffer view.
    summary: >-
      Reads one scalar element by computing offset_bytes plus the sum of
      index*stride byte offsets over the canonical buffer view descriptor.
    attributes:
      base:
        type: astx.AST
      indices:
        type: list[astx.AST]
      type_:
        type: astx.DataType
    """

    base: astx.AST
    indices: list[astx.AST]
    type_: astx.DataType

    def __init__(
        self,
        base: astx.AST,
        indices: Sequence[astx.AST],
    ) -> None:
        """
        title: Initialize one low-level indexed read.
        parameters:
          base:
            type: astx.AST
          indices:
            type: Sequence[astx.AST]
        """
        super().__init__()
        if not indices:
            raise ValueError(
                "buffer view indexing requires at least one index"
            )
        self.base = base
        self.indices = list(indices)
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
        value = {
            "base": self.base.get_struct(simplified),
            "indices": [
                index.get_struct(simplified) for index in self.indices
            ],
        }
        return self._prepare_struct(
            "BufferViewIndex",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class BufferViewStore(astx.base.DataType):
    """
    title: Internal low-level indexed store through a buffer view.
    summary: >-
      Stores one scalar element by computing the canonical descriptor element
      address. This is not a user-facing array mutation API.
    attributes:
      base:
        type: astx.AST
      indices:
        type: list[astx.AST]
      value:
        type: astx.AST
      type_:
        type: astx.Int32
    """

    base: astx.AST
    indices: list[astx.AST]
    value: astx.AST
    type_: astx.Int32

    def __init__(
        self,
        base: astx.AST,
        indices: Sequence[astx.AST],
        value: astx.AST,
    ) -> None:
        """
        title: Initialize one low-level indexed store.
        parameters:
          base:
            type: astx.AST
          indices:
            type: Sequence[astx.AST]
          value:
            type: astx.AST
        """
        super().__init__()
        if not indices:
            raise ValueError(
                "buffer view indexing requires at least one index"
            )
        self.base = base
        self.indices = list(indices)
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
        value = {
            "base": self.base.get_struct(simplified),
            "indices": [
                index.get_struct(simplified) for index in self.indices
            ],
            "value": self.value.get_struct(simplified),
        }
        return self._prepare_struct(
            "BufferViewStore",
            cast(astx.base.ReprStruct, value),
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
    "BufferViewIndex",
    "BufferViewRelease",
    "BufferViewRetain",
    "BufferViewStore",
    "BufferViewType",
    "BufferViewWrite",
]

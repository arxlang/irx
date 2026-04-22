"""
title: IRx-owned array and NDArray AST nodes.
summary: >-
  Provide internal nodes for the Arrow-backed array runtime plus the higher-
  level ndarray abstraction layered on the canonical buffer/view substrate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import astx

from astx.types import AnyType

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


@typechecked
class NDArrayType(AnyType):
    """
    title: Internal NDArray semantic type.
    summary: >-
      Represent the higher-level multidimensional array abstraction while
      reusing the canonical buffer/view representation during lowering.
    attributes:
      element_type:
        type: astx.DataType | None
    """

    element_type: astx.DataType | None

    def __init__(self, element_type: astx.DataType | None = None) -> None:
        """
        title: Initialize one NDArray type.
        parameters:
          element_type:
            type: astx.DataType | None
        """
        super().__init__()
        self.element_type = element_type

    def __str__(self) -> str:
        """
        title: Render the NDArray type.
        returns:
          type: str
        """
        if self.element_type is None:
            return "NDArrayType"
        return f"NDArrayType[{self.element_type}]"


@typechecked
class NDArrayLiteral(astx.base.DataType):
    """
    title: Internal Arrow-backed NDArray literal node.
    summary: >-
      Build one flat Arrow array from scalar values, then attach ndarray shape
      and stride metadata over the resulting storage.
    attributes:
      values:
        type: list[astx.AST]
      element_type:
        type: astx.DataType
      shape:
        type: tuple[int, Ellipsis]
      strides:
        type: tuple[int, Ellipsis] | None
      offset_bytes:
        type: int
      type_:
        type: NDArrayType
    """

    values: list[astx.AST]
    element_type: astx.DataType
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    offset_bytes: int
    type_: NDArrayType

    def __init__(
        self,
        values: Sequence[astx.AST],
        *,
        element_type: astx.DataType,
        shape: Sequence[int],
        strides: Sequence[int] | None = None,
        offset_bytes: int = 0,
    ) -> None:
        """
        title: Initialize one NDArray literal.
        parameters:
          values:
            type: Sequence[astx.AST]
          element_type:
            type: astx.DataType
          shape:
            type: Sequence[int]
          strides:
            type: Sequence[int] | None
          offset_bytes:
            type: int
        """
        super().__init__()
        self.values = list(values)
        self.element_type = element_type
        self.shape = tuple(shape)
        self.strides = None if strides is None else tuple(strides)
        self.offset_bytes = offset_bytes
        self.type_ = NDArrayType(element_type)

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the NDArray literal.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        value = {
            "values": [item.get_struct(simplified) for item in self.values],
            "element_type": self.element_type.get_struct(simplified),
            "shape": list(self.shape),
            "strides": (None if self.strides is None else list(self.strides)),
            "offset_bytes": self.offset_bytes,
        }
        return self._prepare_struct(
            "NDArrayLiteral",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class NDArrayView(astx.base.DataType):
    """
    title: Internal NDArray view node.
    summary: >-
      Build one shallow ndarray view by reusing the base storage and replacing
      the logical shape, strides, and offset metadata.
    attributes:
      base:
        type: astx.AST
      shape:
        type: tuple[int, Ellipsis]
      strides:
        type: tuple[int, Ellipsis] | None
      offset_bytes:
        type: int
      type_:
        type: NDArrayType
    """

    base: astx.AST
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    offset_bytes: int
    type_: NDArrayType

    def __init__(
        self,
        base: astx.AST,
        *,
        shape: Sequence[int],
        strides: Sequence[int] | None = None,
        offset_bytes: int = 0,
    ) -> None:
        """
        title: Initialize one NDArray view.
        parameters:
          base:
            type: astx.AST
          shape:
            type: Sequence[int]
          strides:
            type: Sequence[int] | None
          offset_bytes:
            type: int
        """
        super().__init__()
        self.base = base
        self.shape = tuple(shape)
        self.strides = None if strides is None else tuple(strides)
        self.offset_bytes = offset_bytes
        self.type_ = NDArrayType()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the NDArray view.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        value = {
            "base": self.base.get_struct(simplified),
            "shape": list(self.shape),
            "strides": (None if self.strides is None else list(self.strides)),
            "offset_bytes": self.offset_bytes,
        }
        return self._prepare_struct(
            "NDArrayView",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class NDArrayIndex(astx.base.DataType):
    """
    title: Internal NDArray indexed read.
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
        title: Initialize one NDArray indexed read.
        parameters:
          base:
            type: astx.AST
          indices:
            type: Sequence[astx.AST]
        """
        super().__init__()
        if not indices:
            raise ValueError("ndarray indexing requires at least one index")
        self.base = base
        self.indices = list(indices)
        self.type_ = AnyType()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the indexed read.
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
            "NDArrayIndex",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class NDArrayStore(astx.base.DataType):
    """
    title: Internal NDArray indexed store.
    summary: >-
      Stores one scalar through ndarray shape and stride metadata. Arrow-backed
      ndarrays remain readonly in this phase, but the node keeps the surface
      aligned with future writable views.
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
        title: Initialize one NDArray indexed store.
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
            raise ValueError("ndarray indexing requires at least one index")
        self.base = base
        self.indices = list(indices)
        self.value = value
        self.type_ = astx.Int32()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the indexed store.
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
            "NDArrayStore",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class NDArrayNDim(astx.base.DataType):
    """
    title: Internal NDArray rank query.
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
        title: Initialize one NDArray rank query.
        parameters:
          base:
            type: astx.AST
        """
        super().__init__()
        self.base = base
        self.type_ = astx.Int32()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the rank query.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        return self._prepare_struct(
            "NDArrayNDim",
            self.base.get_struct(simplified),
            simplified,
        )


@typechecked
class NDArrayShape(astx.base.DataType):
    """
    title: Internal NDArray shape-entry query.
    attributes:
      base:
        type: astx.AST
      axis:
        type: int
      type_:
        type: astx.Int64
    """

    base: astx.AST
    axis: int
    type_: astx.Int64

    def __init__(self, base: astx.AST, axis: int) -> None:
        """
        title: Initialize one NDArray shape query.
        parameters:
          base:
            type: astx.AST
          axis:
            type: int
        """
        super().__init__()
        self.base = base
        self.axis = axis
        self.type_ = astx.Int64()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the shape query.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        value = {
            "base": self.base.get_struct(simplified),
            "axis": self.axis,
        }
        return self._prepare_struct(
            "NDArrayShape",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class NDArrayStride(astx.base.DataType):
    """
    title: Internal NDArray stride-entry query.
    attributes:
      base:
        type: astx.AST
      axis:
        type: int
      type_:
        type: astx.Int64
    """

    base: astx.AST
    axis: int
    type_: astx.Int64

    def __init__(self, base: astx.AST, axis: int) -> None:
        """
        title: Initialize one NDArray stride query.
        parameters:
          base:
            type: astx.AST
          axis:
            type: int
        """
        super().__init__()
        self.base = base
        self.axis = axis
        self.type_ = astx.Int64()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the stride query.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        value = {
            "base": self.base.get_struct(simplified),
            "axis": self.axis,
        }
        return self._prepare_struct(
            "NDArrayStride",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class NDArrayElementCount(astx.base.DataType):
    """
    title: Internal NDArray element-count query.
    attributes:
      base:
        type: astx.AST
      type_:
        type: astx.Int64
    """

    base: astx.AST
    type_: astx.Int64

    def __init__(self, base: astx.AST) -> None:
        """
        title: Initialize one NDArray element-count query.
        parameters:
          base:
            type: astx.AST
        """
        super().__init__()
        self.base = base
        self.type_ = astx.Int64()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the element-count query.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        return self._prepare_struct(
            "NDArrayElementCount",
            self.base.get_struct(simplified),
            simplified,
        )


@typechecked
class NDArrayByteOffset(astx.base.DataType):
    """
    title: Internal NDArray byte-offset query for indexed addressing.
    attributes:
      base:
        type: astx.AST
      indices:
        type: list[astx.AST]
      type_:
        type: astx.Int64
    """

    base: astx.AST
    indices: list[astx.AST]
    type_: astx.Int64

    def __init__(
        self,
        base: astx.AST,
        indices: Sequence[astx.AST],
    ) -> None:
        """
        title: Initialize one NDArray byte-offset query.
        parameters:
          base:
            type: astx.AST
          indices:
            type: Sequence[astx.AST]
        """
        super().__init__()
        if not indices:
            raise ValueError("ndarray indexing requires at least one index")
        self.base = base
        self.indices = list(indices)
        self.type_ = astx.Int64()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the byte-offset query.
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
            "NDArrayByteOffset",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class NDArrayRetain(astx.base.DataType):
    """
    title: Internal explicit retain for NDArray-backed storage.
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
        title: Initialize one NDArray retain helper.
        parameters:
          base:
            type: astx.AST
        """
        super().__init__()
        self.base = base
        self.type_ = astx.Int32()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the retain helper.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        return self._prepare_struct(
            "NDArrayRetain",
            self.base.get_struct(simplified),
            simplified,
        )


@typechecked
class NDArrayRelease(astx.base.DataType):
    """
    title: Internal explicit release for NDArray-backed storage.
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
        title: Initialize one NDArray release helper.
        parameters:
          base:
            type: astx.AST
        """
        super().__init__()
        self.base = base
        self.type_ = astx.Int32()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the release helper.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        return self._prepare_struct(
            "NDArrayRelease",
            self.base.get_struct(simplified),
            simplified,
        )


__all__ = [
    "ArrayInt32ArrayLength",
    "NDArrayByteOffset",
    "NDArrayElementCount",
    "NDArrayIndex",
    "NDArrayLiteral",
    "NDArrayNDim",
    "NDArrayRelease",
    "NDArrayRetain",
    "NDArrayShape",
    "NDArrayStore",
    "NDArrayStride",
    "NDArrayType",
    "NDArrayView",
]

"""
title: IRx-owned Tensor AST nodes.
summary: >-
  Provide internal nodes for the Arrow-backed tensor runtime, aligned with
  Apache Arrow's homogeneous N-dimensional tensor model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import astx

from astx.types import AnyType

from irx.typecheck import typechecked


@typechecked
class TensorType(AnyType):
    """
    title: Internal Tensor semantic type.
    summary: >-
      Represent the homogeneous N-dimensional tensor abstraction while reusing
      the canonical buffer/view representation during lowering.
    attributes:
      element_type:
        type: astx.DataType | None
    """

    element_type: astx.DataType | None

    def __init__(self, element_type: astx.DataType | None = None) -> None:
        """
        title: Initialize one Tensor type.
        parameters:
          element_type:
            type: astx.DataType | None
        """
        super().__init__()
        self.element_type = element_type

    def __str__(self) -> str:
        """
        title: Render the Tensor type.
        returns:
          type: str
        """
        if self.element_type is None:
            return "TensorType"
        return f"TensorType[{self.element_type}]"


@typechecked
class TensorLiteral(astx.base.DataType):
    """
    title: Internal Arrow-backed Tensor literal node.
    summary: >-
      Build one Arrow tensor-style value from scalar values plus shape and
      stride metadata.
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
        type: TensorType
    """

    values: list[astx.AST]
    element_type: astx.DataType
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    offset_bytes: int
    type_: TensorType

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
        title: Initialize one Tensor literal.
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
        self.type_ = TensorType(element_type)

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the Tensor literal.
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
            "TensorLiteral",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class TensorView(astx.base.DataType):
    """
    title: Internal Tensor view node.
    summary: >-
      Build one shallow tensor view by reusing the base storage and replacing
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
        type: TensorType
    """

    base: astx.AST
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    offset_bytes: int
    type_: TensorType

    def __init__(
        self,
        base: astx.AST,
        *,
        shape: Sequence[int],
        strides: Sequence[int] | None = None,
        offset_bytes: int = 0,
    ) -> None:
        """
        title: Initialize one Tensor view.
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
        self.type_ = TensorType()

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the Tensor view.
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
            "TensorView",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class TensorIndex(astx.base.DataType):
    """
    title: Internal Tensor indexed read.
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
        title: Initialize one Tensor indexed read.
        parameters:
          base:
            type: astx.AST
          indices:
            type: Sequence[astx.AST]
        """
        super().__init__()
        if not indices:
            raise ValueError("tensor indexing requires at least one index")
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
            "TensorIndex",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class TensorStore(astx.base.DataType):
    """
    title: Internal Tensor indexed store.
    summary: >-
      Stores one scalar through tensor shape and stride metadata. Arrow-backed
      tensors remain readonly in this phase, but the node keeps the surface
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
        title: Initialize one Tensor indexed store.
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
            raise ValueError("tensor indexing requires at least one index")
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
            "TensorStore",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class TensorNDim(astx.base.DataType):
    """
    title: Internal Tensor rank query.
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
        title: Initialize one Tensor rank query.
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
            "TensorNDim",
            self.base.get_struct(simplified),
            simplified,
        )


@typechecked
class TensorShape(astx.base.DataType):
    """
    title: Internal Tensor shape-entry query.
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
        title: Initialize one Tensor shape query.
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
            "TensorShape",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class TensorStride(astx.base.DataType):
    """
    title: Internal Tensor stride-entry query.
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
        title: Initialize one Tensor stride query.
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
            "TensorStride",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class TensorElementCount(astx.base.DataType):
    """
    title: Internal Tensor element-count query.
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
        title: Initialize one Tensor element-count query.
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
            "TensorElementCount",
            self.base.get_struct(simplified),
            simplified,
        )


@typechecked
class TensorByteOffset(astx.base.DataType):
    """
    title: Internal Tensor byte-offset query for indexed addressing.
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
        title: Initialize one Tensor byte-offset query.
        parameters:
          base:
            type: astx.AST
          indices:
            type: Sequence[astx.AST]
        """
        super().__init__()
        if not indices:
            raise ValueError("tensor indexing requires at least one index")
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
            "TensorByteOffset",
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class TensorRetain(astx.base.DataType):
    """
    title: Internal explicit retain for Tensor-backed storage.
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
        title: Initialize one Tensor retain helper.
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
            "TensorRetain",
            self.base.get_struct(simplified),
            simplified,
        )


@typechecked
class TensorRelease(astx.base.DataType):
    """
    title: Internal explicit release for Tensor-backed storage.
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
        title: Initialize one Tensor release helper.
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
            "TensorRelease",
            self.base.get_struct(simplified),
            simplified,
        )


__all__ = [
    "TensorByteOffset",
    "TensorElementCount",
    "TensorIndex",
    "TensorLiteral",
    "TensorNDim",
    "TensorRelease",
    "TensorRetain",
    "TensorShape",
    "TensorStore",
    "TensorStride",
    "TensorType",
    "TensorView",
]

"""
title: Tensor layout helpers layered on the builtin tensor runtime.
summary: >-
  Define IRx's backend-neutral Tensor metadata helpers on top of the canonical
  buffer/view substrate and the Arrow C++ backed tensor runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import astx

from public import public

from irx.buffer import (
    BUFFER_FLAG_VALIDITY_BITMAP,
    BufferHandle,
    BufferMutability,
    BufferOwnership,
    BufferViewMetadata,
    buffer_dtype_handle,
    buffer_view_flags,
)
from irx.builtins.collections.array_primitives import (
    ARRAY_PRIMITIVE_TYPE_SPECS,
)
from irx.typecheck import typechecked

TENSOR_LAYOUT_EXTRA = "tensor_layout"
TENSOR_ELEMENT_TYPE_EXTRA = "tensor_element_type"
TENSOR_FLAGS_EXTRA = "tensor_flags"
_DTYPE_ELEMENT_SIZE_BYTES = {
    buffer_dtype_handle(spec.name): spec.element_size_bytes
    for spec in ARRAY_PRIMITIVE_TYPE_SPECS.values()
    if spec.element_size_bytes is not None
}


@public
@typechecked
class TensorOrder(str, Enum):
    """
    title: Canonical contiguous layout order for Tensor helpers.
    """

    C = "C"
    F = "F"


@public
@typechecked
@dataclass(frozen=True)
class TensorLayout:
    """
    title: Static Tensor layout metadata.
    summary: >-
      Represent the logical rank, shape, strides, and byte offset of one Tensor
      value without duplicating the lower-level storage machinery.
    attributes:
      shape:
        type: tuple[int, Ellipsis]
      strides:
        type: tuple[int, Ellipsis]
      offset_bytes:
        type: int
    """

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset_bytes: int = 0

    @property
    def ndim(self) -> int:
        """
        title: Return the rank encoded by the layout.
        returns:
          type: int
        """
        return len(self.shape)


@typechecked
def _shape_extent(shape: tuple[int, ...]) -> int:
    """
    title: Multiply one shape tuple into a logical element count.
    parameters:
      shape:
        type: tuple[int, Ellipsis]
    returns:
      type: int
    """
    extent = 1
    for dim in shape:
        extent *= dim
    return extent


@public
@typechecked
def tensor_element_count(layout: TensorLayout) -> int:
    """
    title: Return the logical element count for one layout.
    parameters:
      layout:
        type: TensorLayout
    returns:
      type: int
    """
    return _shape_extent(layout.shape)


@public
@typechecked
def tensor_default_strides(
    shape: tuple[int, ...],
    item_size_bytes: int,
    *,
    order: TensorOrder = TensorOrder.C,
) -> tuple[int, ...]:
    """
    title: Return canonical byte strides for one contiguous Tensor shape.
    parameters:
      shape:
        type: tuple[int, Ellipsis]
      item_size_bytes:
        type: int
      order:
        type: TensorOrder
    returns:
      type: tuple[int, Ellipsis]
    """
    if item_size_bytes <= 0:
        raise ValueError("tensor item_size_bytes must be positive")
    if not shape:
        return ()

    strides = [0] * len(shape)
    stride = item_size_bytes

    if order is TensorOrder.C:
        indices = range(len(shape) - 1, -1, -1)
    else:
        indices = range(len(shape))

    for axis in indices:
        strides[axis] = stride
        stride *= max(shape[axis], 1)

    return tuple(strides)


@public
@typechecked
def tensor_is_c_contiguous(
    layout: TensorLayout,
    item_size_bytes: int,
) -> bool:
    """
    title: Return whether one layout matches canonical C-order strides.
    parameters:
      layout:
        type: TensorLayout
      item_size_bytes:
        type: int
    returns:
      type: bool
    """
    return layout.strides == tensor_default_strides(
        layout.shape,
        item_size_bytes,
        order=TensorOrder.C,
    )


@public
@typechecked
def tensor_is_f_contiguous(
    layout: TensorLayout,
    item_size_bytes: int,
) -> bool:
    """
    title: Return whether one layout matches canonical Fortran-order strides.
    parameters:
      layout:
        type: TensorLayout
      item_size_bytes:
        type: int
    returns:
      type: bool
    """
    return layout.strides == tensor_default_strides(
        layout.shape,
        item_size_bytes,
        order=TensorOrder.F,
    )


@public
@typechecked
def validate_tensor_layout(
    layout: TensorLayout,
) -> tuple[str, ...]:
    """
    title: Validate one static Tensor layout.
    parameters:
      layout:
        type: TensorLayout
    returns:
      type: tuple[str, Ellipsis]
    """
    errors: list[str] = []

    if len(layout.strides) != layout.ndim:
        errors.append("tensor stride length must match ndim")
    if any(dim < 0 for dim in layout.shape):
        errors.append("tensor shape dimensions must be non-negative")
    if any(stride < 0 for stride in layout.strides):
        errors.append("tensor strides must be non-negative")
    if layout.offset_bytes < 0:
        errors.append("tensor offset_bytes must be non-negative")

    return tuple(errors)


@public
@typechecked
def tensor_byte_bounds(
    layout: TensorLayout,
) -> tuple[int, int] | None:
    """
    title: Return the minimum and maximum element-start byte offsets.
    summary: >-
      The result is relative to the underlying data pointer. None means the
      logical layout has zero extent and therefore addresses no elements.
    parameters:
      layout:
        type: TensorLayout
    returns:
      type: tuple[int, int] | None
    """
    if tensor_element_count(layout) == 0:
        return None

    minimum = layout.offset_bytes
    maximum = layout.offset_bytes

    for dim, stride in zip(layout.shape, layout.strides, strict=True):
        if dim <= 1:
            continue
        span = (dim - 1) * stride
        if span < 0:
            minimum += span
        else:
            maximum += span

    return minimum, maximum


@public
@typechecked
def tensor_primitive_type_name(type_: astx.DataType | None) -> str | None:
    """
    title: Return the builtin primitive storage name for one Tensor element.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: str | None
    """
    if isinstance(type_, astx.Boolean):
        return "bool"
    if isinstance(type_, astx.Int8):
        return "int8"
    if isinstance(type_, astx.Int16):
        return "int16"
    if isinstance(type_, astx.Int32):
        return "int32"
    if isinstance(type_, astx.Int64):
        return "int64"
    if isinstance(type_, astx.UInt8):
        return "uint8"
    if isinstance(type_, astx.UInt16):
        return "uint16"
    if isinstance(type_, astx.UInt32):
        return "uint32"
    if isinstance(type_, astx.UInt64):
        return "uint64"
    if isinstance(type_, astx.Float32):
        return "float32"
    if isinstance(type_, astx.Float64):
        return "float64"
    return None


@public
@typechecked
def tensor_element_size_bytes(type_: astx.DataType | None) -> int | None:
    """
    title: Return the byte width for one Tensor element type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: int | None
    """
    primitive_name = tensor_primitive_type_name(type_)
    if primitive_name is None:
        return None
    spec = ARRAY_PRIMITIVE_TYPE_SPECS.get(primitive_name)
    if spec is None:
        return None
    return spec.element_size_bytes


@public
@typechecked
def tensor_buffer_dtype(type_: astx.DataType | None) -> BufferHandle | None:
    """
    title: Return the canonical buffer dtype handle for one Tensor element.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: BufferHandle | None
    """
    primitive_name = tensor_primitive_type_name(type_)
    if primitive_name is None:
        return None
    return buffer_dtype_handle(primitive_name)


@public
@typechecked
def tensor_buffer_view_metadata(
    *,
    data: BufferHandle,
    owner: BufferHandle,
    dtype: BufferHandle,
    layout: TensorLayout,
    ownership: BufferOwnership,
    mutability: BufferMutability,
    has_validity_bitmap: bool = False,
) -> BufferViewMetadata:
    """
    title: Bridge one Tensor layout into canonical buffer/view metadata.
    parameters:
      data:
        type: BufferHandle
      owner:
        type: BufferHandle
      dtype:
        type: BufferHandle
      layout:
        type: TensorLayout
      ownership:
        type: BufferOwnership
      mutability:
        type: BufferMutability
      has_validity_bitmap:
        type: bool
    returns:
      type: BufferViewMetadata
    """
    c_contiguous = False
    f_contiguous = False
    item_size_bytes = tensor_element_size_bytes_from_dtype(dtype)
    if item_size_bytes is not None:
        c_contiguous = tensor_is_c_contiguous(layout, item_size_bytes)
        f_contiguous = tensor_is_f_contiguous(layout, item_size_bytes)

    flags = buffer_view_flags(
        ownership,
        mutability,
        c_contiguous=c_contiguous,
        f_contiguous=f_contiguous,
    )
    if has_validity_bitmap:
        flags |= BUFFER_FLAG_VALIDITY_BITMAP

    return BufferViewMetadata(
        data=data,
        owner=owner,
        dtype=dtype,
        ndim=layout.ndim,
        shape=layout.shape,
        strides=layout.strides,
        offset_bytes=layout.offset_bytes,
        flags=flags,
    )


@typechecked
def tensor_element_size_bytes_from_dtype(dtype: BufferHandle) -> int | None:
    """
    title: Return the byte width for one canonical primitive dtype handle.
    parameters:
      dtype:
        type: BufferHandle
    returns:
      type: int | None
    """
    if dtype.is_null:
        return None
    return _DTYPE_ELEMENT_SIZE_BYTES.get(dtype)


__all__ = [
    "TENSOR_ELEMENT_TYPE_EXTRA",
    "TENSOR_FLAGS_EXTRA",
    "TENSOR_LAYOUT_EXTRA",
    "TensorLayout",
    "TensorOrder",
    "tensor_buffer_dtype",
    "tensor_buffer_view_metadata",
    "tensor_byte_bounds",
    "tensor_default_strides",
    "tensor_element_count",
    "tensor_element_size_bytes",
    "tensor_element_size_bytes_from_dtype",
    "tensor_is_c_contiguous",
    "tensor_is_f_contiguous",
    "tensor_primitive_type_name",
    "validate_tensor_layout",
]

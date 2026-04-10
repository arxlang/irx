"""
title: Low-level buffer/view substrate model.
summary: >-
  Define the canonical IRx buffer owner and buffer view contract that Arx may
  target without introducing a user-facing scientific array API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from public import public

from irx.typecheck import typechecked

BUFFER_VIEW_TYPE_NAME = "irx_buffer_view"
BUFFER_VIEW_METADATA_EXTRA = "buffer_view_metadata"
BUFFER_VIEW_FIELD_NAMES = (
    "data",
    "owner",
    "dtype",
    "ndim",
    "shape",
    "strides",
    "offset_bytes",
    "flags",
)
BUFFER_VIEW_FIELD_INDICES = {
    name: index for index, name in enumerate(BUFFER_VIEW_FIELD_NAMES)
}

BUFFER_FLAG_BORROWED = 1 << 0
BUFFER_FLAG_OWNED = 1 << 1
BUFFER_FLAG_EXTERNAL_OWNER = 1 << 2
BUFFER_FLAG_READONLY = 1 << 3
BUFFER_FLAG_WRITABLE = 1 << 4
BUFFER_FLAG_C_CONTIGUOUS = 1 << 5
BUFFER_FLAG_F_CONTIGUOUS = 1 << 6

BUFFER_OWNERSHIP_FLAGS = (
    BUFFER_FLAG_BORROWED,
    BUFFER_FLAG_OWNED,
    BUFFER_FLAG_EXTERNAL_OWNER,
)
BUFFER_MUTABILITY_FLAGS = (
    BUFFER_FLAG_READONLY,
    BUFFER_FLAG_WRITABLE,
)


@public
@typechecked
class BufferOwnership(str, Enum):
    """
    title: Buffer view ownership state.
    """

    BORROWED = "borrowed"
    OWNED = "owned"
    EXTERNAL_OWNER = "external_owner"


@public
@typechecked
class BufferMutability(str, Enum):
    """
    title: Buffer view mutability state.
    """

    READONLY = "readonly"
    WRITABLE = "writable"


@public
@typechecked
@dataclass(frozen=True)
class BufferHandle:
    """
    title: Static opaque handle reference for buffer descriptors.
    summary: >-
      Model a pointer-valued handle for semantic validation and deterministic
      lowering. None means the handle is statically null.
    attributes:
      address:
        type: int | None
    """

    address: int | None = None

    def __post_init__(self) -> None:
        """
        title: Validate one static handle.
        """
        if self.address is not None and self.address <= 0:
            raise ValueError(
                "non-null buffer handles must use a positive token"
            )

    @property
    def is_null(self) -> bool:
        """
        title: Return whether this handle is statically null.
        returns:
          type: bool
        """
        return self.address is None


@public
@typechecked
@dataclass(frozen=True)
class BufferViewMetadata:
    """
    title: Static metadata for one low-level buffer view descriptor.
    summary: >-
      Represents the semantic contents of the canonical IRx buffer view. This
      is not a scientific array object; it is a plain descriptor shape.
    attributes:
      data:
        type: BufferHandle
      owner:
        type: BufferHandle
      dtype:
        type: BufferHandle
      ndim:
        type: int
      shape:
        type: tuple[int, Ellipsis]
      strides:
        type: tuple[int, Ellipsis]
      offset_bytes:
        type: int
      flags:
        type: int
    """

    data: BufferHandle
    owner: BufferHandle
    dtype: BufferHandle
    ndim: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset_bytes: int
    flags: int


@public
@typechecked
def buffer_view_flags(
    ownership: BufferOwnership,
    mutability: BufferMutability,
    *,
    c_contiguous: bool = False,
    f_contiguous: bool = False,
) -> int:
    """
    title: Build canonical buffer view flags.
    parameters:
      ownership:
        type: BufferOwnership
      mutability:
        type: BufferMutability
      c_contiguous:
        type: bool
      f_contiguous:
        type: bool
    returns:
      type: int
    """
    flags = {
        BufferOwnership.BORROWED: BUFFER_FLAG_BORROWED,
        BufferOwnership.OWNED: BUFFER_FLAG_OWNED,
        BufferOwnership.EXTERNAL_OWNER: BUFFER_FLAG_EXTERNAL_OWNER,
    }[ownership]
    flags |= {
        BufferMutability.READONLY: BUFFER_FLAG_READONLY,
        BufferMutability.WRITABLE: BUFFER_FLAG_WRITABLE,
    }[mutability]
    if c_contiguous:
        flags |= BUFFER_FLAG_C_CONTIGUOUS
    if f_contiguous:
        flags |= BUFFER_FLAG_F_CONTIGUOUS
    return flags


@public
@typechecked
def buffer_flags_include(flags: int, flag: int) -> bool:
    """
    title: Return whether one buffer flag is set.
    parameters:
      flags:
        type: int
      flag:
        type: int
    returns:
      type: bool
    """
    return (flags & flag) == flag


@public
@typechecked
def buffer_view_is_readonly(flags: int) -> bool:
    """
    title: Return whether a buffer view is readonly.
    parameters:
      flags:
        type: int
    returns:
      type: bool
    """
    return buffer_flags_include(flags, BUFFER_FLAG_READONLY)


@public
@typechecked
def validate_buffer_view_metadata(
    metadata: BufferViewMetadata,
) -> tuple[str, ...]:
    """
    title: Validate static buffer view metadata.
    parameters:
      metadata:
        type: BufferViewMetadata
    returns:
      type: tuple[str, Ellipsis]
    """
    errors: list[str] = []

    ownership = [
        flag
        for flag in BUFFER_OWNERSHIP_FLAGS
        if buffer_flags_include(metadata.flags, flag)
    ]
    if len(ownership) != 1:
        errors.append(
            "buffer view must set exactly one ownership flag: "
            "borrowed, owned, or external_owner"
        )
    elif ownership[0] == BUFFER_FLAG_BORROWED and not metadata.owner.is_null:
        errors.append("borrowed buffer views must use a null owner handle")
    elif (
        ownership[0]
        in {
            BUFFER_FLAG_OWNED,
            BUFFER_FLAG_EXTERNAL_OWNER,
        }
        and metadata.owner.is_null
    ):
        errors.append("owned buffer views must use a non-null owner handle")

    mutability = [
        flag
        for flag in BUFFER_MUTABILITY_FLAGS
        if buffer_flags_include(metadata.flags, flag)
    ]
    if len(mutability) != 1:
        errors.append(
            "buffer view must set exactly one mutability flag: "
            "readonly or writable"
        )

    if metadata.ndim < 0:
        errors.append("buffer view ndim must be non-negative")
    if len(metadata.shape) != metadata.ndim:
        errors.append("buffer view shape length must match ndim")
    if len(metadata.strides) != metadata.ndim:
        errors.append("buffer view strides length must match ndim")
    if any(dim < 0 for dim in metadata.shape):
        errors.append("buffer view shape dimensions must be non-negative")
    if metadata.offset_bytes < 0:
        errors.append("buffer view offset_bytes must be non-negative")
    if metadata.dtype.is_null:
        errors.append("buffer view dtype handle must be non-null")

    extent = 1
    for dim in metadata.shape:
        extent *= dim
    if extent > 0 and metadata.data.is_null:
        errors.append("buffer view with nonzero extent must use non-null data")

    return tuple(errors)


__all__ = [
    "BUFFER_FLAG_BORROWED",
    "BUFFER_FLAG_C_CONTIGUOUS",
    "BUFFER_FLAG_EXTERNAL_OWNER",
    "BUFFER_FLAG_F_CONTIGUOUS",
    "BUFFER_FLAG_OWNED",
    "BUFFER_FLAG_READONLY",
    "BUFFER_FLAG_WRITABLE",
    "BUFFER_MUTABILITY_FLAGS",
    "BUFFER_OWNERSHIP_FLAGS",
    "BUFFER_VIEW_FIELD_INDICES",
    "BUFFER_VIEW_FIELD_NAMES",
    "BUFFER_VIEW_METADATA_EXTRA",
    "BUFFER_VIEW_TYPE_NAME",
    "BufferHandle",
    "BufferMutability",
    "BufferOwnership",
    "BufferViewMetadata",
    "buffer_flags_include",
    "buffer_view_flags",
    "buffer_view_is_readonly",
    "validate_buffer_view_metadata",
]

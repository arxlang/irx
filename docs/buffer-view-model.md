# Buffer/View Model

IRx defines one canonical low-level memory/container substrate: a buffer owner
handle plus a buffer view descriptor. This is not a user-facing scientific array
API. It does not define NumPy semantics, broadcasting, slicing syntax,
reductions, tensor algebra, or high-level dtype policy.

The model exists so Arx can lower scientific data structures into stable plain
ABI data without baking array-library behavior into IRx.

## Canonical Representation

The canonical view descriptor is the identified struct `irx_buffer_view` with
this stable field order:

| Index | Field          | Meaning                                  |
| ----- | -------------- | ---------------------------------------- |
| 0     | `data`         | Opaque data pointer.                     |
| 1     | `owner`        | Opaque owner handle or null.             |
| 2     | `dtype`        | Opaque dtype handle or stable token.     |
| 3     | `ndim`         | Rank as `i32`.                           |
| 4     | `shape`        | Pointer to `i64` shape metadata.         |
| 5     | `strides`      | Pointer to `i64` stride metadata.        |
| 6     | `offset_bytes` | Byte offset from `data`.                 |
| 7     | `flags`        | Ownership, mutability, and layout flags. |

The lowered LLVM shape is:

```llvm
%"irx_buffer_view" = type {i8*, i8*, i8*, i32, i64*, i64*, i64, i32}
```

There are no hidden headers or backend-only object layouts. IRx lowers the
descriptor as a plain struct value, consistent with the project struct ABI
foundation.

## Ownership

Ownership is explicit in the descriptor flags and is never inferred from
unrelated fields.

- Borrowed views set the borrowed flag, use a null owner handle, and do not free
  memory.
- Owned views set the owned flag and use a non-null owner handle managed by
  runtime/native helpers.
- External-owner views set the external-owner flag and use a non-null owner
  handle representing imported storage or host-managed lifetime.
- Exactly one ownership flag must be set.
- Copying a descriptor copies metadata only.
- Descriptor copies are shallow.
- Deep copy is explicit and is not performed by assignment, argument passing, or
  generic lowering.
- Retain and release are explicit runtime/native operations, not hidden generic
  lowering behavior.

The current `buffer` runtime feature provides opaque owner retain/release
helpers and view retain/release helpers. The owner handle remains opaque in LLVM
IR.

## Mutability

Mutability is attached to the view.

- Readonly views set the readonly flag.
- Writable views set the writable flag.
- Exactly one mutability flag must be set.
- Writes through statically readonly views are rejected during semantic
  analysis.
- Mutable borrowed views are allowed when the producer's storage contract makes
  that safe.

This deliberately separates "may write through this view" from "who owns the
underlying allocation."

## Shape And Strides

Shape and strides describe logical indexing, not ownership.

- `ndim` must be non-negative.
- Shape length must match `ndim`.
- Stride length must match `ndim`.
- Shape dimensions must be non-negative.
- `offset_bytes` must be non-negative.
- Null data with nonzero static extent is rejected.
- Contiguity is represented as explicit or derivable metadata; rank alone does
  not imply contiguity.

The descriptor supports contiguous and non-contiguous views. IRx does not add
high-level slicing or broadcasting semantics in this layer.

## Runtime And Native Boundaries

IRx-internal lowering may pass buffer views as plain struct values where that is
consistent with the existing struct-lowering rules. Runtime/native boundaries
use explicit helper calls and pointer-to-descriptor conventions for lifetime
operations.

The `buffer` runtime feature is activated only when a buffer helper symbol is
requested, such as `irx_buffer_view_retain` or `irx_buffer_view_release`. Plain
descriptor lowering does not drag in native runtime artifacts.

The runtime boundary is intentionally conservative:

- owner handles are opaque at the IR level
- retain/release are explicit calls
- ownership transfer is a helper-level concern
- descriptor copies remain shallow metadata copies
- no particular high-level array library is assumed

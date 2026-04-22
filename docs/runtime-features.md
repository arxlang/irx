# Runtime Features

IRx lowers ASTx nodes to LLVM IR with `llvmlite`, but some capabilities are
better modeled as calls into a native runtime instead of handwritten container
logic in LLVM IR. The runtime-feature system exists for exactly that boundary.

## Why This Exists

IRx already had a small precedent for external/native symbols such as `puts`.
That worked for a few direct libc calls, but it did not provide a maintainable
way to:

- declare external symbols once per feature
- activate native capabilities per compilation unit
- compile and link native C sources only when they are actually needed
- keep native runtime ownership rules outside the LLVM IR middle-end

This runtime-feature layer keeps IRx focused on lowering while allowing Arx to
grow additional native integrations later. It is also the public
native-dependency side of IRx's stable FFI contract.

## Architecture

The runtime stack is layered in four parts:

1. `irx.builder.runtime.features` Defines feature specs: external symbols,
   native artifacts, linker flags, and metadata.
2. `irx.builder.runtime.registry` Registers features by name and tracks
   activation/declarations for one LLVM module.
3. `irx.builder.runtime.linking` Compiles native C sources and links optional
   objects only for active features.
4. Feature packages such as `libc` and `array` Consume the generic system
   without special cases in the builder.

## Activation Model

Runtime features are named and activated per compilation unit. The builtin array
runtime is packaged with IRx even though its native artifacts are linked only
when needed.

- `libc` Declares symbols such as `puts`, `malloc`, and `snprintf`.
- `assertions` Declares `__arx_assert_fail(...)` and links the native fatal
  assertion helper that emits machine-readable stderr reports.
- `libm` Declares math symbols such as `sqrt` and contributes `-lm`.
- `buffer` Declares the low-level buffer owner/view lifetime helper ABI.
- `array` Declares the builtin array runtime surface and links the Arrow-backed
  native implementation.

The builder and visitor cooperate as follows:

- explicit extern declarations may declare `runtime_feature` or
  `runtime_features` on `FunctionPrototype`
- lowering requests feature-owned symbols through
  `require_runtime_symbol(feature, symbol)`
- the request activates the feature for that compilation unit
- the linker step collects native artifacts only from active features
- inactive features contribute nothing to the link command

This is intentionally separate from any future language-level import or module
system. A future Arx array-facing layer can decide when to activate `array`, but
the native integration remains owned by IRx.

## Extern Declarations And Feature-Backed Linking

Public FFI declarations now use one consistent rule:

- extern declarations with no runtime features emit only an LLVM external
  declaration and rely on the system linker/toolchain to resolve the symbol
- extern declarations with `runtime_feature` / `runtime_features` still lower as
  ordinary externs, but they also activate the named runtime features for that
  compilation unit
- if a runtime feature already owns a matching symbol declaration, lowering
  reuses that feature-owned declaration instead of inventing a parallel native
  path
- runtime features remain the only place where IRx packages native objects,
  native C sources, or extra linker flags

Example split:

- plain `puts` extern: system linker resolution only
- `sqrt` extern with `runtime_feature = "libm"`: LLVM declaration plus the
  `libm` feature's `-lm` linker flag
- Array helpers: IRx-owned nodes imply the `array` feature and its packaged
  native runtime

## External Symbols

External declarations are centralized inside each feature definition instead of
being scattered through visitor methods.

Benefits:

- declarations are reused per module
- function signatures live in one place
- future features can add their own symbol sets without changing the linker or
  builder architecture

## Native Linking

IRx still emits the main object file with `llvmlite` and links with `clang`. The
difference now is that runtime features may add native artifacts such as:

- C source files
- prebuilt objects
- static libraries

The current builtin array runtime uses C sources only, which keeps the build
path reproducible on Linux and macOS without introducing dynamic loading.

## Assertion Failure Reporting

The `assertions` runtime feature exists for fatal `AssertStmt` lowering. Its
native helper writes one machine-readable line to `stderr` before exiting the
process with a non-zero status:

```text
ARX_ASSERT_FAIL|<source>|<line>|<col>|<message>
```

IRx also exposes small Python-side parsing helpers under
`irx.builder.runtime.assertions` so higher-level runners can extract one stable
report from `stderr` without scraping human-oriented text. Source and message
payloads escape backslashes, newlines, carriage returns, tabs, and protocol
delimiters before printing so the report always remains one physical line. The
source field uses the analyzed module display name when available and otherwise
falls back to the module name stored in the AST.

## Builtin Array Runtime

IRx array support is implemented as a builtin native runtime backed by Arrow,
not as handwritten LLVM IR container logic.

Current array substrate:

- opaque runtime handles for schemas, array builders, and arrays
- supported primitive storage types: `int8`, `int16`, `int32`, `int64`, `uint8`,
  `uint16`, `uint32`, `uint64`, `float32`, `float64`, and `bool`
- explicit builder / import / inspect / export / release lifecycle
- Arrow C Data import/export support with copy and move/adopt imports
- explicit nullability and validity-bitmap inspection on Arrow handles
- readonly bridge from supported fixed-width numeric arrays into
  `irx_buffer_view`
- Python `nanoarrow` dependency installed by default in IRx
- `nanoarrow` used internally for schema/array helpers and validation

Current initial NDArray layer on top of that substrate:

- ndarray values lower through the same `irx_buffer_view` descriptor used by the
  low-level buffer/view model
- ndarray construction uses Arrow-backed array storage plus a buffer-owner
  bridge so the view can manage Arrow-backed lifetime explicitly
- indexing and byte-offset calculation reuse descriptor `shape`, `strides`, and
  `offset_bytes`
- shallow ndarray views may replace shape/stride/offset metadata without
  creating a second storage runtime
- current ndarray lowering supports fixed-width numeric element types only

What IRx does not do here:

- no direct LLVM struct encoding of Arrow containers
- no full Arrow type system
- no Arx language syntax or module layer
- no RecordBatch, Table, or ArrowArrayStream runtime yet
- no dataframe/query semantics or compute-kernel surface

## ABI Boundary

The public high-level abstraction is array-oriented, while the low-level ABI
exposed to generated LLVM IR and native harnesses remains the IRx-owned Arrow C
ABI under `irx_arrow_*`.

Key rules:

- handles are opaque pointers
- runtime-owned memory is released with explicit `irx_arrow_*_release()`
- `nanoarrow` stays internal to the implementation
- Arrow C Data structs are the interchange boundary
- import is explicit:
  - `irx_arrow_array_import_copy(...)` copies external C Data into a new
    runtime-owned array handle
  - `irx_arrow_array_import_move(...)` adopts external C Data into a new
    runtime-owned array handle and leaves the input structs moved-from on
    success
- export is explicit:
  - `irx_arrow_array_export(...)` copies a runtime-owned array handle into an
    independent Arrow C Data pair that the caller releases separately
- schema handles use the same pattern through
  `irx_arrow_schema_import_copy(...)` and `irx_arrow_schema_export(...)`

## Ownership Rules

Current ownership model:

- builder handles own their mutable Arrow builder state
- finishing a builder transfers ownership into an immutable array handle
- schema and array handles are refcounted through explicit retain/release calls
- array handles own their schema plus array resources
- exported Arrow C Data structs own their copied resources and must be released
  independently
- copied imports leave the caller's Arrow C Data ownership unchanged
- move/adopt imports transfer ownership into IRx on success

## Nullability And Buffer Bridges

Arrow nullability stays Arrow-specific in this layer.

- Arrow arrays may be nullable independently of `irx_buffer_view`
- `irx_arrow_array_is_nullable(...)`, `irx_arrow_array_null_count(...)`, and
  `irx_arrow_array_has_validity_bitmap(...)` expose Arrow-side null metadata
- `irx_arrow_array_validity_bitmap(...)` exposes the physical validity bitmap
  pointer plus bit offset and length
- `irx_buffer_view` remains a plain physical view; generic indexing and writes
  do not become null-aware
- `irx_arrow_array_borrow_buffer_view(...)` projects only the physical value
  buffer and always returns a borrowed readonly `irx_buffer_view`
- when a bridged Arrow array has a validity bitmap, the returned view sets
  `IRX_BUFFER_FLAG_VALIDITY_BITMAP`
- bool arrays are supported as Arrow handles but are not buffer-view compatible
  because their values are bit-packed
- caller code that needs null semantics must keep using Arrow inspection APIs

The buffer bridge is intentionally conservative:

- only fixed-width byte-addressable primitive arrays are bridged
- the bridge is 1-D and columnar (`shape[0] == length`,
  `stride == element_size`)
- writable views are not exposed in this phase
- borrowed views use a null owner handle, so the caller must keep the Arrow
  array handle alive explicitly

The NDArray layer builds on top of this bridge rather than bypassing it:

- fresh ndarray literals allocate Arrow-backed storage, then wrap that storage
  in an external-owner `irx_buffer_view`
- ndarray views stay shallow and metadata-driven
- readonly semantics are preserved for Arrow-backed NDArrays in this phase

## Nanoarrow

IRx now depends on the Python `nanoarrow` package by default and uses it in the
test suite to validate Arrow C Data interoperability against the installed
package.

IRx also depends on `arx-nanoarrow-sources` for the generated nanoarrow header
and C source bundle used by the native Arrow runtime feature itself.

Reasons:

- the installed Python `nanoarrow` package does not ship the raw `nanoarrow.h`
  header or C sources that IRx compiles into its native runtime
- reproducible native builds in CI and local development without keeping a
  second nanoarrow copy inside the IRx repo
- clear ownership of the narrow C runtime surface while keeping `nanoarrow`
  hidden behind the IRx ABI

IRx compiles the packaged nanoarrow sources with
`-DNANOARROW_NAMESPACE=IrxNanoarrow` to keep those helper symbols internal to
the feature implementation.

## Buffer As A Runtime Feature

The `buffer` feature owns lifetime-sensitive helper operations for the canonical
buffer/view substrate. Plain `irx_buffer_view` descriptors lower as structs and
do not activate this feature. Explicit helper calls such as
`irx_buffer_view_retain` and `irx_buffer_view_release` activate it.

The feature keeps owner handles opaque at the IR level. Native code may retain
or release an owner handle, but generic lowering does not infer ownership
transfer or emit hidden retains/releases for descriptor copies. Statically known
borrowed views are rejected before retain/release lowering; descriptor-pointer
runtime calls are reserved for owned or external-owner views.

## What Exists Now

Implemented in this phase:

- generic runtime-feature registry/state/linking
- `libc` routed through the new feature system
- low-level `buffer` runtime feature for owner/view retain-release helpers
- builtin array runtime feature with packaged nanoarrow sources
- Python `nanoarrow` dependency and direct interop tests
- centralized Arrow runtime symbol declarations
- one internal array lowering path: `irx.astx.ArrayInt32ArrayLength`
- tests for registry behavior, IR declarations, build integration, primitive
  type coverage, nullability, move/copy ownership, and Arrow-to-buffer-view
  projection

## Follow-up Roadmap

Phase 2:

- string and binary arrays
- richer schema helpers
- better Arrow import/export diagnostics

Phase 3:

- RecordBatch and Table handles
- ArrowArrayStream support
- richer stream-oriented interop helpers

Phase 4:

- limited native compute kernels where justified
- optional Arrow compute backend evaluation if a future Arx layer needs it

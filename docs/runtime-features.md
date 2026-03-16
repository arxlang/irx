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
grow optional native integrations later.

## Architecture

The runtime stack is layered in four parts:

1. `irx.runtime.features` Defines feature specs: external symbols, native
   artifacts, linker flags, and metadata.
2. `irx.runtime.registry` Registers features by name and tracks
   activation/declarations for one LLVM module.
3. `irx.runtime.linking` Compiles native C sources and links optional objects
   only for active features.
4. Feature packages such as `libc` and `arrow` Consume the generic system
   without special cases in the builder.

## Activation Model

Runtime features are named, optional, and per-compilation-unit.

- `libc` Declares symbols such as `puts`, `malloc`, and `snprintf`.
- `arrow` Declares the IRx-owned Arrow runtime ABI and links the native Arrow
  runtime.

The builder and visitor cooperate as follows:

- lowering requests feature-owned symbols through
  `require_runtime_symbol(feature, symbol)`
- the request activates the feature for that compilation unit
- the linker step collects native artifacts only from active features
- inactive features contribute nothing to the link command

This is intentionally separate from any future language-level import or module
system. A future Arx `std.arrow` layer can decide when to activate `arrow`, but
the native integration remains owned by IRx.

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

The current Arrow feature uses C sources only, which keeps the build path
reproducible on Linux and macOS without introducing dynamic loading.

## Arrow As A Runtime Feature

Arrow support is implemented as an optional native runtime, not as handwritten
LLVM IR container logic.

Current Arrow MVP:

- opaque runtime handles only
- primitive `int32` arrays only
- explicit create / append / finish / inspect / release lifecycle
- Arrow C Data import/export support
- Python `nanoarrow` dependency installed by default in IRx
- `nanoarrow` used internally for schema/array helpers and validation

What IRx does not do here:

- no direct LLVM struct encoding of Arrow containers
- no full Arrow type system
- no Arx language syntax or module layer
- no RecordBatch, Table, or ArrowArrayStream runtime yet

## ABI Boundary

The public ABI exposed to generated LLVM IR and native harnesses is an IRx-owned
C ABI under `irx_arrow_*`.

Key rules:

- handles are opaque pointers
- runtime-owned memory is released with explicit `irx_arrow_*_release()`
- `nanoarrow` stays internal to the implementation
- Arrow C Data structs are the interchange boundary

The Arrow runtime currently copies arrays on import/export. That keeps ownership
simple for the first phase and avoids leaking runtime-private storage details.

## Ownership Rules

Current ownership model:

- builder handles own their mutable Arrow builder state
- finishing a builder transfers ownership into an immutable array handle
- array handles own their schema plus array resources
- exported Arrow C Data structs own their copied resources and must be released
  independently
- imported Arrow C Data values are copied into a new IRx array handle

Nullable arrays are intentionally deferred to the next phase, so MVP import
currently rejects arrays with nulls.

## Nanoarrow

IRx now depends on the Python `nanoarrow` package by default and uses it in the
test suite to validate Arrow C Data interoperability against the installed
package.

IRx still vendors a pinned copy of `apache-arrow-nanoarrow-0.6.0` for the native
Arrow runtime feature itself.

Reasons:

- the installed Python package does not ship the raw `nanoarrow.h` header or C
  sources that IRx compiles into its native runtime
- reproducible native builds in CI and local development
- clear ownership of the narrow C runtime surface while keeping `nanoarrow`
  hidden behind the IRx ABI

IRx compiles the vendored nanoarrow sources with
`-DNANOARROW_NAMESPACE=IrxNanoarrow` to keep those helper symbols internal to
the feature implementation.

## What Exists Now

Implemented in this phase:

- generic runtime-feature registry/state/linking
- `libc` routed through the new feature system
- Arrow native runtime feature with vendored nanoarrow
- Python `nanoarrow` dependency and direct interop tests
- centralized Arrow runtime symbol declarations
- one internal Arrow lowering path: `irx.system.ArrowInt32ArrayLength`
- tests for registry behavior, IR declarations, build integration, runtime ABI,
  and Arrow C Data roundtrip

## Follow-up Roadmap

Phase 2:

- nullable primitive arrays
- string arrays
- richer schema helpers
- better Arrow import/export diagnostics

Phase 3:

- RecordBatch and Table handles
- ArrowArrayStream support
- more primitive element types

Phase 4:

- limited native compute kernels where justified
- optional Arrow compute backend evaluation if the Arx layer needs it

# Architecture

IRx is organized as a small compiler pipeline with a deliberate boundary between
semantic meaning and backend-specific lowering. The goal is to keep the codebase
easy to extend without letting semantic rules slowly drift into code generation.

## Design Goals

The current architecture is shaped by a few practical goals:

- Keep parsing, semantic analysis, and code generation as distinct phases.
- Make semantic analysis the authority for meaning and program validity.
- Keep backend packages focused on emission, not interpretation.
- Preserve method-based multiple dispatch for visitor-driven lowering.
- Use package structure to communicate architecture instead of large utility
  modules or generic `helpers/` folders.

## Pipeline Overview

IRx currently follows this high-level flow:

`ASTx parser output -> semantic analysis -> resolved semantic sidecars -> backend code generation`

The parser produces raw ASTx nodes. Those nodes are still close to surface
syntax and may not yet have enough information for direct lowering. The
semantic-analysis phase walks that tree, resolves symbols and types, validates
program rules, and attaches a structured `node.semantic` sidecar to the nodes
that backend code needs.

By the time a backend starts lowering, it should not need to infer meaning from
raw syntax or re-run language validation from scratch.

## Semantic Analysis

The semantic-analysis package lives in `src/irx/analysis/` and is intentionally
independent from LLVM or `llvmlite`.

It is responsible for:

- symbol resolution
- lexical scope tracking
- mutability and assignment validation
- function and return validation
- loop-control legality such as `break` and `continue`
- expression typing and promotion policy
- operator normalization
- semantic flag normalization such as unsigned and fast-math intent
- diagnostics collection and semantic error reporting

The public entry points are:

- `irx.analysis.analyze(node)`
- `irx.analysis.analyze_module(module)`
- `irx.analysis.analyze_modules(root, resolver)`

These entry points return the same AST root after attaching semantic sidecars.
If semantic validation fails, analysis raises `SemanticError` before codegen
begins.

### Semantic Contract

The host-facing semantic boundary is now explicit in code through
`irx.analysis.get_semantic_contract()`. That contract names the stable semantic
phases, the `SemanticInfo` and `CompilationSession` metadata that must exist
before codegen, and the boundary between semantic, lowering, and linking/runtime
failures.

See [Semantic Contract](semantic-contract.md) for the concise contract summary.

### Why sidecars instead of a separate HIR?

For the current size of IRx, attaching explicit semantic sidecars to AST nodes
is the lightest approach that still creates a clean boundary. It gives codegen
resolved information without introducing a second full tree structure before it
is needed.

If the language grows to the point where a true HIR becomes useful, the current
phase split still leaves room for that evolution.

### Multi-Module Boundary

IRx now also supports a parser-agnostic multi-module path for imports.

The boundary is explicit:

- the host compiler parses source text into `astx.Module` objects
- the host compiler decides how an import specifier maps to a module
- IRx receives `ParsedModule` objects plus an `ImportResolver`
- IRx expands the reachable dependency graph, performs cross-module semantic
  analysis, and lowers the reachable graph into one LLVM module for the MVP

Import-from resolution remains symbol-first, but it also supports child-module
namespace sugar: `import stats from sciarx` may bind `sciarx.stats` as a local
module namespace when `sciarx` does not already expose an importable symbol
named `stats`.

IRx does not parse source text, search the filesystem, or implement package
discovery. Those responsibilities stay outside the library.

### Template Specialization Metadata

IRx also carries semantic-only template metadata for compile-time
specialization. The current scope is bounded template functions and methods.

Semantic analysis preserves:

- template parameters attached to callable definitions
- finite union bounds used as specialization domains
- unresolved template type variables inside generic signatures
- explicit template arguments attached to call sites
- stable specialization identities and generated concrete callables

During analysis, template bodies are validated over every admissible bound
substitution. Successful specializations are materialized as generated concrete
functions so backend lowering can continue to operate mostly on ordinary
non-template callables. That generated specialization set is treated as
per-analysis state and is cleared before rerunning semantic analysis on the same
AST module.

For v1, template methods lower only as direct concrete specializations. They do
not participate in class dispatch slots or virtual-style dispatch tables.

### Compilation Session

The multi-module path is centered on `CompilationSession` in
`src/irx/analysis/session.py`.

That session owns:

- the root parsed module and resolver callback
- the cache of reachable parsed modules
- the import dependency graph and stable load order
- cycle diagnostics
- per-module visible top-level bindings used for direct imports and module
  namespace aliases
- semantic-only module namespace values and namespace-member lookup metadata

Semantic identity for top-level functions and structs is module-aware. Backend
lowering consumes that semantic identity rather than raw source names, which is
what keeps same-named declarations in different modules from colliding in LLVM.

## Shared Visitor Foundation

IRx also has a shared visitor layer in `src/irx/base/visitors/`.

It currently provides:

- `BaseVisitorProtocol`: the minimal typing contract shared by visitor-style
  classes
- `BaseVisitor`: a concrete Plum-dispatch scaffold with explicit
  `NotImplementedError` defaults for the current ASTx node surface

This keeps typing and runtime behavior separate:

- protocols define what visitor-like objects must expose
- the concrete base class defines what happens for unsupported nodes

In practice:

- `SemanticAnalyzer` inherits `BaseVisitor`
- `BuilderVisitor` inherits `BaseVisitor`
- builder-specific protocols such as `builder.VisitorProtocol` extend
  `BaseVisitorProtocol`

## Builder Architecture

IRx now exposes a single builder package at `src/irx/builder/`. The package path
identifies the concrete LLVM builder, while the public classes inside it use
short generic names.

For example, `src/irx/builder` exposes:

- `Builder`
- `Visitor`
- `VisitorProtocol`
- optional `VisitorCore` as a module-private implementation class

This keeps the public API concise without reintroducing legacy class prefixes.

## Builder Package Layout

The LLVM backend is split into first-class modules instead of one monolithic
builder:

- `../src/irx/base/visitors/`: shared visitor protocol and runtime scaffold
- `backend.py`: public backend entry points
- `core.py`: shared mutable lowering state and backend lifecycle
- `protocols.py`: typing contract used by mixins and runtime features
- `types.py`, `casting.py`, `vector.py`, `strings.py`, `runtime/`: shared IR
  infrastructure
- `lowering/`: concern-grouped `visit(...)` overloads
- `../src/irx/buffer.py`: the canonical low-level buffer owner/view semantic
  substrate that Arx can target without exposing an array API

Foundational modules stay at the package root because they are architectural
components, not incidental helpers.

## Buffer/View Indexing

IRx treats first-class indexing as a low-level operation over the canonical
buffer/view descriptor in `src/irx/buffer.py`. It is the stable memory/container
path that Arx can target for element access such as `a[i]`, `a[i, j]`, and the
corresponding stores. It is not a NumPy-like array API and does not define
slicing, broadcasting, fancy indexing, masks, or shape inference.

Indexed access has an explicit IRx node surface for reads and stores. Semantic
analysis validates the descriptor base, the number of indices, index scalar
types, mutability for stores, static bounds when descriptor shape and literal
indices make the answer provable, and the scalar element type used by lowering.
The MVP requires static descriptor metadata for rank validation. Dynamic-rank
runtime checks are intentionally deferred.

Backend lowering keeps address computation separate from load/store emission.
The address helper extracts descriptor fields through
`BUFFER_VIEW_FIELD_INDICES`, starts from `data`, includes `offset_bytes`, loads
byte strides from `strides`, and computes:

`effective_byte_offset = offset_bytes + sum(index_k * stride_k)`

The result is cast to the resolved element pointer type. Indexed reads emit a
load from that pointer; indexed stores cast the right-hand side to the resolved
element type and emit a store. The default bounds policy means semantic static
bounds rejection when provable and no emitted runtime bounds helper yet. Future
checked and unchecked runtime modes can reuse the same element-pointer helper.

## Dynamic List Construction

IRx also exposes one intentionally small list-building surface for frontend-
emitted AST:

- `ListCreate(element_type)` creates an empty list value with explicit element
  type
- `ListAppend(base, value)` grows a mutable list variable or field
- regular `SubscriptExpr` lowering may read from produced list values

This is deliberately narrower than a full collection API. The goal is to let
frontends author pure source routines that accumulate list results inside loops
without moving collection policy into the frontend. The current runtime owns
append/growth and indexed reads only; list teardown is intentionally deferred to
a future ownership API.

## Common Collection Methods

IRx also exposes backend-neutral query nodes for common collection operations:

- `CollectionLength(base)` returns the logical length as `Int32`
- `CollectionIsEmpty(base)` returns a Boolean emptiness check
- `CollectionContains(base, value)` checks list, tuple, or set values and dict
  keys
- `CollectionIndex(base, value)` returns the first list/tuple index or `-1`
- `CollectionCount(base, value)` returns the number of list/tuple matches

Semantic analysis validates the receiver kind, probe type, and result type and
attaches a `ResolvedCollectionMethod` sidecar. Lowering consumes that sidecar
instead of re-resolving the collection operation from raw AST shape.

Literal lists, tuples, sets, and dictionaries support common length, emptiness,
and containment queries. Dynamic IRx lists also support length, emptiness,
contains, index, and count by reusing the existing list runtime and emitting
small search loops where needed. Dynamic set and dictionary method lowering
remains intentionally deferred until those runtime representations exist.

## Iterable Semantics

IRx now models iteration as a semantic capability instead of as backend-specific
collection probing. Semantic analysis resolves known iterable expressions into a
`ResolvedIteration` sidecar that records the adapter kind, yielded element type,
ordering contract, and loop/comprehension target symbol. Backend lowering
consumes that sidecar instead of rediscovering whether an expression is a list,
set, or dict.

The executable MVP supports `ForInLoopStmt` and `ListComprehension` over list
iterables, including literal lists and dynamic IRx lists. List iteration follows
index order and evaluates the iterable once when that loop or comprehension
clause is entered. Dict and set literals are recognized semantically as
iterables as well: dict iteration yields keys, while set iteration order remains
unspecified. Their dynamic lowering is intentionally guarded until IRx has
runtime-backed dynamic dict and set construction APIs.

## NDArray Layering

IRx now treats NDArray support as a distinct semantic layer built on two
existing foundations:

- the builtin Arrow-backed array runtime provides storage and backend
  interoperability
- the canonical `irx_buffer_view` substrate provides rank, shape, strides,
  offset, and layout flags

That split keeps high-level naming backend-neutral:

- `array` remains the storage/runtime-oriented abstraction
- `buffer/view` remains the low-level layout and ownership substrate
- `ndarray` is the multidimensional semantic abstraction layered on top

Current ndarray lowering stays intentionally conservative:

- literals build flat Arrow arrays, then wrap them in external-owner buffer
  views
- indexing and byte-offset queries reuse buffer/view stride arithmetic
- view construction is shallow and metadata-driven
- fixed-width numeric element types are supported in this phase
- Arrow-backed NDArrays remain readonly in this phase

## Why `visit(...)` Remains the Public Lowering Boundary

The codegen layer continues to use method-based Plum multiple dispatch:

- `visit(self, node: ...)`

This remains the only public dispatch boundary for backend lowering. IRx does
not use a free-function dispatch registry or a second public API like
`lower(...)` or `build_node(...)`.

That choice keeps backend code readable and local:

- AST-family-specific lowering remains attached to the visitor class.
- Mixins can group overloads by concern without changing the public surface.
- Shared lowering state stays on the visitor instance instead of moving into a
  registry-driven design.

## Core Class and Protocol

`VisitorProtocol` and `VisitorCore` serve different purposes:

- `VisitorProtocol` defines the stable interface that mixins and runtime feature
  declarations depend on for typing, building on `BaseVisitorProtocol`.
- `VisitorCore` is the concrete implementation center that owns mutable state,
  module setup, helper methods, and backend lifecycle.

`VisitorCore` is still internal to the backend package. IRx uses
`from public import private` for module-level internal helpers and internal
implementation classes when a clear non-underscored name reads better than an
underscore-prefixed export. That keeps internal names readable without making
them part of the intended public surface.

The protocol is not a replacement for the core class. It exists so backend
subsystems can depend on a narrow contract instead of the full concrete type.

## Visitor Mixins

The final backend visitor is composed from concern-specific mixins plus the
shared core. Each mixin should contain:

- `@dispatch def visit(self, node: ...)` overloads for one concern
- a small number of private helpers local to that concern

Examples of concern boundaries include:

- literals
- variables
- unary and binary operators
- control flow
- functions
- runtime or domain-specific lowering

This keeps dispatch organization aligned with language structure while still
sharing one lowering state object.

## Canonical Loop Lowering

IRx now treats loop lowering as one small shared control-flow contract instead
of three ad hoc visitors:

- `while`: `cond -> body -> exit`, with `continue` targeting `cond`
- `for-count`: `cond -> body -> update -> exit`, with `continue` targeting
  `update`
- `for-range`: `cond -> body -> step -> exit`, with `continue` targeting `step`

Loop variables remain semantic symbols rather than backend-only temporaries.
For-count initializers are visible only within the loop. For-range induction
variables are body-visible, loop-scoped, and restored after lowering so outer
shadowed bindings remain stable. Mutable post-loop state reconciles through the
existing variable-slot model instead of accidental value-stack state.

## Contributor Guidelines

When extending IRx, these rules help preserve the architecture:

- Put semantic meaning and validation in `analysis/`, not in a backend.
- Let codegen consume normalized semantic information instead of re-deriving it.
- Keep buffer/view support framed as a low-level memory/container substrate, not
  as NumPy-like user-facing array behavior.
- Keep shared visitor dispatch defaults in `src/irx/base/visitors/` so semantic
  and backend visitors fail consistently for unsupported ASTx nodes.
- Add new backend-wide infrastructure at the package root, not under `helpers/`.
- Keep mutable lowering state instance-local.
- Prefer explicit code over clever abstractions.
- Use the package name, not class prefixes, to identify the backend.

## If Another Backend Ever Returns

IRx currently standardizes on a single builder package. If another backend is
ever introduced again, keep the public class names generic and make the package
split an explicit architecture decision instead of quietly rebuilding a plural
builders namespace.

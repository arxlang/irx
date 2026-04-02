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

These entry points return the same AST root after attaching semantic sidecars.
If semantic validation fails, analysis raises `SemanticError` before codegen
begins.

### Why sidecars instead of a separate HIR?

For the current size of IRx, attaching explicit semantic sidecars to AST nodes
is the lightest approach that still creates a clean boundary. It gives codegen
resolved information without introducing a second full tree structure before it
is needed.

If the language grows to the point where a true HIR becomes useful, the current
phase split still leaves room for that evolution.

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
- backend-specific protocols such as `llvmliteir.VisitorProtocol` extend
  `BaseVisitorProtocol`

## Backend Architecture

Each backend should live in its own package under `src/irx/builders/`. The
package path identifies the backend, while the classes inside the package use
short generic names.

For example, `src/irx/builders/llvmliteir/` exposes:

- `Builder`
- `Visitor`
- `VisitorProtocol`
- optional `VisitorCore` as a module-private implementation class

This naming convention matters for future backends. A contributor adding a new
backend should not need to invent unique class prefixes when the package path
already provides the context.

## `llvmliteir` Package Layout

The LLVM backend is split into first-class modules instead of one monolithic
builder:

- `../src/irx/base/visitors/`: shared visitor protocol and runtime scaffold
- `facade.py`: public backend entry points
- `core.py`: shared mutable lowering state and backend lifecycle
- `protocols.py`: typing contract used by mixins and runtime features
- `types.py`, `casting.py`, `vector.py`, `strings.py`, `runtime.py`: shared IR
  infrastructure
- `visitors/`: concern-grouped `visit(...)` overloads

Foundational modules stay at the package root because they are architectural
components, not incidental helpers.

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

## Contributor Guidelines

When extending IRx, these rules help preserve the architecture:

- Put semantic meaning and validation in `analysis/`, not in a backend.
- Let codegen consume normalized semantic information instead of re-deriving it.
- Keep shared visitor dispatch defaults in `src/irx/base/visitors/` so semantic
  and backend visitors fail consistently for unsupported ASTx nodes.
- Add new backend-wide infrastructure at the package root, not under `helpers/`.
- Keep mutable lowering state instance-local.
- Prefer explicit code over clever abstractions.
- Use the package name, not class prefixes, to identify the backend.

## When To Add A New Backend

If IRx gains another backend, it should follow the same broad shape:

- a package under `src/irx/builders/`
- a public `Builder`
- a public `Visitor`
- a `VisitorProtocol` if mixins or runtime hooks need typed access
- an optional module-private `VisitorCore` for shared state and infrastructure

That keeps backend packages consistent for both contributors and users of the
library.

# Semantic Contract

IRx exposes a small but explicit semantic boundary between host parsing and
backend lowering. That boundary is defined in code by
`irx.analysis.get_semantic_contract()` and is enforced by the public analysis
entrypoints in `irx.analysis.api`.

## Stable Semantic Phases

IRx currently treats these phases as stable:

- `module_graph_expansion`: `analyze_modules(...)` asks the host
  `ImportResolver` for every reachable `ParsedModule`, records import edges, and
  produces a stable dependency order in `CompilationSession.load_order`.
- `top_level_predeclaration`: `analyze_modules(...)` registers top-level
  functions and structs for every reachable module before body validation.
- `top_level_import_resolution`: `analyze_modules(...)` resolves module-top-
  level imports into module-visible bindings and rejects unsupported import
  forms.
- `semantic_validation`: `analyze(...)`, `analyze_module(...)`, and
  `analyze_modules(...)` attach semantic sidecars, normalize resolved meaning,
  and raise `SemanticError` if diagnostics exist.

## Metadata Required Before Codegen

Before lowering starts, IRx guarantees that analyzed nodes may carry
`node.semantic: SemanticInfo` with these stable fields:

- `resolved_type`
- `resolved_symbol`
- `resolved_function`
- `resolved_struct`
- `resolved_module`
- `resolved_imports`
- `resolved_operator`
- `resolved_assignment`
- `semantic_flags`
- `extras`

For multi-module compilation, IRx also guarantees the following
`CompilationSession` state before lowering:

- `root`
- `modules`
- `graph`
- `load_order`
- `visible_bindings`

Lowering should consume this semantic metadata instead of re-deriving meaning
from raw syntax.

## Error Boundaries

- Semantic errors: invalid programs, unsupported semantic input, and import
  contract violations are reported as diagnostics and surfaced as
  `SemanticError` from the public analysis entrypoints.
- Lowering errors: once semantic analysis succeeds, failures during LLVM IR
  emission belong to backend lowering, not to semantic validation.
- Linking/runtime errors: native artifact compilation, linker execution, and
  runtime integration failures happen after lowering and are outside the
  semantic contract.

## What Arx May Hand to IRx

- Arx owns parsing. IRx accepts ASTx nodes and host-owned `ParsedModule` values;
  it does not parse source files or perform package discovery.
- Single-root lowering may use `analyze(...)` or `analyze_module(...)` when no
  cross-module import graph is required.
- Cross-module lowering must use `analyze_modules(root, resolver)` with a
  host-supplied `ImportResolver`.
- Imports are currently part of the stable contract only at module top level.
- Wildcard imports and import expressions are not part of the current stable
  lowering contract and are rejected semantically.

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
- `resolved_field_access`
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

## Scalar Numeric Foundation

Binary scalar numerics use one canonical promotion table:

| Operand mix           | Promoted operand type                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| `float + float`       | wider float                                                                                          |
| `float + integer`     | float widened to cover the integer width floor (`16`, `32`, or `64` bits), capped at `Float64`       |
| `signed + signed`     | wider signed integer                                                                                 |
| `unsigned + unsigned` | wider unsigned integer                                                                               |
| `signed + unsigned`   | wider signed integer when the signed operand is strictly wider; otherwise the wider unsigned integer |

Comparison operators (`<`, `>`, `<=`, `>=`, `==`, `!=`) promote their operands
with the same table and always return `Boolean` semantically and `i1` in LLVM
IR.

## Boolean And Comparison Contract

Boolean behavior is part of the stable semantic boundary:

- comparisons always return `Boolean`
- `if`, `while`, and `for-count` conditions must be `Boolean`
- `&&`, `||`, and `!` require `Boolean` operands
- implicit truthiness is forbidden for integers, floats, pointers, and other
  non-boolean values

Lowering should branch directly on the analyzed Boolean `i1` value for control
flow instead of inventing zero-comparison truthiness rules during codegen.

## Struct Contract

Structs are IRx's stable composite storage and ABI foundation.

- struct names are stable semantic symbols
- field order is exactly declaration order
- field names must be unique within a struct
- field types must resolve semantically before lowering
- field layout must not be implicitly reordered by semantics or lowering
- field access must resolve semantically before codegen and lower by stable
  field index
- nested structs by value are allowed when every referenced struct is fully
  defined
- direct by-value recursive structs are forbidden
- mutual by-value recursive structs are forbidden
- structs can be passed and returned by value within IRx-defined functions
- emitted LLVM struct types are plain data with no hidden headers, metadata,
  tags, or runtime object payloads

For now, empty structs are rejected explicitly instead of relying on backend-
specific behavior.

## Buffer/View Model

IRx defines a canonical buffer owner plus buffer view substrate for low-level
memory/container interop. This is not a user-facing scientific array API, and it
does not define broadcasting, slicing syntax, reductions, or tensor algebra.

The canonical view descriptor is a plain stable struct conceptually equivalent
to:

- `data: ptr`
- `owner: ptr | null`
- `dtype: opaque handle or stable token`
- `ndim: i32`
- `shape: ptr<i64>`
- `strides: ptr<i64>`
- `offset_bytes: i64`
- `flags: i32`

Semantic rules:

- ownership is explicit as borrowed, owned, or external-owner
- exactly one ownership flag must be present
- borrowed views do not free memory and use a null owner handle
- owned and external-owner views use non-null opaque owner handles
- descriptor copies are shallow metadata copies
- deep copy is explicit and never implicit
- retain and release go through runtime/native helpers
- statically known borrowed views are rejected for retain/release helpers
- mutability is attached to the view, not only the allocation
- readonly and writable views are mutually exclusive
- writes through statically readonly views are rejected semantically
- raw byte writes require an 8-bit integer value and are not typed element
  stores
- shape and strides describe logical indexing, not ownership
- offset support is part of the descriptor model
- null data with statically nonzero extent is rejected

Lowering uses `irx_buffer_view` as a named plain struct with stable field order:

```llvm
%"irx_buffer_view" = type {i8*, i8*, i8*, i32, i64*, i64*, i64, i32}
```

Runtime/native lifetime operations are feature-gated behind the `buffer` runtime
feature. Plain descriptors do not pull native helper symbols into a module
unless a helper is used.

Example scalar wrapper:

```python
astx.StructDefStmt(
    name="ScalarBox",
    attributes=[
        astx.VariableDeclaration(name="value", type_=astx.Int32()),
    ],
)
```

Example nested record:

```python
astx.StructDefStmt(
    name="Descriptor",
    attributes=[
        astx.VariableDeclaration(name="point", type_=astx.StructType("Point")),
        astx.VariableDeclaration(name="ready", type_=astx.Boolean()),
    ],
)
```

### Canonical Cast Policy

Implicit promotions in variable initializers, assignments, call arguments, and
returns are intentionally narrower than explicit casts:

- same-type assignment is always allowed
- signed integers may widen to wider signed integers
- unsigned integers may widen to wider unsigned integers
- unsigned integers may widen to strictly wider signed integers
- integers may promote to floats when the target float width meets the same
  `16`/`32`/`64` floor used by the numeric-promotion table
- floats may widen to wider floats
- implicit sign-changing integer casts to unsigned targets are rejected
- implicit narrowing casts are rejected
- implicit float-to-integer and numeric-to-boolean casts are rejected

Explicit `Cast(...)` expressions allow the full scalar conversions:

- numeric-to-numeric casts
- boolean-to-numeric casts using `0` and `1`
- numeric-to-boolean casts using `!= 0` or `!= 0.0`
- string-to-string casts
- numeric/boolean-to-string casts through runtime formatting

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

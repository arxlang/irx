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
  level imports into module-visible bindings, including semantic-only module
  namespace aliases for plain module imports, and rejects unsupported import
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
- `resolved_callable`
- `resolved_call`
- `resolved_return`
- `resolved_struct`
- `resolved_class`
- `resolved_module`
- `resolved_imports`
- `resolved_operator`
- `resolved_assignment`
- `resolved_iteration`
- `resolved_field_access`
- `resolved_module_member_access`
- `resolved_class_field_access`
- `resolved_base_class_field_access`
- `resolved_static_class_field_access`
- `resolved_method_call`
- `resolved_context_manager`
- `resolved_class_construction`
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

When lowering or build layers discover that this contract has been violated,
they now raise structured diagnostics instead of flattening the failure to a
plain Python exception string. In other words:

- semantic failures continue to aggregate in `SemanticError`
- lowering failures surface as `LoweringError`
- native runtime-artifact compilation failures surface as `NativeCompileError`
- final executable link failures surface as `LinkingError`
- runtime feature activation and symbol-resolution failures surface as
  `RuntimeFeatureError`

Each of those exception types carries one `Diagnostic` record with stable code,
phase, and best-effort source attribution when IRx can recover it.

## Diagnostic Contract

IRx now uses one shared diagnostics model across semantic analysis, lowering,
native artifact compilation, final linking, and runtime feature resolution.

Every diagnostic may include:

- phase
- message
- logical code such as `S010` or `K001`
- module attribution when known
- best-effort source location derived from `node.loc`
- note and hint lines
- wrapped cause information
- related secondary locations

Semantic analysis still aggregates multiple diagnostics in `DiagnosticBag` and
raises `SemanticError` only after the semantic pass completes. Later phases
raise one structured diagnostic exception immediately because they do not have a
bagging pass today.

### Source Locations

IRx centralizes source extraction through shared helpers:

- `get_node_source_location(node)` safely reads `node.loc` without assuming
  every AST node carries a full span
- `SourceLocation` stores line and column today and already has optional end
  fields for future span-aware parsers
- `format_source_location(...)` renders module and line/column consistently for
  semantic and non-semantic diagnostics

IRx does not invent fake spans. If an AST node does not carry location data, the
diagnostic still formats cleanly without a location prefix.

### Diagnostic Codes And Prefixes

Diagnostic codes are split into a stable logical identifier and a configurable
display prefix.

- IRx stores logical identifiers such as `S001`, `S010`, `L001`, `F001`, `R001`,
  `C001`, and `K001`
- IRx renders them through one shared `DiagnosticCodeFormatter`
- the default display prefix is `IRX-`

Current high-level families:

- `Sxxx`: semantic analysis
- `Fxxx`: public FFI contract
- `Lxxx`: lowering and codegen
- `Rxxx`: runtime feature activation and symbol resolution
- `Cxxx`: native runtime-artifact compilation
- `Kxxx`: final executable linking

Downstream compilers can override the prefix without forking IRx formatting
logic:

```python
from irx.diagnostics import set_diagnostic_code_prefix

set_diagnostic_code_prefix("ARX-")
```

After that override, the same logical code renders as `ARX-S010`, `ARX-L001`,
`ARX-R001`, and so on.

### Formatting

IRx keeps the first diagnostic line compact:

```text
module_a:12:8: error[IRX-S010]: argument 1 of call to 'puts' expects UTF8String but got Int32
```

When extra context exists, the formatter appends indented follow-up lines:

```text
module_a:4:2: error[IRX-S002]: Identifier already declared: value
  note: duplicate declarations in one scope are not allowed
  related: module_a:1:1: previous declaration is here
```

Non-semantic failures also keep their phase visible:

```text
error[IRX-K001] (link): link failed while producing 'demo'
  note: command: clang /tmp/irx_module.o -o /tmp/demo
  note: stderr: undefined reference to `sqrt`
```

## Function Signature And Calling Contract

Callable semantics are part of IRx's stable semantic boundary.

- every callable is normalized into one canonical semantic signature before
  lowering
- the canonical signature includes callable identity, ordered parameters, return
  type, calling convention class, variadic flag, extern/native status, and
  lowered symbol name
- extern signatures additionally record required runtime features and validated
  public FFI classification metadata
- parameter order is stable and exactly matches declaration order
- duplicate parameter names are rejected semantically
- parameters without defaults must not follow parameters with defaults
- unresolved parameter or return types are rejected semantically
- conflicting declarations are rejected semantically

Calling conventions are classified semantically even when current LLVM emission
is shared:

- `irx_default` for IRx-defined functions
- `c` for explicit extern/native declarations

Current declaration metadata is intentionally narrow. When present on
`FunctionPrototype`, IRx consumes:

- `is_extern`
- `calling_convention`
- `is_variadic`
- `symbol_name`
- `runtime_feature`
- `runtime_features`

## Class Inheritance Contract

Class semantics are also part of IRx's stable semantic boundary.

- every analyzed class records a deterministic C3 linearization in
  `SemanticClass.mro`
- multiple inheritance is allowed when C3 can produce a consistent order; if it
  cannot, semantic analysis raises a diagnostic before lowering
- method lookup follows MRO order for inherited candidates, but class methods
  now normalize into exact-signature overload groups in
  `SemanticClass.method_groups` and `SemanticClass.method_resolution`
- `SemanticClass.member_table` remains the single-name lookup surface for
  attributes and method names with exactly one visible overload; overloaded
  method families live only in the overload-group metadata
- same-name inherited attributes from distinct ancestors are rejected as
  ambiguous unless they collapse to one logical shared ancestor in the MRO
- same-call-signature inherited methods from sibling bases are rejected unless
  one candidate dominates unambiguously through inheritance or the subclass
  supplies an exact-signature override
- diamond inheritance is allowed semantically; `SemanticClass.shared_ancestors`
  records ancestors reached through more than one direct-base lineage so later
  layout/lowering phases can reuse that metadata instead of re-deriving it
- private members do not participate in inherited lookup; non-private inherited
  members are normalized before lowering in `SemanticClass.member_table` and
  `SemanticClass.method_groups`

## Class Layout Contract

IRx now records a deterministic low-level object layout for every analyzed class
before lowering.

- class values lower as pointers to identified object structs, not by-value
  composites
- every class object reserves two hidden header slots first: one type-descriptor
  pointer slot and one dispatch-table pointer slot
- the type-descriptor header slot is reserved in semantic layout metadata now,
  but codegen does not populate a descriptor global yet; that slot is kept
  stable for later construction/runtime work
- instance storage is flattened in one canonical ancestor-first order with
  shared ancestors stored once per logical base class
- `SemanticClass.layout.instance_fields` records stable storage indices for all
  declared instance attributes, including inherited storage that is not visible
  for lookup
- declared static attributes lower to internal module globals named in
  `SemanticClass.layout.static_fields`
- `SemanticClass.layout.visible_field_slots` and
  `SemanticClass.layout.visible_static_storage` let later lowering phases reuse
  semantic member resolution instead of recomputing layout lookups from syntax

## Class Method Contract

IRx class methods now lower as explicit functions with analysis-owned dispatch
metadata rather than as implicit runtime behavior.

- every analyzed class method records a normalized source signature plus a
  lowered callable form in `SemanticClassMember.lowered_function`
- abstract methods are marked with `SemanticClassMember.is_abstract`; they carry
  signature metadata for call validation but do not lower an executable body
- `SemanticClass.is_abstract` is true for explicitly abstract classes and for
  classes with unresolved abstract methods; `SemanticClass.abstract_methods`
  records the remaining abstract method set
- instance methods gain one hidden leading `self` parameter whose type is the
  declaring class pointer representation
- static methods keep their declared parameter list and do not receive an
  implicit receiver
- class methods support exact-signature multidispatch by method name and
  explicit argument types; overload selection does not rank implicit numeric or
  class-hierarchy conversions when more than one overload is visible
- non-private instance methods receive hierarchy-family-local dispatch slots in
  `SemanticClassMember.dispatch_slot`; valid exact-signature overrides reuse the
  inherited slot, and unrelated class families do not affect those slot numbers
- `SemanticClass.layout.dispatch_entries` and
  `SemanticClass.layout.visible_method_slots` provide the lowering-facing view
  of one class dispatch table after MRO resolution, keyed by exact signature
- `MethodCall` analysis records one `ResolvedMethodCall` with the chosen member,
  overload key, candidate set, validated argument conversions, dispatch mode,
  receiver class, and dispatch slot when indirect dispatch is required
- implicit `Derived -> Base` upcasts participate in assignment, call, and return
  compatibility, which makes overridden instance methods usable through
  base-typed receivers
- full multiple-inheritance ancestor field views are still deferred; the current
  upcast guarantee is specifically for class pointer compatibility and method
  dispatch
- conversion-ranked overload selection is intentionally deferred; callers that
  want a non-exact overload must spell the conversion explicitly with `Cast`
- `StaticMethodCall` analysis always resolves to direct calls because static
  methods never consume a hidden receiver and do not participate in instance
  dispatch
- lowering emits one internal dispatch table global per class when at least one
  visible instance method has a dispatch slot, and instance call sites load the
  callee through that table instead of re-resolving semantics from syntax
- abstract instance methods reserve dispatch slots so abstract-base-typed calls
  can dispatch through concrete subclass dispatch tables; abstract slots are
  emitted as null entries on abstract class dispatch tables

## Class Member Access Contract

IRx now distinguishes instance and class-qualified member access forms
explicitly instead of inferring them from generic field syntax.

- `FieldAccess` remains the low-level read form for `obj.attr` on structs and
  class instances
- `BaseFieldAccess` is the low-level read/write form for one explicit
  base-qualified instance attribute view on a class receiver
- `StaticFieldAccess` is the low-level read/write form for
  `ClassName.static_attr`
- `MethodCall` remains the low-level call form for `obj.method(...)` and uses
  direct or indirect dispatch from analyzed method metadata
- `BaseMethodCall` is the low-level call form for one explicit base-qualified
  instance method invocation and lowers as a direct call to the selected base
  implementation
- `StaticMethodCall` remains the low-level call form for
  `ClassName.static_method(...)` and always lowers as a direct call
- base-qualified member access is legal only when the named base appears in the
  receiver class MRO; analysis rejects unrelated class names before lowering
- static field reads resolve through
  `SemanticClass.layout.visible_static_storage` first and only fall back to
  qualified-name storage metadata for the selected inherited member when needed
- instance field reads resolve through
  `SemanticClass.layout.visible_field_slots` and the canonical flattened storage
  layout recorded during class analysis
- explicit base-qualified field reads resolve through the concrete receiver
  layout using the selected base member's qualified storage slot
- lowering consumes the resolved storage and dispatch metadata attached during
  semantic analysis and does not re-run class member lookup from syntax
- direct writes through `FieldAccess`, `BaseFieldAccess`, and
  `StaticFieldAccess` reuse the same analyzed layout or storage metadata as
  their read paths
- constant class members reject both assignment and unary mutation during
  semantic analysis before lowering
- static field initialization remains limited to literal/default construction in
  this phase; phase 8 adds mutability and write enforcement without changing
  initialization order
- implicit ancestor field views remain deferred; IRx now supports only explicit
  base-qualified ancestor access in this phase

## Class Access Control Contract

IRx enforces class visibility during semantic analysis instead of deferring it
to lowering or runtime behavior.

- `public` members are accessible from any context that can already resolve the
  containing class value
- `private` members are accessible only while analyzing methods of their
  declaring class
- `protected` members are accessible only within the declaring class and its
  subclasses; IRx does not add same-module or friend-style access
- private members still stay out of inherited lookup tables, but access sites
  diagnose hidden base members as inaccessible rather than pretending they do
  not exist
- explicit `BaseFieldAccess` and `BaseMethodCall` follow the same visibility
  rules as ordinary member access; qualifying a base does not bypass access
  control
- when the declaring class accesses one of its private members through a
  derived-typed receiver, analysis resolves the originating base member and
  lowering reuses the existing class-pointer upcast path

## Class Construction Contract

IRx now exposes one low-level default-construction path for classes without
introducing high-level constructor syntax yet.

- `ClassConstruct("Name")` allocates one heap object and returns the analyzed
  class pointer type for `Name`
- construction initializes object headers first, then instance fields in the
  same canonical flattened storage order recorded in
  `SemanticClass.layout.instance_fields`
- the reserved type-descriptor header slot is initialized to null for now
- the dispatch-table header slot is initialized to the class dispatch global
  when one exists, otherwise null
- instance fields use their declaration initializer when present; otherwise they
  receive the same zero/null default used for ordinary local declarations
- instance constant fields must have declaration initializers in the current
  model because dedicated constructors are not implemented yet
- static class fields initialize once at module load from literal declaration
  initializers or their zero/null default when no initializer is present
- non-literal static field initializers are rejected during semantic analysis in
  this phase so codegen never has to invent runtime initialization order

## Class ABI And Interop Contract

IRx now makes the internal class ABI explicit without treating it as a stable
foreign object ABI.

- source-level `ClassType` parameters and return types inside IRx-defined
  functions lower through the same pointer ABI as method receivers; they are not
  copied by value
- class method bodies lower to internal LLVM symbols named with
  `mangle_class_method_name(module_key, class_name, method_name, overload_key)`
  so the module, declaring class, method name, and exact overload signature all
  participate in the symbol name
- class static attributes lower to internal globals named with
  `mangle_class_static_name(module_key, class_name, member_name)`
- dispatch tables and reserved descriptor metadata remain internal globals and
  should be treated as opaque implementation details rather than a public ABI
- IRx does not promise a stable foreign ABI for general classes in this phase;
  explicit extern declarations reject `ClassType` directly and through pointer
  pointees
- ABI-oriented interop should continue to use plain structs, plain extern
  functions, typed pointers, and opaque handles at foreign boundaries

## Class Diagnostics Contract

IRx now treats class errors as stable semantic diagnostics rather than backend
failures.

- unknown base classes diagnose before MRO resolution completes
- duplicate direct bases, self-inheritance, inheritance cycles, and inconsistent
  C3 linearizations diagnose on the class definition
- duplicate member names, duplicate exact method signatures, return-type-only
  overloads, and mixed static/instance method families diagnose during member
  normalization
- inherited attribute ambiguity, inherited member conflicts, visibility
  reductions on override, and static/instance status changes across inheritance
  diagnose before lowering
- inaccessible private/protected member use, instance-vs-static access misuse,
  and unrelated explicit base qualification diagnose at semantic access sites
- constant class members reject assignment and unary mutation during semantic
  analysis before codegen
- invalid constant or static initialization rules diagnose while building the
  canonical class initialization plan
- abstract methods with bodies, concrete classes with unresolved abstract
  methods, direct calls to abstract methods, and construction of abstract
  classes diagnose during semantic analysis

## Public FFI Contract

IRx now treats explicit extern/native declarations as one public FFI layer
instead of an incidental backend escape hatch.

### What Qualifies As A Public FFI Callable

- only explicit extern declarations participate in the public FFI contract
- extern declarations must not define an IRx body
- extern declarations default to calling convention `c`
- source-level function names default to `symbol_name == name`
- `symbol_name` may override the linked/native symbol while keeping a different
  IRx-visible wrapper name
- `runtime_feature` or `runtime_features` may declare explicit native dependency
  packaging for that extern
- semantic analysis records the public/source name, linked symbol name, calling
  convention, variadic flag, extern flag, required runtime features, and public
  FFI admissibility metadata before lowering

### Public FFI Type Policy

IRx intentionally keeps the public FFI type surface narrow in this phase.

Accepted in extern signatures:

- scalar integers
- scalar floats
- `Boolean`
- `NoneType` only as a return type (`void`)
- `String` / `UTF8String` / `UTF8Char` only as pointer-based extern values
- `PointerType(T)` when `T` is itself FFI-admissible
- `PointerType()` as an opaque pointer
- `OpaqueHandleType("name")`
- `BufferOwnerType` as a named opaque handle
- ABI-compatible structs
- nested ABI-compatible structs by value
- the canonical `BufferViewType` descriptor, which is a stable plain ABI struct

Rejected in extern signatures:

- unresolved or unsized types
- `ClassType(...)` values and pointers to class pointees
- non-ABI-stable internal-only composite forms
- temporal and other IRx-only types without an explicit public FFI ABI contract
- pointers to unsupported pointee types
- arbitrary variadic IRx-defined callables
- function pointers and callbacks in this phase

### ABI-Compatible Structs For FFI

The public FFI layer accepts a validated subset of IRx structs:

- fields must resolve semantically before lowering
- declaration order is the ABI field order
- empty structs are rejected
- direct or mutual by-value recursive layouts are rejected
- every field must itself be FFI-admissible
- nested structs are allowed when every nested field remains ABI-admissible
- by-value and by-pointer passing both use the same validated layout assumptions
- lowering emits the same plain LLVM struct layout that semantic validation
  approved; no hidden headers or runtime payloads are introduced

### Pointers And Opaque Handles

- `PointerType(T)` represents a typed native pointer
- `PointerType()` represents an opaque pointer with no visible pointee layout
- `OpaqueHandleType("name")` represents a first-class named native handle whose
  layout is intentionally hidden
- opaque handles may be passed, returned, stored, and compared when comparison
  is otherwise semantically supported
- opaque handles do not support field access or indexing
- nullability is not modeled statically yet; null is currently a runtime-level
  concern rather than a typed IRx value

### Symbol Resolution And Runtime Features

- an extern with no runtime features emits only an LLVM external declaration;
  the final system toolchain/linker is expected to resolve the symbol
- an extern with `runtime_feature` / `runtime_features` still emits one semantic
  extern declaration, but it also activates the named runtime feature set for
  that compilation unit
- runtime features remain the only place where IRx packages native C sources,
  objects, static libraries, or linker flags
- duplicate extern declarations with incompatible ABI or runtime-feature meaning
  are rejected semantically
- duplicate source-level declarations or duplicate `symbol_name` aliases must be
  compatible in calling convention, variadic status, symbol name, parameter
  types, return type, and required runtime features

### Intentionally Unsupported For Now

- dynamic loading or plugin discovery
- callbacks and public function-pointer interop
- broad platform-specific ABI tuning beyond the current LLVM/data-layout model
- arbitrary variadic IRx-defined functions
- auto-coercion between incompatible pointers, structs, or opaque handles

Minimal examples:

```python
puts = astx.FunctionPrototype(
    "puts",
    args=astx.Arguments(astx.Argument("message", astx.UTF8String())),
    return_type=astx.Int32(),
)
puts.is_extern = True
puts.calling_convention = "c"
puts.symbol_name = "puts"
```

```python
sqrt = astx.FunctionPrototype(
    "sqrt",
    args=astx.Arguments(astx.Argument("value", astx.Float64())),
    return_type=astx.Float64(),
)
sqrt.is_extern = True
sqrt.calling_convention = "c"
sqrt.symbol_name = "sqrt"
sqrt.runtime_feature = "libm"
```

```python
open_handle = astx.FunctionPrototype(
    "open_handle",
    args=astx.Arguments(),
    return_type=astx.OpaqueHandleType("demo_handle"),
)
open_handle.is_extern = True
open_handle.calling_convention = "c"
open_handle.symbol_name = "open_handle"
```

```python
astx.StructDefStmt(
    name="Point",
    attributes=[
        astx.VariableDeclaration(name="x", type_=astx.Float64()),
        astx.VariableDeclaration(name="y", type_=astx.Float64()),
    ],
)

take_point = astx.FunctionPrototype(
    "take_point",
    args=astx.Arguments(astx.Argument("point", astx.StructType("Point"))),
    return_type=astx.Int32(),
)
take_point.is_extern = True
take_point.calling_convention = "c"
take_point.symbol_name = "take_point"
```

## Call And Return Validation

Function calls are validated through one semantic path before lowering:

- callee resolution must produce a callable symbol with a canonical signature
- fixed-arity calls may omit only a trailing suffix covered by declared defaults
- variadic calls are limited to explicit extern/native declarations
- fixed prefix arguments use the canonical implicit-cast policy
- declared default expressions are analyzed semantically before lowering and
  lowered at the call site when callers omit those trailing arguments
- successful call analysis records resolved callable metadata, resolved argument
  types, result type, and any inserted implicit conversions
- lowering must consume that metadata instead of repairing malformed calls

Returns are also validated semantically before lowering:

- `return expr` is valid only in non-void functions
- bare `return` is valid only in void functions
- implicit return conversion follows the same canonical cast policy used for
  assignments and call arguments
- non-void functions must not fall through
- structured control flow is analyzed conservatively; missing returns on any
  reachable path are rejected

Representative examples of the current semantic style:

- `cannot assign Float64 to 'count' of type Int32`
- `argument 2 of call to 'sqrt' expects Float64 but got Int32`
- `if condition must be Boolean, got Int32`
- `extern 'take_point' is not FFI-safe: parameter 'point' field 'x' uses unsupported FFI type 'DateTime'`

Void and non-void usage is explicit:

- void calls may be used as statements
- void calls may not be used as expression values
- non-void calls may be used as expressions or discarded as statements

## Generator And Yield Contract

IRx supports an initial typed generator contract for named functions:

- a function that contains `yield` must declare `GeneratorType(T)` as its return
  type
- each yielded value must be assignable to `T`
- calling a generator function returns a first-class generator object
- generator objects are one-shot iterables and resolve to
  `IterationKind.GENERATOR`
- `ForInLoopStmt` consumes generator values through the same `ResolvedIteration`
  sidecar used by other iterables
- bare `return` and `return None` terminate a generator
- non-`None` generator return values, `yield from`, `send`, `throw`, `close`,
  async generators, and generator expressions are deferred

The current LLVM lowering supports straight-line named generator bodies with
top-level `YieldStmt` nodes. Yield suspension inside nested control flow is
diagnosed before lowering until frame and cleanup suspension are extended.

## `main` Contract

`main` is part of the stable semantic contract rather than a backend caveat:

- `main` must be `Int32 main()`
- `main` must not be variadic
- `main` must not be extern
- `main` must return deterministically along every path

IRx no longer accepts loose `void main` behavior or non-deterministic
fallthrough.

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

## Loop Contract

Loop semantics are now part of the stable lowering contract for structured
control flow:

- `break` exits the nearest enclosing loop
- `continue` targets the canonical re-entry block for the active loop form
- `WhileStmt` re-enters through its condition block
- `ForCountLoopStmt` evaluates initializer once, condition before each
  iteration, update after each fallthrough or `continue`, and stores the update
  result as the next loop-variable value
- `ForRangeLoopStmt` initializes its induction variable once before the first
  condition check, observes `end` and `step` before iteration begins, runs body
  mutations before the step block, and does not expose the loop variable after
  the loop
- loop misuse such as `break` or `continue` outside a loop is rejected
  semantically before lowering and still surfaces as a structured lowering
  diagnostic in direct backend use

## Context Manager Contract

`WithStmt(manager, body, target=None)` models a Python-like context-manager
statement over IRx class values:

- the manager expression is evaluated exactly once
- the manager must resolve to a class value
- semantic analysis resolves zero-argument instance methods named `__enter__`
  and `__exit__`
- `__enter__` runs before the body
- `__exit__` runs on normal fallthrough and before structured `return`, `break`,
  or `continue` transfers leave the body
- an optional target binds the `__enter__` result for the with-body scope only
- targets require `__enter__` to return a non-void value
- the initial contract does not implement Python exception propagation or
  `__exit__(exc_type, exc_value, traceback)` suppression semantics

Lowering consumes the resolved context-manager sidecar instead of rediscovering
method names in codegen.

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
- the public FFI layer accepts only the ABI-compatible subset described above
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

Stable built-in primitive dtype tokens are also available when a producer does
not need an out-of-band dtype handle:

- `1: bool`
- `2: int8`
- `3: int16`
- `4: int32`
- `5: int64`
- `6: uint8`
- `7: uint16`
- `8: uint32`
- `9: uint64`
- `10: float32`
- `11: float64`

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
- `IRX_BUFFER_FLAG_VALIDITY_BITMAP` may advertise producer-side validity
  metadata, but generic buffer operations remain null-agnostic

Lowering uses `irx_buffer_view` as a named plain struct with stable field order:

```llvm
%"irx_buffer_view" = type {i8*, i8*, i8*, i32, i64*, i64*, i64, i32}
```

Runtime/native lifetime operations are feature-gated behind the `buffer` runtime
feature. Plain descriptors do not pull native helper symbols into a module
unless a helper is used.

## Array Runtime Interop Contract

IRx exposes arrays through a builtin runtime surface backed by Arrow. The
high-level abstraction is array-oriented; the low-level FFI ABI remains
Arrow-specific. It is not a first-class language container model.

IRx also now exposes an initial internal NDArray layer on top of that runtime:

- ndarray storage still comes from Arrow-backed array handles
- ndarray layout still lowers through the canonical `irx_buffer_view` descriptor
- ndarray indexing and view construction reuse explicit `shape`, `strides`, and
  `offset_bytes` metadata
- current NDArray support in this initial phase is limited to fixed-width
  numeric element types and readonly Arrow-backed storage

Stable scope in this phase:

- supported plain primitive Arrow storage types: `bool`, `int8`, `int16`,
  `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float32`, and
  `float64`
- opaque schema, array builder, and array handles under `irx_arrow_*`
- Arrow C Data import/export as the external interchange boundary
- explicit Arrow-to-`irx_buffer_view` projection for supported fixed-width
  numeric arrays

Import/export rules:

- `irx_arrow_array_import_copy(...)` copies external Arrow C Data into a new
  runtime-owned array handle
- `irx_arrow_array_import_move(...)` adopts external Arrow C Data into a new
  runtime-owned array handle and leaves the source structs moved-from on success
- `irx_arrow_array_export(...)` copies a runtime-owned array handle into an
  independent Arrow C Data pair that the caller releases separately
- schema handles use the same copy-oriented pattern through
  `irx_arrow_schema_import_copy(...)` and `irx_arrow_schema_export(...)`

Nullability rules:

- Arrow nullability is modeled on Arrow handles, not as generic `BufferViewType`
  element semantics
- `irx_arrow_array_is_nullable(...)`, `irx_arrow_array_null_count(...)`, and
  `irx_arrow_array_has_validity_bitmap(...)` are the stable Arrow-side
  inspection surface
- `irx_arrow_array_validity_bitmap(...)` exposes the physical validity bitmap
  pointer plus bit offset and length
- generic buffer indexing, stores, and raw writes remain null-agnostic

Arrow-to-buffer-view bridge rules:

- only fixed-width, byte-addressable primitive arrays are buffer-view compatible
  in this phase
- the bridge is always readonly and borrowed
- bridged views use a null owner handle; the caller must keep the Arrow array
  handle alive explicitly
- bridged views populate dtype, shape, strides, and offset for one 1-D columnar
  value buffer
- when a validity bitmap exists, the returned view sets
  `IRX_BUFFER_FLAG_VALIDITY_BITMAP`
- bool arrays are supported as Arrow handles but are not buffer-view compatible
  because their values are bit-packed

NDArray layering rules:

- fresh ndarray literals build flat Arrow arrays, then wrap them in
  external-owner buffer views so lifetime remains explicit
- ndarray view nodes are shallow metadata rewrites over the same storage
- ndarray shape/stride queries and indexed addressing reuse the same descriptor
  fields used by low-level buffer/view indexing
- dynamic-rank runtime validation, broadcasting, and source-language slicing
  syntax remain out of scope in this phase

Intentionally out of scope here:

- ArrowArrayStream, RecordBatch, and Table runtime handles
- dataframe/query semantics
- compute kernels
- nested, dictionary, temporal, decimal, and other non-primitive Arrow layouts
- implicit null-aware scalar semantics on generic buffer views

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

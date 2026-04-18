# IRx

**IRx** is a Python library that lowers
[**ARXLang ASTx**](https://astx.arxlang.org) nodes to **LLVM IR** using
[llvmlite]. It provides a visitor-based codegen pipeline and a small builder API
that can **translate** ASTs to LLVM IR text or **produce runnable executables**
via `clang`.

> Status: early but functional. Arithmetic, variables, functions, returns,
> structured control flow with canonical loop lowering, fatal assertion
> statements, and a few system-level expressions (e.g. `PrintExpr`) are
> supported.

## Features

- **ASTx → LLVM IR** via multiple-dispatch visitors
  ([`plum`](https://github.com/beartype/plum)).
- **Back end:** IR construction and object emission with [llvmlite].
- **Native build:** links with `clang` to produce an executable.
- **Optional runtime features:** native capabilities are feature-gated per
  compilation unit instead of being linked into every binary.
- **PIE-friendly objects:** emits PIC-compatible objects by default to work with
  modern PIE-default linkers.
- **Supported nodes (subset; exact ASTx class names):**

  - **Literals:** `LiteralInt16`, `LiteralInt32`, `LiteralString`
  - **Variables:** `Variable`, `VariableDeclaration`,
    `InlineVariableDeclaration`
  - **Ops:** `UnaryOp` (`++`, `--`), `BinaryOp` (`+ - * / < >`) with documented
    scalar numeric promotion and cast rules
  - **Flow:** `IfStmt`, `WhileStmt`, `ForCountLoopStmt`, `ForRangeLoopStmt`,
    `BreakStmt`, `ContinueStmt`, `system.AssertStmt`
  - **Functions:** `FunctionPrototype`, `Function`, `FunctionReturn`,
    `FunctionCall`
  - **System:** `system.PrintExpr` (string printing)
  - **Assertions:** `system.AssertStmt` (fatal assertion with machine-readable
    stderr reporting)

- **Built-ins:** `putchar`, `putchard` (emitted as IR); `puts` declaration when
  needed.
- **Optional native runtimes:** `libc` externs are routed through the runtime
  feature layer, feature-backed externs can request `libm`, the fatal assertion
  helper is linked on demand through `assertions`, and Arrow is now available as
  an optional native runtime feature.
- **Low-level classes:** pointer-based class objects with deterministic C3 MRO,
  multiple inheritance, dispatch metadata, static globals, access control, and
  explicit construction/member-access forms.

## Class Support

IRx now includes a first-class low-level class model in addition to plain
structs.

- instance and static attributes
- instance and static methods
- multiple inheritance with deterministic C3 linearization
- `public` / `protected` / `private`
- `static` / `constant` / `mutable`
- deterministic flattened object layout and dispatch metadata
- explicit `ClassConstruct`, `StaticFieldAccess`, `BaseFieldAccess`, and
  `BaseMethodCall` forms

For the user-facing overview and examples, see
[docs/class-model.md](docs/class-model.md). For the normative semantic and
lowering contract, see [docs/semantic-contract.md](docs/semantic-contract.md).

## Quick Start

### Requirements

- Python **3.10 – 3.13**.
- A recent **LLVM/Clang** toolchain available on `PATH`.
- A working **C standard library** (e.g., system libc) for linking calls like
  `puts`.
- Python deps: `llvmlite`, `pytest`, etc. (see `pyproject.toml` /
  `requirements.txt`).
  - Note: llvmlite has **specific Python/LLVM compatibility windows**; see its
    docs.

### Install (dev)

```bash
git clone https://github.com/arxlang/irx.git
cd irx
conda env create --file conda/dev.yaml
conda activate irx
poetry install
```

You can also install it from PyPI: `pip install pyirx`.

More details:
[https://irx.arxlang.org/installation/](https://irx.arxlang.org/installation/)

## Minimal Examples

### 1) Translate to LLVM IR (no linking)

```python
import astx
from irx.builder import Builder

builder = Builder()
module = builder.module()

# int main() { return 0; }
proto = astx.FunctionPrototype("main", astx.Arguments(), astx.Int32())
body = astx.Block()
body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
module.block.append(astx.Function(prototype=proto, body=body))

ir_text = builder.translate(module)
print(ir_text)  # LLVM IR text (str)
```

**`translate`** returns a `str` with LLVM IR. It does not produce an object file
or binary; use it for inspection, tests, or feeding another tool.

### 2) Build and run a tiny program that prints and returns `0`

```python
import astx
from irx.builder import Builder
from irx.system import PrintExpr

builder = Builder()
module = builder.module()

# int main() { print("Hello, IRx!"); return 0; }
main_proto = astx.FunctionPrototype("main", astx.Arguments(), astx.Int32())
body = astx.Block()
body.append(PrintExpr(astx.LiteralString("Hello, IRx!")))
body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
module.block.append(astx.Function(prototype=main_proto, body=body))

builder.build(module, "hello")   # emits object + links with clang
result = builder.run()           # executes ./hello → CommandResult
print(result.stdout)             # "Hello, IRx!"
```

## How It Works

### Builders & Visitors

- **`Builder` (public API)**

  - `translate(ast) -> str` — generate LLVM IR text.
  - `build(ast, output_path)` — emit object via llvmlite and link with `clang`.
  - `run()` — execute the produced binary; returns a `CommandResult` with
    `.stdout`, `.stderr`, `.returncode`, and `.success`.

- **`Visitor` (codegen)**
  - Uses `@dispatch` to visit each ASTx node type.
  - Maintains a **value stack** (`result_stack`) and **symbol table**
    (`named_values`).
  - Emits LLVM IR with `llvmlite.ir.IRBuilder`.

### Loop Lowering

Loop lowering now follows one canonical control-flow shape per loop form:

- `WhileStmt`: `while.cond -> while.body -> while.exit`
- `ForCountLoopStmt`:
  `for.count.cond -> for.count.body -> for.count.update -> for.count.exit`
- `ForRangeLoopStmt`:
  `for.range.cond -> for.range.body -> for.range.step -> for.range.exit`

Semantic invariants:

- `break` exits the nearest enclosing loop
- `continue` targets the canonical re-entry block for that loop form
- for-count initializer symbols are loop-scoped and visible only to the loop
  condition, body, and update
- for-range induction variables are loop-scoped, body-visible, mutable inside
  the body, and not visible after the loop
- for-range `start`, `end`, and `step` are observed before the first iteration;
  body mutation of the induction variable feeds the step block

### System Printing

`PrintExpr` is an `astx.Expr` holding a `LiteralString`. Its lowering:

1. Create a global constant for the string (with `\0`).
2. GEP to an `i8*` pointer.
3. Declare (or reuse) `i32 @puts(i8*)`.
4. Call `puts`.

### Assertions

`AssertStmt` is a fatal statement-level check. Its lowering:

1. Evaluate the Boolean condition.
2. Branch to a pass block when the condition is true.
3. On failure, call the native `__arx_assert_fail(...)` runtime helper.
4. Emit one machine-readable stderr line of the form
   `ARX_ASSERT_FAIL|<source>|<line>|<col>|<message>`.
5. Terminate the failing process with a non-zero exit code.

The source field uses the analyzed module display name when IRx has one;
otherwise it falls back to the module name embedded in the AST.

### Optional Runtime Features

IRx now has a generic runtime-feature system for native integrations that do not
belong as handwritten LLVM container logic.

- Features are registered by name, such as `libc`, `libm`, and `arrow`.
- Features can declare external symbols, native C sources, objects, or static
  libraries.
- The linker only compiles and links artifacts for features that are active in
  the current compilation unit.
- This is intentionally separate from any future Arx import/module layer.

Public extern declarations integrate with the same layer:

- plain externs emit an LLVM declaration and rely on the system linker
- externs with `runtime_feature` / `runtime_features` activate the named runtime
  features for that compilation unit
- known feature-owned symbols are declared through the runtime registry instead
  of a separate ad hoc native path

Arrow uses this path as its first substantial consumer:

- native runtime implemented in C under `src/irx/builder/runtime/arrow/`
- opaque `irx_arrow_*` handles for schemas, builders, and arrays
- Arrow C Data import/export boundary with explicit copy and move/adopt import
  modes
- supported primitive storage types: `int8`, `int16`, `int32`, `int64`, `uint8`,
  `uint16`, `uint32`, `uint64`, `float32`, `float64`, and `bool`
- explicit Arrow-side nullability inspection plus a readonly value-buffer bridge
  into the generic `irx_buffer_view` substrate for fixed-width numeric arrays
- Python `nanoarrow` installed by default for interop and tests
- `arx-nanoarrow-sources` installed by default for native runtime builds

The Arrow layer remains intentionally low-level: handles, lifecycle, inspection,
C Data interop, and a conservative buffer/view bridge. IRx still does not encode
dataframe semantics, query/table APIs, or direct Arrow containers in LLVM IR.

## Scalar Numeric Semantics

IRx now treats scalar numerics as a stable substrate instead of an ad hoc
"simple promotion" layer:

- one canonical promotion table for signed integers, unsigned integers, and
  floats
- one canonical implicit-promotion vs explicit-cast policy
- comparisons always resolve to `Boolean` / LLVM `i1`

The full contract lives in
[docs/semantic-contract.md](https://github.com/arxlang/irx/blob/main/docs/semantic-contract.md).

## Function Signatures And Calling Semantics

IRx now treats callable semantics as a stable semantic contract instead of
reconstructing function meaning during lowering:

- every declared or defined callable is normalized into one canonical semantic
  signature before codegen
- parameter order is semantic and preserved exactly as declared
- IRx-defined functions default to calling convention `irx_default`
- explicit extern/native declarations default to calling convention `c`
- lowering preserves the semantic calling-convention classification even when
  LLVM emission is currently the same
- calls are validated semantically before lowering: callee resolution, arity,
  narrow extern varargs policy, and canonical implicit argument conversions all
  happen in one path
- returns are validated semantically before lowering: `return expr` is only for
  non-void functions, bare `return` is only for void functions, and non-void
  fallthrough is rejected
- void calls may be used as statements, but not as values in assignments,
  returns, operators, or other expressions
- `main` is now explicit and deterministic: it must be `Int32 main()`, it may
  not be variadic or extern, and it must return along every path

The current ASTx surface remains intentionally small. When present, IRx reads
the following `FunctionPrototype` attributes during semantic predeclaration:
`is_extern`, `calling_convention`, `is_variadic`, `symbol_name`,
`runtime_feature`, and `runtime_features`.

## Public FFI Layer

IRx now exposes one explicit public FFI contract that Arx can target for native
scientific libraries:

- explicit extern declarations are the public entrypoint
- `PointerType` and `OpaqueHandleType` provide stable public pointer/handle
  types
- ABI-safe structs are validated semantically before lowering
- symbol-name overrides and runtime-feature dependencies are part of the
  canonical semantic signature
- plain externs and feature-backed externs share one lowering path and one
  link/runtime story

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

See `docs/semantic-contract.md` for the exact admissible FFI type subset and
symbol-resolution rules.

## Testing

```bash
pytest -vv
```

Example style (simplified):

```python
def test_binary_op_basic():
    builder = Builder()
    module = builder.module()

    decl_a = astx.VariableDeclaration("a", astx.Int32(), astx.LiteralInt32(1))
    decl_b = astx.VariableDeclaration("b", astx.Int32(), astx.LiteralInt32(2))

    a, b = astx.Variable("a"), astx.Variable("b")
    expr = astx.LiteralInt32(1) + b - a * b / a

    proto = astx.FunctionPrototype("main", astx.Arguments(), astx.Int32())
    block = astx.Block()
    block.append(decl_a); block.append(decl_b)
    block.append(astx.FunctionReturn(expr))
    module.block.append(astx.Function(proto, block))

    ir_text = builder.translate(module)
    assert "add" in ir_text
```

## Troubleshooting

### macOS: `ld: library 'System' not found`

- Ensure **Xcode Command Line Tools** are installed: `xcode-select --install`.
- Verify `clang --version` works.
- If needed:

  ```bash
  export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
  ```

- CI note: macOS jobs currently run on **Python 3.12** only.

### `main` rejected by semantic analysis

- IRx now requires `main` to be `Int32 main()` with a deterministic return on
  every control-flow path.
- `void main`, variadic `main`, extern `main`, and non-returning non-void `main`
  bodies are rejected before lowering.

### `plum.resolver.NotFoundLookupError`

- A visitor is missing `@dispatch` or is typed against a different class than
  the one instantiated. Ensure signatures match the exact runtime class (e.g.,
  `visit(self, node: PrintExpr)`).

### Linker or `clang` not found

- Install a recent LLVM/Clang. On Linux, use distro packages.
- On macOS, install Xcode CLT.
- On Windows, ensure LLVM’s `bin` directory is on `PATH`.

### PIE mismatch (`R_X86_64_32 ... can not be used when making a PIE object`)

- This usually means your linker is enforcing PIE while the object was compiled
  with non-PIE relocations.
- Current IRx defaults to PIC-compatible object emission, which should work with
  PIE-default linkers.
- If you are using an older ARX/IRX stack, update first.
- If you must link externally as a workaround, use:

  ```bash
  clang -no-pie file.o -o program
  ```

## Platform Notes

- **Linux & macOS:** supported and used in CI.
- **Windows:** expected to work with a proper LLVM/Clang setup; consider it
  experimental. `builder.run()` will execute `hello.exe`.

## Roadmap

- More ASTx coverage (booleans, arrays, structs, varargs/options).
- Richer stdlib bindings (I/O, math).
- Optimization toggles/passes.
- Alternative backends and/or JIT runner.
- Better diagnostics and source locations in IR.
- Expand optional [Apache Arrow](https://arrow.apache.org/) runtime support:
  streams, variable-width primitives, and higher-level interop handles.

## Contributing

Please see the [contributing guide](https://irx.arxlang.org/contributing/). Add
tests for new features and keep visitors isolated (avoid special-casing derived
nodes inside generic visitors).

## Acknowledgments

- [LLVM] and [llvmlite] for the IR infrastructure.
- **ASTx / ARXLang** for the front-end AST.
- Contributors and users experimenting with IRx.

## License

License: BSD-3-Clause. See [LICENSE](./LICENSE).

[LLVM]: https://llvm.org/
[llvmlite]: https://llvmlite.readthedocs.io/

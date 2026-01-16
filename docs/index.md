# IRx

**IRx** is a Python library that lowers
[**ARXLang ASTx**](https://astx.arxlang.org) nodes to **LLVM IR** using
[llvmlite]. It provides a visitor-based codegen pipeline and a small builder API
that can **translate** ASTs to LLVM IR text or **produce runnable executables**
via `clang`.

> Status: early but functional. Arithmetic, variables, functions, returns, basic
> control flow, and a few system-level expressions (e.g. `PrintExpr`) are
> supported.

## Features

- **ASTx → LLVM IR** via multiple-dispatch visitors
  ([`plum`](https://github.com/beartype/plum)).
- **Back end:** IR construction and object emission with [llvmlite].
- **Native build:** links with `clang` to produce an executable.
- **Supported nodes (subset; exact ASTx class names):**

  - **Literals:** `LiteralInt16`, `LiteralInt32`, `LiteralString`
  - **Variables:** `Variable`, `VariableDeclaration`,
    `InlineVariableDeclaration`
  - **Ops:** `UnaryOp` (`++`, `--`), `BinaryOp` (`+ - * / < >`) with simple type
    promotion
  - **Flow:** `IfStmt`, `ForCountLoopStmt`, `ForRangeLoopStmt`
  - **Functions:** `FunctionPrototype`, `Function`, `FunctionReturn`,
    `FunctionCall`
  - **System:** `system.PrintExpr` (string printing)

- **Built-ins:** `putchar`, `putchard` (emitted as IR); `puts` declaration when
  needed.

## Quick Start

### Requirements

- Python **3.9 – 3.13**.
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

More details: [Installation Guide](installation.md)

## Minimal Examples

### 1) Translate to LLVM IR (no linking)

```python
import astx
from irx.builders.llvmliteir import LLVMLiteIR

builder = LLVMLiteIR()
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
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

builder = LLVMLiteIR()
module = builder.module()

# int main() { print("Hello, IRx!"); return 0; }
main_proto = astx.FunctionPrototype("main", astx.Arguments(), astx.Int32())
body = astx.Block()
body.append(PrintExpr(astx.LiteralString("Hello, IRx!")))
body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
module.block.append(astx.Function(prototype=main_proto, body=body))

builder.build(module, "hello")  # emits object + links with clang
builder.run()                   # executes ./hello (or hello.exe on Windows)
```

## How It Works

### Builders & Visitors

- **`LLVMLiteIR` (public API)**

  - `translate(ast) -> str` — generate LLVM IR text.
  - `build(ast, output_path)` — emit object via llvmlite and link with `clang`.
  - `run()` — execute the produced binary.

- **`LLVMLiteIRVisitor` (codegen)**

  - Uses `@dispatch` to visit each ASTx node type.
  - Maintains a **value stack** (`result_stack`) and **symbol table**
    (`named_values`).
  - Emits LLVM IR with `llvmlite.ir.IRBuilder`.

### System Printing

`PrintExpr` is an `astx.Expr` holding a `LiteralString`. Its lowering:

1. Create a global constant for the string (with `\0`).
2. GEP to an `i8*` pointer.
3. Declare (or reuse) `i32 @puts(i8*)`.
4. Call `puts`.

## Testing

```bash
pytest -vv
```

Example style (simplified):

```python
def test_binary_op_basic():
    builder = LLVMLiteIR()
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

### Non-zero exit when function returns `void`

- Define `main` to return **`Int32`** and emit `return 0`. Falling off the end
  or returning `void` can yield an arbitrary exit code.

### `plum.resolver.NotFoundLookupError`

- A visitor is missing `@dispatch` or is typed against a different class than
  the one instantiated. Ensure signatures match the exact runtime class (e.g.,
  `visit(self, node: PrintExpr)`).

### Linker or `clang` not found

- Install a recent LLVM/Clang. On Linux, use distro packages.
- On macOS, install Xcode CLT.
- On Windows, ensure LLVM's `bin` directory is on `PATH`.

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
- Integration with [Apache Arrow](https://arrow.apache.org/).

## Contributing

Please see the [contributing guide](contributing.md). Add tests for new features
and keep visitors isolated (avoid special-casing derived nodes inside generic
visitors).

## Acknowledgments

- [LLVM] and [llvmlite] for the IR infrastructure.
- **ASTx / ARXLang** for the front-end AST.
- Contributors and users experimenting with IRx.

## License

License: BSD-3-Clause. See [LICENSE](../LICENSE).

[LLVM]: https://llvm.org/
[llvmlite]: https://llvmlite.readthedocs.io/

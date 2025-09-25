# IRx

**IRx** is a Python library that lowers
[**ARXLang ASTx**](https://astx.arxlang.org) nodes to **LLVM IR** using
[llvmlite]. It provides a clean visitor-based codegen pipeline and a small
builder API that can both **translate** ASTs to LLVM IR and **produce runnable
executables** via `clang`.

> Status: early but functional. Arithmetic, variables, functions, returns, basic
> control flow and a few system expressions (e.g. `PrintExpr`) are supported.

## Features

- **ASTx → LLVM IR** via a multiple-dispatch visitor
  ([`plum`](https://github.com/beartype/plum)).
- **Back-end: llvmlite** IR construction and object emission.
- **Native build**: links with `clang` to produce an executable.
- Supported nodes (subset): literals (`Int16`, `Int32`, `LiteralString`),
  variables & declarations, unary (`++`, `--`), binary ops (`+ - * / < >`),
  `FunctionPrototype`, `Function`, `FunctionReturn`, `Block`, `If`, `For`
  (count/range), `FunctionCall` (generic), and `system.PrintExpr`.
- Minimal built-ins: `putchar`, `putchard` (emitted as IR), and `puts`
  declaration when needed.

## Quick Start

### Requirements

- Python 3.9–3.13 (project tests commonly target these; macOS CI currently runs
  3.12).
- `clang` and a working LLVM toolchain on your PATH.
- `llvmlite`, `pytest`, and other project deps (see `pyproject.toml` /
  `requirements.txt`).

### Install (dev)

Check out https://irx.arxlang.org/installation/

## Minimal Example

Build and run a tiny program that prints and returns `0`.

```python
import astx
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

builder = LLVMLiteIR()
module = builder.module()

# int main() { print("Hello, IRx!"); return 0; }
main_proto = astx.FunctionPrototype(
    name="main", args=astx.Arguments(), return_type=astx.Int32()
)
body = astx.Block()
body.append(PrintExpr(astx.LiteralString("Hello, IRx!")))
body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
main_fn = astx.Function(prototype=main_proto, body=body)

module.block.append(main_fn)

# Compile and link with clang; produce native binary "hello"
builder.build(module, "hello")
builder.run()  # executes ./hello
```

## How It Works

### Builders & Visitors

- `LLVMLiteIR` (public API)

  - `translate(ast) -> str`: returns LLVM IR text.
  - `build(ast, output_path)`: emits object via llvmlite, links with `clang`.
  - `run()`: executes the produced binary.

- `LLVMLiteIRVisitor` (codegen)

  - Uses `@dispatch` (from `plum`) to visit each ASTx node type.
  - Maintains a small **value stack** (`result_stack`) and **symbol table**
    (`named_values`).
  - Emits LLVM IR via `llvmlite.ir.IRBuilder`.

### Types

Mapped in `VariablesLLVM.get_data_type`:

- `float`, `double`, `int8`, `int16`, `int32`, `void`.

### Selected Nodes

- **Literals**: `LiteralInt16`, `LiteralInt32`, `LiteralString`.
- **Vars**: `Identifier`, `VariableDeclaration`, `InlineVariableDeclaration`.
- **Ops**: `UnaryOp` (`++`, `--`), `BinaryOp` (`+ - * / < >`) with simple type
  promotion.
- **Flow**: `IfStmt`, `ForCountLoopStmt`, `ForRangeLoopStmt`.
- **Functions**: `FunctionPrototype`, `Function`, `FunctionReturn`,
  `FunctionCall`.
- **System**: `PrintExpr(astx.LiteralString)` lowers to a call to `puts`, which
  appends a newline at the end.

### System Printing

`PrintExpr` is an `astx.Expr` holding a `LiteralString`. The visitor:

1. Creates a global constant for the string (with `\0`).
2. GEPs to `i8*`.
3. Declares (or reuses) `i32 @puts(i8*)`.
4. Calls `puts`.

## Testing

Run the test suite:

```bash
pytest -vvv -q
```

Example style (simplified):

```python
def test_binary_op_basic():
    builder = LLVMLiteIR()
    module = builder.module()

    a = astx.VariableDeclaration("a", astx.Int32(), astx.LiteralInt32(1))
    b = astx.VariableDeclaration("b", astx.Int32(), astx.LiteralInt32(2))
    expr = (astx.LiteralInt32(1) + astx.Variable("b")
           - astx.Variable("a") * astx.Variable("b") / astx.Variable("a"))

    proto = astx.FunctionPrototype("main", astx.Arguments(), astx.Int32())
    block = astx.Block()
    block.append(a); block.append(b)
    block.append(astx.FunctionReturn(expr))

    module.block.append(astx.Function(proto, block))

    # Typically compare IR or build & run
    builder.build(module, "binop_example")
```

## Troubleshooting

### macOS: `ld: library 'System' not found`

- Ensure **Xcode Command Line Tools** are installed: `xcode-select --install`.
- For local builds, `clang --version` should work.
- In some environments, setting `SDKROOT` helps:

  ```bash
  export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
  ```

- CI note: the project currently runs macOS jobs on **Python 3.12** only.

### Exit code not zero when function returns `void`

- Always define `main` as returning **`Int32`** in ASTx and emit `return 0`.
- If you fall off the end or return `void`, the process exit code may be an
  arbitrary register value (e.g., appears as 32).

### `plum.resolver.NotFoundLookupError`

- A visitor for a node type is missing the `@dispatch` decorator, or is not
  imported.
- Ensure the specialized method signature matches the **exact** class used at
  runtime (e.g., `visit(self, node: PrintExpr)`).

### Linker or `clang` not found

- Install a recent LLVM/Clang. On Linux, use your distro packages (e.g., `llvm`,
  `clang`). On macOS, install Xcode CLT.

## Roadmap

- More ASTx coverage (booleans, arrays, structs, calls with varargs/options).
- Richer stdlib bindings (I/O, math).
- Optimization toggles and passes.
- Alternative backends and/or JIT runner integration.
- Better diagnostics and source locations in IR.
- Integration with Apache Arrow

## Contributing

Check out the [contributing guide](https://irx.arxlang.org/contributing/).

## Acknowledgments

- [LLVM] and [llvmlite] for the IR infrastructure.
- **ASTx / ARXLang** for the front-end AST.
- Contributors and users experimenting with IRx.

## License

License: BSD-3-Clause; See [LICENSE](./LICENSE) in the repository.

[LLVM]: https://llvm.org/
[llvmlite]: https://llvmlite.readthedocs.io/

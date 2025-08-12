# IRx

**IRx** is the LLVM-IR backend for [**ASTx**](https://github.com/arxlang/astx),
translating ASTx nodes into LLVM Intermediate Representation (IR) using
[**llvmlite**](https://github.com/numba/llvmlite). It is a core component of the
[ArxLang](https://github.com/arxlang) toolchain, enabling compilation from a
high-level AST structure into low-level LLVM IR, which can be optimized and
compiled into native executables.

---

## ✨ Features

- Translates **ASTx** nodes into LLVM IR.
- Supports primitive types: `float16`, `float32`, `double`, `boolean`, `int8`,
  `int16`, `int32`, `int64`, `char`, `void`.
- Implements control flow constructs: `if/else`, `while`, `for` (count-based and
  range-based loops).
- Supports arithmetic, logical, and comparison operators with automatic type
  promotion.
- Built-in function integration (e.g., `putchar`, `puts`).
- Variable declarations, assignments, and inline initialization.
- Type casting between integer, floating-point, and half-precision formats.
- `PrintExpr` support for emitting strings at runtime.
- Direct object file emission and native executable compilation via `clang`.

---

## 📦 Installation

### Stable Release

```bash
pip install pyirx
```

### From Source

```bash
git clone https://github.com/arxlang/irx
cd irx
poetry install
```

Or with Conda/Mamba:

```bash
mamba env create --file conda/dev.yaml
conda activate irx
poetry install
```

---

## 🚀 Usage

The core entry point is the `LLVMLiteIR` class, which takes an ASTx object and
produces both LLVM IR and a compiled executable.

Example:

```python
import astx
from irx.builders.llvm import LLVMLiteIR

# Build a simple ASTx (example, replace with actual ASTx creation)
node = astx.Module(nodes=[
    astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments([]),
            return_type=astx.Int32Type(),
        ),
        body=astx.Block([
            astx.FunctionReturn(astx.LiteralInt32(0))
        ])
    )
])

ir_builder = LLVMLiteIR()
ir_builder.build(node, output_file="main.out")

print("Executable built: main.out")
```

This will:

1. Translate the ASTx to LLVM IR.
2. Emit an object file.
3. Compile it to a native executable via `clang`.

---

## 🛠 Project Layout

- `src/irx/builders/llvm.py` → LLVM-IR translation logic (`LLVMLiteIRVisitor`
  and `LLVMLiteIR`).
- `src/irx/tools/typing.py` → Type checking utilities.
- `tests/` → Unit tests for IRx features.

---

## 🧩 LLVM-IR Builder Highlights

The [`LLVMLiteIRVisitor`](src/irx/builders/llvm.py) implements a **visitor
pattern** over ASTx nodes:

- **Type Handling**

  ```python
  self._llvm.FLOAT_TYPE   = ir.FloatType()
  self._llvm.FLOAT16_TYPE = ir.HalfType()
  self._llvm.INT32_TYPE   = ir.IntType(32)
  ```

  `VariablesLLVM.get_data_type()` maps ASTx type names to LLVM types.

- **Control Flow**

  - `visit(astx.IfStmt)` → Generates conditional branches + PHI node merging.
  - `visit(astx.WhileStmt)` → Loops with condition checks and branching.
  - `visit(astx.ForCountLoopStmt)` / `visit(astx.ForRangeLoopStmt)` → Iterative
    loops with variable allocation and updates.

- **Operators** Supports `+`, `-`, `*`, `/`, `<`, `<=`, `>`, `>=`, `&&`, `||`,
  `!` with correct integer/floating-point instructions.

- **Variables**

  - `InlineVariableDeclaration` and `VariableDeclaration` allocate memory in the
    function entry block.
  - `VariableAssignment` stores new values into named allocations.

- **Casting** Handles:

  - Integer <-> Integer (extend/truncate)
  - Integer <-> Float (sitofp/fptosi)
  - Float <-> Half (fptrunc/fpext)

- **Builtins** `_add_builtins()` injects `putchar` and `putchard` into the LLVM
  module.

- **Printing** `PrintExpr` generates a global constant string and calls
  `puts()`.

---

## 🧪 Development Setup

1. **Fork & Clone**

   ```bash
   git clone git@github.com:your_name_here/irx.git
   cd irx
   ```

2. **Create Environment**

   ```bash
   mamba env create --file conda/dev.yaml
   conda activate irx
   poetry install
   ```

3. **Run Tests**

   ```bash
   makim tests.linter
   makim tests.unittest
   ```

4. **Run Specific Test**

   ```bash
   pytest tests/test_binary_op.py -k mytest_func
   ```

---

## 🤝 Contributing

We welcome contributions! You can:

- Report bugs: [GitHub Issues](https://github.com/arxlang/irx/issues)
- Submit PRs for bug fixes or features
- Improve documentation

**Commit Guidelines:** We use
[semantic-release](https://semantic-release.gitbook.io/) with Conventional
Commits (`feat:`, `fix:`, `chore:`, `docs:`). Breaking changes:
`feat!: description`.

---

## 📄 License

[BSD 3-Clause License](LICENSE)

---

## 📚 References

- [LLVM Language Reference](https://llvm.org/docs/LangRef.html)
- [llvmlite Documentation](https://llvmlite.readthedocs.io/en/latest/)
- [ASTx Project](https://github.com/arxlang/astx)

---

## 🔄 Compilation Flow

```mermaid
graph LR
    A[ASTx Nodes] --> B[LLVMLiteIRVisitor]
    B --> C[LLVM IR]
    C --> D[Object File (.o)]
    D --> E[clang Linker]
    E --> F[Native Executable]
```

```

---

Do you want me to also make a **`docs/` folder with separate usage and developer guide markdowns** so the repo looks more professional? That would complement this README.
```

# AI Skill: IRx Contributor Guide

This file is the shared operating manual for AI contributors working in `irx`.
Use it to keep implementation style, review standards, and delivery quality
consistent across different agents.

## When To Use This Skill

Use this guidance for any change inside the IRx repository:

- LLVM lowering or codegen behavior
- ASTx node support in visitors/builders
- build/link/run behavior
- tests, typing, lint, and coverage work
- docs/examples updates
- CI/release-related maintenance

## Core Objectives

1. Preserve existing IR semantics unless the task explicitly changes behavior.
2. Keep lowering code, tests, and docs aligned.
3. Keep quality gates green (tests, mypy, ruff, pre-commit, coverage).
4. Make minimal, targeted edits with clear intent.

## Project Snapshot

- Package: `pyirx`
- Runtime: Python `>=3.10,<4`
- Main architecture:
  `ASTx -> LLVMLiteIRVisitor -> LLVM IR -> object -> clang link`
- Key dependencies:
  - `astx` for AST nodes
  - `llvmlite` for IR and object emission
  - `plum-dispatch` for visitor dispatch
  - `xhell` for invoking `clang`
- Docs stack: MkDocs + Material + mkdocstrings

## Repository Layout

- `src/irx/builders/base.py`: generic builder interfaces and command runner
- `src/irx/builders/llvmliteir.py`: main LLVM visitor/builder implementation
- `src/irx/system.py`: IRx-specific ASTx expression helpers (`PrintExpr`,
  `Cast`)
- `src/irx/symbol_table.py`: register/symbol table helpers
- `src/irx/tools/typing.py`: typing/typeguard helpers
- `tests/`: unit/integration tests (translate and build/run flows)
- `docs/`: project documentation and notebook tutorials
- `.makim.yaml`: local task runner definitions
- `.github/workflows/main.yaml`: CI pipeline

## Architecture And Responsibilities

### `src/irx/builders/base.py`

- Defines `BuilderVisitor` and `Builder` abstractions.
- `Builder.module()` creates ASTx modules.
- `Builder.translate()` delegates to visitor translation.
- `Builder.run()` executes generated binaries via `run_command()`.

### `src/irx/builders/llvmliteir.py`

- `VariablesLLVM`: canonical LLVM types and module/builder state.
- `LLVMLiteIRVisitor`: ASTx -> LLVM lowering via `@dispatch` methods.
- `LLVMLiteIR`: public builder API (`translate`, `build`, `run`).
- Handles:
  - literals, identifiers, assignments, unary/binary ops
  - functions/prototypes/calls/returns
  - control flow (`if`, loops)
  - system expressions (`PrintExpr`, `Cast`)

### `src/irx/system.py`

- Defines IRx expression helpers used by lowering:
  - `PrintExpr`
  - `Cast`
- These are ASTx expressions and should remain structurally serializable.

## Codegen Invariants You Must Preserve

- `result_stack` discipline:
  - push only real produced values
  - never assume a value exists after statement-only or terminating branches
- Terminator safety:
  - never emit instructions after a block terminator (`ret`, `br`, etc.)
  - when temporarily moving insertion point (entry allocas), restore the
    previous block
- If/merge behavior:
  - build PHI only when both incoming paths fall through and value types match
  - allow branch-terminating paths without forced stack pops
- `safe_pop()` behavior is optional (`None` on empty stack); callers must guard
  appropriately
- Keep generated IR parseable by LLVM (`llvm.parse_assembly`).

## Build And Runtime Contract

- `translate()` only returns LLVM IR text.
- `build()` parses IR, emits object, and links with `clang`.
- `run()` executes the compiled output file.
- Build-path tests require `clang` available on `PATH`.

## Code Style And Standards

### Formatting and static quality

- Python style:
  - 4-space indentation (`.editorconfig`)
  - keep lines shorter than 80 characters (`ruff` uses 79)
- Ruff:
  - `ruff check src tests`
  - `ruff format src tests`
- Typing:
  - `mypy` strict mode (`check_untyped_defs = true`, `strict = true`)
- Pre-commit also runs:
  - `bandit`, `vulture`, `mccabe`, and project hooks
- Design:
  - follow SOLID principles
  - prefer the Never Nest philosophy as much as possible
  - do not reuse the same variable for different types
  - avoid unnecessary or obvious comments in code

### Python docstring convention in this repo

- Docstrings are written in Douki format:
  - https://github.com/arxlang/douki/
- Keep existing docstring style (`title: ...`, optional metadata fields).
- Preserve docstrings for public-facing symbols when adding/updating code.

### Error handling

- Keep errors explicit and local to failure context.
- Avoid introducing broad catch-all behavior that hides IR-generation faults.

## Tooling And Commands

Environment setup:

```bash
mamba env create --file conda/dev.yaml
conda activate irx
poetry install
```

High-value commands:

```bash
# tests
pytest tests -q

# strict typing
mypy src tests

# lint/format
ruff check src tests
ruff format src tests

# project lint stack
makim tests.linter

# CI-like local run
makim tests.ci

# coverage-gated unit run (fails under 80%)
makim tests.unit

# docs
mkdocs build --config-file mkdocs.yaml
```

IR/build debug helpers:

```bash
# generate temporary C test binary
makim tests.build

# emit LLVM IR from C for comparison experiments
makim tests.emit-ir

# build binary from .ll
makim tests.build-from-ir --file <path-without-extension>
```

## CI Contract (What Must Stay Green)

GitHub Actions (`.github/workflows/main.yaml`) runs:

- tests on Python 3.10, 3.11, 3.12, 3.13, 3.14
- operating systems: ubuntu + windows (windows/3.13 excluded)
- linter job on ubuntu (Python 3.13) with pre-commit stack

Do not merge feature work that assumes only one interpreter/OS path.

## Testing Contract

- Prefer targeted tests near changed behavior (`tests/test_*.py`).
- For lowering/control-flow work:
  - add at least one `translate`-path assertion (IR validity/shape)
  - add `build`/runtime assertions when behavior depends on execution
- Keep/extend regressions for previously fixed edge cases (e.g.,
  terminator-sensitive branches).

## Documentation Contract

When behavior changes, update docs in the same PR:

- `README.md` usage/examples
- `docs/installation.md` / `docs/contributing.md` when setup changes
- tutorial notebooks when user-facing behavior changes

## Change Playbooks

### Adding support for a new ASTx node

1. Add/adjust a `@dispatch` visitor in `llvmliteir.py`.
2. Ensure type handling and stack behavior are explicit.
3. Add tests for both happy path and at least one invalid/edge path.
4. Validate generated IR is parseable.
5. Update README/docs if feature is user-visible.

### Changing control flow or PHI logic

1. Validate block terminator states before branching/merging.
2. Avoid unconditional pops from `result_stack`.
3. Add branch-termination regressions (both terminate, one terminates, neither).
4. Verify IR parse + runtime behavior.

### Changing build/link behavior

1. Keep `translate` and `build` responsibilities separate.
2. Preserve `clang` invocation behavior across platforms where possible.
3. Add tests that fail clearly when toolchain assumptions are not met.

## Contributor Workflow Expectations

1. Make minimal focused changes.
2. Add/update tests for behavior changes.
3. Run local checks before finalizing (`ruff`, `mypy`, targeted `pytest`).
4. Keep AGENTS/docs consistent with any workflow or architecture changes.

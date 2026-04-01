# AI Skill: IRx Contributor Guide

This file is the shared operating manual for AI contributors working in the
`irx` repository. Use it to keep implementation style, architecture decisions,
and verification habits consistent across different agents.

## Repository Identity

- Repository name: `irx`
- Canonical upstream repository: `git@github.com:arxlang/irx.git`
- Typical fork remote in this workspace: `git@github.com:xmnlab/irx.git`
- Expected git remotes in this clone:
  - `origin` -> personal or workspace fork
  - `upstream` -> `arxlang/irx`

If a task depends on repository provenance, check `git remote -v` from the repo
root instead of guessing.

## Environment And Command Execution

IRx development uses the Conda environment named `irx` plus Poetry-managed
dependencies.

Create the environment with:

```bash
mamba env create --file conda/dev.yaml
conda activate irx
poetry install
```

For AI agents, prefer running commands through Conda explicitly instead of
relying on shell activation state.

Use:

```bash
conda run -n irx <command>
```

The important rule is: run tooling inside the `irx` Conda environment, and do
not assume the current shell is already activated correctly.

Recommended command style:

```bash
conda run -n irx pytest tests/test_semantic_pipeline.py -q
conda run -n irx mypy .
conda run -n irx ruff check src tests
conda run -n irx ruff format src tests
conda run -n irx pre-commit run -a
```

Run commands from the repository root unless a task explicitly needs another
working directory.

## When To Use This Guide

Use this guidance for any change inside the IRx repository, including:

- semantic analysis
- LLVM lowering or backend behavior
- runtime feature registration
- build, link, or execution behavior
- tests, typing, lint, and coverage work
- docs and tutorial updates
- CI or release-related maintenance

## Core Objectives

1. Preserve existing language behavior unless the task explicitly changes it.
2. Keep semantic analysis, codegen, tests, and docs aligned.
3. Keep quality gates green before finalizing work.
4. Make minimal, targeted edits with clear architectural intent.

## Project Snapshot

- Python package: `pyirx`
- Python support: `>=3.10,<4`
- Core dependencies:
  - `astx`
  - `llvmlite`
  - `plum-dispatch`
  - `xhell`
- Docs stack:
  - MkDocs
  - Material for MkDocs
  - mkdocstrings
  - mkdocs-jupyter

Architecture reference:

- [`docs/architecture.md`](./docs/architecture.md)

## Current Compiler Architecture

IRx now treats semantic analysis as a first-class phase before backend lowering:

`ASTx -> semantic analysis -> resolved semantic sidecars -> backend codegen`

### Semantic analysis

The semantic layer lives in `src/irx/analysis/` and is responsible for meaning
and validity, including:

- symbol resolution
- scope tracking
- type inference and compatibility
- promotion and signedness policy
- mutability and assignment validation
- loop and return legality
- operator normalization
- semantic diagnostics

The public entry points are `irx.analysis.analyze(...)` and
`irx.analysis.analyze_module(...)`.

### LLVM backend

The LLVM backend lives in `src/irx/builders/llvmliteir/`.

Important public backend names:

- `irx.builders.llvmliteir.Builder`
- `irx.builders.llvmliteir.Visitor`
- `irx.builders.llvmliteir.VisitorProtocol`

Important rule: the package path identifies the backend. Do not reintroduce
public names like `LLVMLiteIRVisitor` or `LLVMLiteIR`. Backend class names are
intentionally short and generic.

### Dispatch boundary

Codegen keeps method-based Plum multiple dispatch as its public lowering
boundary:

- `visit(self, node: ...)`

Do not replace it with a free-function registry or add a second public lowering
API.

## Repository Layout

- `src/irx/analysis/`: semantic analysis package
- `src/irx/builders/base.py`: generic builder abstractions
- `src/irx/builders/llvmliteir/`: LLVM backend package
- `src/irx/builders/_llvmliteir_legacy.py`: internal transitional detail; avoid
  adding new behavior here unless the task specifically requires it
- `src/irx/runtime/`: runtime feature declarations and linking helpers
- `src/irx/system.py`: IRx-specific ASTx expression helpers
- `src/irx/arrow.py`: Arrow-related ASTx helpers
- `src/irx/symbol_table.py`: older symbol-table utilities still present in the
  repo
- `tests/`: unit and integration tests
- `docs/`: documentation and notebook tutorials
- `.pre-commit-config.yaml`: authoritative local hook stack
- `.github/workflows/`: CI definitions

## Architectural Rules To Preserve

- Put semantic meaning and validation in `src/irx/analysis/`, not in backend
  codegen.
- Let backends consume analyzed or normalized node information instead of
  re-deriving raw AST meaning.
- Keep foundational backend infrastructure at the backend package root instead
  of creating a generic `helpers/` folder.
- Keep mutable codegen state instance-local.
- Preserve `visit(self, node: ...)` as the public codegen dispatch boundary.
- Use package names, not class prefixes, to distinguish backends.

## Working In `llvmliteir`

The LLVM backend package is structured around:

- `facade.py`: public `Builder` and `Visitor`
- `core.py`: shared concrete visitor state and lifecycle
- `protocols.py`: typing contract for mixins and runtime features
- `types.py`, `casting.py`, `vector.py`, `strings.py`, `runtime.py`: shared
  lowering infrastructure
- `visitors/`: concern-grouped `visit(...)` overloads

When changing LLVM lowering:

- keep semantic checks out of codegen when they belong in analysis
- preserve `result_stack` discipline
- avoid emitting instructions after block terminators
- validate generated IR with LLVM parsing whenever behavior changes

## Tooling And Verification

### Preferred commands

Run these from the repo root through the `irx` Conda environment:

```bash
conda run -n irx pytest tests -q
conda run -n irx mypy .
conda run -n irx ruff check src tests
conda run -n irx ruff format src tests
conda run -n irx mkdocs build --config-file mkdocs.yaml
```

### Pre-commit

Pre-commit is an important quality gate in this repository. Before finalizing a
change, prefer running:

```bash
conda run -n irx pre-commit run -a
```

Current pre-commit hooks include:

- `trailing-whitespace`
- `end-of-file-fixer`
- `check-json`
- `check-toml`
- `check-xml`
- `debug-statements`
- `check-builtin-literals`
- `check-case-conflict`
- `check-docstring-first`
- `detect-private-key`
- `prettier`
- `ruff format`
- `ruff check`
- `douki sync`
- `mypy .`
- `shellcheck`
- `bandit`
- `vulture --min-confidence 80`
- `python -m mccabe --min 10`

Notes for agents:

- `ruff format` can rewrite files.
- `ruff check` can surface issues fixed by formatting first.
- `douki sync` validates and can normalize docstring metadata.
- `mypy .` runs repo-wide, not only on changed files.
- `prettier` covers docs and non-Python formatting concerns.

## Build And Runtime Contract

- `translate()` returns LLVM IR text.
- `build()` parses IR, emits an object file, and links with `clang`.
- `run()` executes the compiled output and returns a command result.

Build-path tests require a working `clang` on `PATH`. If `clang` is missing, say
so clearly in your final update instead of implying full verification passed.

## Testing Expectations

- Prefer targeted tests near changed behavior.
- For semantic changes, add or update analysis tests first.
- For backend changes, keep at least one LLVM-IR-level assertion when practical.
- For behavior that depends on execution, add or update build/run coverage.
- When changing public imports or documentation-facing behavior, add a small
  surface regression test.

Useful examples:

```bash
conda run -n irx pytest tests/test_semantic_pipeline.py -q
conda run -n irx pytest tests/test_vector.py -q
conda run -n irx pytest tests/test_cast.py -q
```

## Documentation Expectations

When behavior or workflow changes, update docs in the same change set.

Common places to touch:

- `README.md`
- `docs/installation.md`
- `docs/contributing.md`
- `docs/architecture.md`
- tutorial notebooks under `docs/tutorials/`

## AI-Agent Workflow Guidance

- Inspect the current file layout before assuming an older design still applies.
- Prefer `rg` and focused test runs for fast feedback.
- Do not revert unrelated user changes in a dirty worktree.
- If you touch Python files, expect `pre-commit`, `ruff`, `mypy`, and `vulture`
  to matter.
- If you change docstrings, remember that Douki validation is part of the repo's
  workflow.
- If you touch semantic behavior, verify that failures happen in analysis before
  codegen when appropriate.
- If you touch backend naming, preserve the generic `Builder` / `Visitor`
  convention inside backend packages.

## Contributor Workflow Expectations

1. Make minimal focused changes.
2. Add or update tests for behavior changes.
3. Run local checks through the Conda environment before finalizing.
4. Keep `AGENTS.md` and docs aligned with workflow or architecture changes.

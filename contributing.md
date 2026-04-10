# Contributing to IRx

Thank you for your interest in contributing!

This repository uses Poetry for dependency management and a Conda environment
for development tooling. Follow the steps below to set up a working development
environment.

## Quick start (development setup)

```bash
# You can use mamba, conda, or micromamba
mamba env create --file conda/dev.yaml
conda activate irx

# Install project dependencies
poetry install
```

- If the environment name in `conda/dev.yaml` differs, activate that name
  instead of `irx`.
- Run tests and linters via Makim tasks:

```bash
makim tests.linter
makim tests.unittest
```

- Sync the repository's Codex configuration with:

```bash
makim llm-config.codex
```

This clones `https://github.com/arxlang/llm-config` into `./.tmp/llm-config`,
copies its `.codex` directory into the repository root, and removes the
temporary clone afterward. The underlying `llm-config.setup` and
`llm-config.cleanup` tasks are available for reuse in other Makim hooks.

## Runtime type checking

- Use `irx.typecheck.typechecked` on every module-level function and every
  concrete class under `src/irx`.
- Class decorators are the default way to cover methods; do not add per-method
  decorators unless a class cannot be decorated.
- Keep `@public` or `@private` outermost and place `@typechecked` on the
  implementation boundary; for wrappers like `@lru_cache(...)`, that means
  keeping `@typechecked` closest to the original function.
- Keep class decorators ordered as `@public` or `@private`, then `@typechecked`,
  then `@dataclass(...)`.
- Run `pytest tests/test_typechecked_policy.py -q` when you touch decorator
  coverage or add an exemption.

## Full guidelines

Please see the full contributing guide for project layout, workflow, and release
details: https://irx.arxlang.org/contributing/

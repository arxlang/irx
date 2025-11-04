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

## Full guidelines

Please see the full contributing guide for project layout, workflow, and release
details: https://irx.arxlang.org/contributing/

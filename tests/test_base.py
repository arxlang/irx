"""
title: Tests for builder base helpers.
"""

from typing import Sequence

import astx
import pytest

from irx.builders.base import Builder, BuilderVisitor, run_command


class _DummyVisitor(BuilderVisitor):
    """
    title: Minimal visitor for delegation tests.
    """

    def translate(self, expr: astx.AST) -> str:
        return f"translated:{expr.__class__.__name__}"


class _DummyBuilder(Builder):
    """
    title: Concrete Builder for unit tests.
    attributes:
      translator:
        type: _DummyVisitor
    """

    def __init__(self) -> None:
        super().__init__()
        self.translator: _DummyVisitor = _DummyVisitor()

    def build(self, expr: astx.AST, output_file: str) -> None:
        self.output_file = output_file


def test_run_command_success() -> None:
    """
    title: run_command should return stdout on success.
    """
    out = run_command(["/bin/sh", "-c", "printf ok"])
    assert out == "ok"


def test_run_command_nonzero_returns_code() -> None:
    """
    title: run_command should return exit code string on failure.
    """
    out = run_command(["/bin/sh", "-c", "exit 7"])
    assert out == "7"


def test_builder_visitor_translate_not_implemented() -> None:
    """
    title: Base visitor translate should raise when not overridden.
    """
    visitor = BuilderVisitor()
    with pytest.raises(Exception, match=r"Not implemented yet\."):
        visitor.translate(astx.LiteralInt32(1))


def test_builder_translate_delegates_to_translator() -> None:
    """
    title: Builder.translate should delegate to configured translator.
    """
    builder = _DummyBuilder()
    result = builder.translate(astx.LiteralInt32(1))
    assert result == "translated:LiteralInt32"


def test_builder_run_uses_output_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    title: Builder.run should execute run_command with output path.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
    """
    seen: list[Sequence[str]] = []

    def _fake_run(command: Sequence[str]) -> str:
        seen.append(command)
        return "done"

    builder = _DummyBuilder()
    builder.output_file = "/tmp/fake-bin"
    monkeypatch.setattr("irx.builders.base.run_command", _fake_run)

    assert builder.run() == "done"
    assert seen == [["/tmp/fake-bin"]]

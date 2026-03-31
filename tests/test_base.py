"""
title: Tests for builder base helpers.
"""

from typing import Sequence

import astx
import pytest

from irx.builders.base import (
    Builder,
    BuilderVisitor,
    CommandError,
    CommandResult,
    run_command,
)

EXIT_CODE = 7


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
    title: run_command should return a CommandResult with stdout on success.
    """
    result = run_command(["/bin/sh", "-c", "printf ok"])
    assert isinstance(result, CommandResult)
    assert result.stdout == "ok"
    assert result.returncode == 0
    assert result.success is True


def test_run_command_nonzero_raises() -> None:
    """
    title: run_command should raise CommandError on non-zero exit by default.
    """
    with pytest.raises(CommandError) as exc_info:
        run_command(["/bin/sh", "-c", "exit 7"])
    assert exc_info.value.result.returncode == EXIT_CODE


def test_run_command_nonzero_no_raise() -> None:
    """
    title: >-
      run_command with raise_on_error=False should return result instead of
      raising.
    """
    result = run_command(["/bin/sh", "-c", "exit 7"], raise_on_error=False)
    assert isinstance(result, CommandResult)
    assert result.returncode == EXIT_CODE
    assert result.success is False


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
    result: str = builder.translate(astx.LiteralInt32(1))
    assert result == "translated:LiteralInt32"


def test_run_command_capture_stderr_false_preserves_stdout() -> None:
    """
    title: >-
      run_command with capture_stderr=False should still capture stdout.
    """
    result = run_command(
        ["/bin/sh", "-c", "printf ok"],
        capture_stderr=False,
    )
    assert result.stdout == "ok"
    assert result.stderr == ""


def test_run_command_missing_executable_raises() -> None:
    """
    title: >-
      run_command should raise CommandError for a missing executable
      when raise_on_error=True.
    """
    with pytest.raises(CommandError) as exc_info:
        run_command(["/no/such/binary"], raise_on_error=True)
    assert exc_info.value.result.returncode == 127


def test_run_command_missing_executable_no_raise() -> None:
    """
    title: >-
      run_command with raise_on_error=False should return a result
      for a missing executable instead of raising.
    """
    result = run_command(["/no/such/binary"], raise_on_error=False)
    assert isinstance(result, CommandResult)
    assert result.returncode == 127
    assert result.success is False
    assert result.stderr != ""


def test_builder_run_uses_output_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    title: Builder.run should execute run_command with output path.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
    """
    seen: list[Sequence[str]] = []
    fake_result = CommandResult(
        stdout="done", stderr="", returncode=0, command=["/tmp/fake-bin"]
    )

    def _fake_run(
        command: Sequence[str],
        *,
        raise_on_error: bool = True,
        debug: bool = False,
        capture_stderr: bool = True,
    ) -> CommandResult:
        seen.append(command)
        return fake_result

    builder = _DummyBuilder()
    builder.output_file = "/tmp/fake-bin"
    monkeypatch.setattr("irx.builders.base.run_command", _fake_run)

    result = builder.run()
    assert result == fake_result
    assert seen == [["/tmp/fake-bin"]]

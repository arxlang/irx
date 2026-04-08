"""
title: Tests for builder base helpers.
"""

import sys

from typing import Any, Sequence

import pytest

from irx import astx
from irx.builder.base import (
    Builder,
    BuilderVisitor,
    CommandError,
    CommandResult,
    run_command,
)

EXIT_CODE = 7
EXIT_CODE_NOT_FOUND = 127


class _DummyVisitor(BuilderVisitor):
    """
    title: Minimal visitor for delegation tests.
    """

    def translate(self, expr: astx.AST) -> str:
        """
        title: Translate.
        parameters:
          expr:
            type: astx.AST
        returns:
          type: str
        """
        return f"translated:{expr.__class__.__name__}"


class _DummyBuilder(Builder):
    """
    title: Concrete Builder for unit tests.
    attributes:
      translator:
        type: _DummyVisitor
    """

    def __init__(self) -> None:
        """
        title: Initialize _DummyBuilder.
        """
        super().__init__()
        self.translator: _DummyVisitor = _DummyVisitor()

    def build(self, expr: astx.AST, output_file: str) -> None:
        """
        title: Build.
        parameters:
          expr:
            type: astx.AST
          output_file:
            type: str
        """
        self.output_file = output_file


def test_run_command_success() -> None:
    """
    title: run_command should return a CommandResult with stdout on success.
    """
    result = run_command([sys.executable, "-c", "print('ok', end='')"])
    assert isinstance(result, CommandResult)
    assert result.stdout == "ok"
    assert result.returncode == 0
    assert result.success is True


def test_run_command_nonzero_raises() -> None:
    """
    title: run_command should raise CommandError on non-zero exit by default.
    """
    with pytest.raises(CommandError) as exc_info:
        run_command([sys.executable, "-c", "raise SystemExit(7)"])
    assert exc_info.value.result.returncode == EXIT_CODE


def test_run_command_nonzero_no_raise() -> None:
    """
    title: >-
      run_command with raise_on_error=False should return result instead of
      raising.
    """
    result = run_command(
        [sys.executable, "-c", "raise SystemExit(7)"], raise_on_error=False
    )
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


def test_builder_visitor_visit_not_implemented() -> None:
    """
    title: Shared base visit should raise for unimplemented builder nodes.
    """
    visitor = BuilderVisitor()
    with pytest.raises(
        NotImplementedError,
        match=r"BuilderVisitor\.visit\(LiteralInt32\) is not implemented",
    ):
        visitor.visit(astx.LiteralInt32(1))


def test_builder_translate_delegates_to_translator() -> None:
    """
    title: Builder.translate should delegate to configured translator.
    """
    builder = _DummyBuilder()
    result: str = builder.translate(astx.LiteralInt32(1))
    assert result == "translated:LiteralInt32"


def test_run_command_capture_stderr_false_preserves_stdout() -> None:
    """
    title: run_command with capture_stderr=False should still capture stdout.
    """
    result = run_command(
        [sys.executable, "-c", "print('ok', end='')"],
        capture_stderr=False,
    )
    assert result.stdout == "ok"
    assert result.stderr == ""


def test_run_command_missing_executable_raises() -> None:
    """
    title: >-
      run_command should raise CommandError for a missing executable when
      raise_on_error=True.
    """
    with pytest.raises(CommandError) as exc_info:
        run_command(["/no/such/binary"], raise_on_error=True)
    assert exc_info.value.result.returncode == EXIT_CODE_NOT_FOUND


def test_run_command_missing_executable_no_raise() -> None:
    """
    title: >-
      run_command with raise_on_error=False should return a result for a
      missing executable instead of raising.
    """
    result = run_command(["/no/such/binary"], raise_on_error=False)
    assert isinstance(result, CommandResult)
    assert result.returncode == EXIT_CODE_NOT_FOUND
    assert result.success is False
    assert result.stderr != ""


def test_builder_run_forwards_all_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    title: >-
      Builder.run should forward command, capture_stderr, raise_on_error, and
      debug to run_command.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
    """
    calls: list[dict[str, Any]] = []
    fake_result = CommandResult(
        stdout="done", stderr="", returncode=0, command=["/tmp/fake-bin"]
    )

    def _fake_run(
        command: Sequence[str],
        *,
        capture_stderr: bool = True,
        raise_on_error: bool = True,
        debug: bool = False,
    ) -> CommandResult:
        """
        title: Fake run.
        parameters:
          command:
            type: Sequence[str]
          capture_stderr:
            type: bool
          raise_on_error:
            type: bool
          debug:
            type: bool
        returns:
          type: CommandResult
        """
        calls.append(
            {
                "command": list(command),
                "capture_stderr": capture_stderr,
                "raise_on_error": raise_on_error,
                "debug": debug,
            }
        )
        return fake_result

    builder = _DummyBuilder()
    builder.output_file = "/tmp/fake-bin"
    monkeypatch.setattr("irx.builder.base.run_command", _fake_run)

    result = builder.run(
        capture_stderr=False, raise_on_error=False, debug=True
    )
    assert result == fake_result
    assert calls == [
        {
            "command": ["/tmp/fake-bin"],
            "capture_stderr": False,
            "raise_on_error": False,
            "debug": True,
        }
    ]

"""
title: Define the public irx API.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from plum import dispatch

from irx import astx
from irx.tools.typing import typechecked
from irx.visitors.base import BaseVisitor

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """
    title: Structured result from a shell command.
    attributes:
      stdout:
        type: str
      stderr:
        type: str
      returncode:
        type: int
      command:
        type: Sequence[str]
    """

    stdout: str
    stderr: str
    returncode: int
    command: Sequence[str]

    @property
    def success(self) -> bool:
        """
        title: Return True if the command exited with code 0.
        returns:
          type: bool
        """
        return self.returncode == 0


class CommandError(RuntimeError):
    """
    title: Raised when a shell command exits with a non-zero status.
    attributes:
      result:
        type: CommandResult
    """

    def __init__(self, result: CommandResult) -> None:
        self.result: CommandResult = result
        super().__init__(
            f"Command {list(result.command)!r} failed "
            f"(exit {result.returncode}):\n{result.stderr.strip()}"
        )


@typechecked
def run_command(
    command: Sequence[str],
    *,
    capture_stderr: bool = True,
    raise_on_error: bool = True,
    debug: bool = False,
) -> CommandResult:
    """
    title: Run a shell command and return a structured CommandResult.
    parameters:
      command:
        type: Sequence[str]
      capture_stderr:
        type: bool
        default: true
      raise_on_error:
        type: bool
        default: true
      debug:
        type: bool
        default: false
    returns:
      type: CommandResult
    raises:
      CommandError: When the command exits non-zero and raise_on_error=True.
    """
    if debug:
        logger.debug("run_command: %s", list(command))

    stderr_arg = subprocess.PIPE if capture_stderr else None

    try:
        proc = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=stderr_arg,
            text=True,
        )
    except FileNotFoundError as exc:
        result = CommandResult(
            stdout="",
            stderr=str(exc),
            returncode=127,
            command=command,
        )
        if raise_on_error:
            raise CommandError(result) from exc
        return result

    result = CommandResult(
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        returncode=proc.returncode,
        command=command,
    )

    if debug:
        logger.debug(
            "exit=%d stdout=%r stderr=%r",
            result.returncode,
            result.stdout[:200],
            result.stderr[:200],
        )

    if raise_on_error and not result.success:
        raise CommandError(result)

    return result


@typechecked
class BuilderVisitor(BaseVisitor):
    """
    title: Builder translator visitor built on the shared visitor base.
    """

    @dispatch
    def visit(self, node: astx.AST) -> None:
        super().visit(node)

    def visit_child(self, node: astx.AST) -> None:
        """
        title: Forward a child AST node through the public visit dispatcher.
        parameters:
          node:
            type: astx.AST
        """
        self.visit(node)

    def translate(self, expr: astx.AST) -> str:
        """
        title: Translate an ASTx expression to string.
        parameters:
          expr:
            type: astx.AST
        returns:
          type: str
        examples: |-
          self.visit(expr)
          return str(self.result)
        """
        raise Exception("Not implemented yet.")


@typechecked
class Builder(ABC):
    """
    title: ASTx Builder.
    attributes:
      translator:
        type: BuilderVisitor
      tmp_path:
        type: str
      output_file:
        type: str
      sh_args:
        type: dict[str, Any]
      runtime_feature_names:
        type: set[str]
    """

    translator: BuilderVisitor
    tmp_path: str
    output_file: str

    sh_args: dict[str, Any]
    runtime_feature_names: set[str]

    def __init__(self) -> None:
        """
        title: Initialize Builder object.
        """
        self.translator = BuilderVisitor()
        self.tmp_path = ""
        self.output_file = ""
        self.sh_args: dict[str, Any] = dict(
            _in=sys.stdin,
            _out=sys.stdout,
            _err=sys.stderr,
            _env=os.environ,
            # _new_session=True,
        )
        self.runtime_feature_names = set()

    def module(self) -> astx.Module:
        """
        title: Create a new ASTx Module.
        returns:
          type: astx.Module
        """
        return astx.Module()

    def translate(self, expr: astx.AST) -> str:
        """
        title: Transpile ASTx to LLVM-IR.
        parameters:
          expr:
            type: astx.AST
        returns:
          type: str
        """
        return self.translator.translate(expr)

    def activate_runtime_feature(self, feature_name: str) -> None:
        """
        title: Activate a native runtime feature for this compilation unit.
        parameters:
          feature_name:
            type: str
        """
        self.runtime_feature_names.add(feature_name)

    @abstractmethod
    def build(
        self,
        expr: astx.AST,
        output_file: str,  # noqa: F841, RUF100
    ) -> None:
        """
        title: Transpile ASTx to LLVM-IR and build an executable file.
        parameters:
          expr:
            type: astx.AST
          output_file:
            type: str
        """
        ...

    def run(
        self,
        *,
        capture_stderr: bool = True,
        raise_on_error: bool = True,
        debug: bool = False,
    ) -> CommandResult:
        """
        title: Run the generated executable.
        parameters:
          capture_stderr:
            type: bool
            default: true
          raise_on_error:
            type: bool
            default: true
          debug:
            type: bool
            default: false
        returns:
          type: CommandResult
        """
        return run_command(
            [self.output_file],
            capture_stderr=capture_stderr,
            raise_on_error=raise_on_error,
            debug=debug,
        )

"""
title: Define the public irx API.
"""

from __future__ import annotations

import os
import subprocess
import sys

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import astx

from irx.tools.typing import typechecked


@typechecked
def run_command(command: Sequence[str]) -> str:
    """
    title: Run a shell command and return its stdout as a string.
    summary: >-
      Raises CalledProcessError if the command exits with a non-zero status.
    parameters:
      command:
        type: Sequence[str]
    returns:
      type: str
    """
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = str(e.returncode)

    return output


@typechecked
class BuilderVisitor:
    """
    title: Builder translator visitor.
    """

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
        type: Dict[str, Any]
      runtime_feature_names:
        type: set[str]
    """

    translator: BuilderVisitor
    tmp_path: str
    output_file: str

    sh_args: Dict[str, Any]
    runtime_feature_names: set[str]

    def __init__(self) -> None:
        """
        title: Initialize Builder object.
        """
        self.translator = BuilderVisitor()
        self.tmp_path = ""
        self.output_file = ""
        self.sh_args: Dict[str, Any] = dict(
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

    def run(self) -> str:
        """
        title: Run the generated executable.
        returns:
          type: str
        """
        return run_command([self.output_file])

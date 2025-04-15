"""Define the public irx API."""

from __future__ import annotations

import os
import subprocess
import sys

from abc import ABC, abstractmethod
from typing import Any, Dict

import astx

from irx.tools.typing import typechecked


@typechecked
def run_command(command: list[str]) -> None:
    """Run a command in the operating system."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        # Handle the error as needed


@typechecked
class BuilderVisitor:
    """Builder translator visitor."""

    def translate(self, expr: astx.AST) -> str:
        """
        Translate an ASTx expression to string.

        Example of how it could be implemented:

            self.visit(expr)
            return str(self.result)
        """
        raise Exception("Not implemented yet.")


@typechecked
class Builder(ABC):
    """ASTx Builder."""

    translator: BuilderVisitor
    tmp_path: str
    output_file: str

    sh_args: Dict[str, Any]

    def __init__(self) -> None:
        """Initialize Builder object."""
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

    def module(self) -> astx.Module:
        """Create a new ASTx Module."""
        return astx.Module()

    def translate(self, expr: astx.AST) -> str:
        """Transpile ASTx to LLVM-IR."""
        return self.translator.translate(expr)

    @abstractmethod
    def build(
        self,
        expr: astx.AST,
        output_file: str,  # noqa: F841, RUF100
    ) -> None:
        """Transpile ASTx to LLVM-IR and build an executable file."""
        ...

    def run(self) -> None:
        """Run the generated executable."""
        run_command([self.output_file])

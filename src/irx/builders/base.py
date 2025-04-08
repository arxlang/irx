"""Define the public irx API."""

from __future__ import annotations

import os
import sys

from abc import ABC, abstractmethod
from typing import Any, Dict

import astx

from plum import dispatch


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

    @dispatch.abstract
    def visit(self, expr: astx.AST) -> None:
        """Translate an ASTx expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.FunctionCall) -> None:
        """Translate an ASTx FunctionCall expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.Function) -> None:
        """Translate an ASTx Function expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.FunctionPrototype) -> None:
        """Translate an ASTx FunctionPrototype expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.FunctionReturn) -> None:
        """Translate an ASTx FunctionReturn expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.InlineVariableDeclaration) -> None:
        """Translate an ASTx InlineVariableDeclaration expression."""
        raise Exception("InlineVariableDeclaration not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralInt16) -> None:
        """Translate an ASTx LiteralInt16 expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralInt32) -> None:
        """Translate an ASTx LiteralInt32 expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.ForCountLoopStmt) -> None:
        """Translate an ASTx ForCountLoopStmt expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.ForRangeLoopStmt) -> None:
        """Translate an ASTx ForRangeLoopStmt expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.IfStmt) -> None:
        """Translate an ASTx If expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.BinaryOp) -> None:
        """Translate an ASTx BinaryOp expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.UnaryOp) -> None:
        """Translate an ASTx UnaryOp expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.Block) -> None:
        """Translate an ASTx Block expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.Module) -> None:
        """Translate an ASTx Module expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.Variable) -> None:
        """Translate an ASTx Variable expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.VariableDeclaration) -> None:
        """Translate an ASTx VariableDeclaration expression."""
        raise Exception("Not implemented yet.")


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

    @abstractmethod
    def run(self) -> None:
        """Run the generated executable."""
        ...

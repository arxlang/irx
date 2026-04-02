"""
title: Diagnostic objects for semantic analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from public import public

from irx import astx


@public
@dataclass(frozen=True)
class Diagnostic:
    """
    title: One semantic diagnostic.
    attributes:
      message:
        type: str
      node:
        type: astx.AST | None
      code:
        type: str | None
      severity:
        type: str
    """

    message: str
    node: astx.AST | None = None
    code: str | None = None
    severity: str = "error"

    def format(self) -> str:
        """
        title: Format the diagnostic for human display.
        returns:
          type: str
        """
        location = ""
        if self.node is not None:
            loc = getattr(self.node, "loc", None)
            if loc is not None and getattr(loc, "line", -1) >= 0:
                location = f"{loc.line}:{loc.col}: "
        code = f"[{self.code}] " if self.code else ""
        return f"{location}{code}{self.message}"


@public
class DiagnosticBag:
    """
    title: Collect semantic diagnostics.
    attributes:
      diagnostics:
        type: list[Diagnostic]
    """

    def __init__(self) -> None:
        """
        title: Initialize DiagnosticBag.
        """
        self.diagnostics: list[Diagnostic] = []

    def add(
        self,
        message: str,
        *,
        node: astx.AST | None = None,
        code: str | None = None,
    ) -> None:
        """
        title: Add one error diagnostic.
        parameters:
          message:
            type: str
          node:
            type: astx.AST | None
          code:
            type: str | None
        """
        self.diagnostics.append(
            Diagnostic(message=message, node=node, code=code)
        )

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        """
        title: Extend the bag with diagnostics.
        parameters:
          diagnostics:
            type: Iterable[Diagnostic]
        """
        self.diagnostics.extend(diagnostics)

    def has_errors(self) -> bool:
        """
        title: Return True when any diagnostics exist.
        returns:
          type: bool
        """
        return bool(self.diagnostics)

    def format(self) -> str:
        """
        title: Format the whole bag.
        returns:
          type: str
        """
        return "\n".join(diag.format() for diag in self.diagnostics)

    def raise_if_errors(self) -> None:
        """
        title: Raise SemanticError when errors exist.
        """
        if self.has_errors():
            raise SemanticError(self)


@public
class SemanticError(Exception):
    """
    title: Raised when semantic analysis fails.
    attributes:
      diagnostics:
        type: DiagnosticBag
    """

    diagnostics: DiagnosticBag

    def __init__(self, diagnostics: DiagnosticBag) -> None:
        """
        title: Initialize SemanticError.
        parameters:
          diagnostics:
            type: DiagnosticBag
        """
        self.diagnostics = diagnostics
        super().__init__(diagnostics.format())

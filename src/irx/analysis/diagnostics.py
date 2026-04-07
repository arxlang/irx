"""
title: Diagnostic objects for semantic analysis.
summary: >-
  Provide the error containers and aggregation helpers used throughout semantic
  analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from public import public

from irx import astx
from irx.analysis.module_interfaces import ModuleKey


@public
@dataclass(frozen=True)
class Diagnostic:
    """
    title: One semantic diagnostic.
    summary: >-
      Represent one analysis error or warning together with its source node and
      module attribution.
    attributes:
      message:
        type: str
      node:
        type: astx.AST | None
      code:
        type: str | None
      severity:
        type: str
      module_key:
        type: ModuleKey | None
    """

    message: str
    node: astx.AST | None = None
    code: str | None = None
    severity: str = "error"
    module_key: ModuleKey | None = None

    def format(self) -> str:
        """
        title: Format the diagnostic for human display.
        summary: >-
          Render one diagnostic with module and source-location prefixes when
          they are available.
        returns:
          type: str
        """
        location = ""
        module_prefix = ""
        if self.node is not None:
            loc = getattr(self.node, "loc", None)
            if loc is not None and getattr(loc, "line", -1) >= 0:
                location = f"{loc.line}:{loc.col}: "
        if self.module_key is not None:
            module_prefix = f"{self.module_key}: "
        code = f"[{self.code}] " if self.code else ""
        return f"{module_prefix}{location}{code}{self.message}"


@public
class DiagnosticBag:
    """
    title: Collect semantic diagnostics.
    summary: >-
      Accumulate semantic diagnostics across analysis passes and raise a
      combined exception when needed.
    attributes:
      diagnostics:
        type: list[Diagnostic]
      default_module_key:
        type: ModuleKey | None
    """

    def __init__(self) -> None:
        """
        title: Initialize DiagnosticBag.
        summary: Initialize DiagnosticBag.
        """
        self.diagnostics: list[Diagnostic] = []
        self.default_module_key: ModuleKey | None = None

    def add(
        self,
        message: str,
        *,
        node: astx.AST | None = None,
        code: str | None = None,
        module_key: ModuleKey | None = None,
    ) -> None:
        """
        title: Add one error diagnostic.
        summary: >-
          Append one diagnostic, defaulting its module attribution to the
          currently-active module.
        parameters:
          message:
            type: str
          node:
            type: astx.AST | None
          code:
            type: str | None
          module_key:
            type: ModuleKey | None
        """
        self.diagnostics.append(
            Diagnostic(
                message=message,
                node=node,
                code=code,
                module_key=module_key or self.default_module_key,
            )
        )

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        """
        title: Extend the bag with diagnostics.
        summary: >-
          Append diagnostics from another iterable without changing their
          existing metadata.
        parameters:
          diagnostics:
            type: Iterable[Diagnostic]
        """
        self.diagnostics.extend(diagnostics)

    def has_errors(self) -> bool:
        """
        title: Return True when any diagnostics exist.
        summary: >-
          Report whether analysis has accumulated any diagnostics at all.
        returns:
          type: bool
        """
        return bool(self.diagnostics)

    def format(self) -> str:
        """
        title: Format the whole bag.
        summary: Join all diagnostics into a multi-line human-readable message.
        returns:
          type: str
        """
        return "\n".join(diag.format() for diag in self.diagnostics)

    def raise_if_errors(self) -> None:
        """
        title: Raise SemanticError when errors exist.
        summary: >-
          Stop analysis immediately once at least one diagnostic has been
          recorded.
        """
        if self.has_errors():
            raise SemanticError(self)


@public
class SemanticError(Exception):
    """
    title: Raised when semantic analysis fails.
    summary: >-
      Wrap a diagnostic bag so callers can surface all semantic failures from
      one analysis attempt.
    attributes:
      diagnostics:
        type: DiagnosticBag
    """

    diagnostics: DiagnosticBag

    def __init__(self, diagnostics: DiagnosticBag) -> None:
        """
        title: Initialize SemanticError.
        summary: Initialize SemanticError.
        parameters:
          diagnostics:
            type: DiagnosticBag
        """
        self.diagnostics = diagnostics
        super().__init__(diagnostics.format())

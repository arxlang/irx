"""
title: Host-facing module interfaces for multi-module analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from public import public

from irx import astx


@public
@dataclass(frozen=True)
class ModuleKey:
    """
    title: Stable host-provided module identity.
    attributes:
      value:
        type: str
    """

    value: str

    def __str__(self) -> str:
        """
        title: Return the printable module key.
        returns:
          type: str
        """
        return self.value


@public
@dataclass(frozen=True)
class ParsedModule:
    """
    title: One parsed module supplied by the host compiler.
    attributes:
      key:
        type: ModuleKey
      ast:
        type: astx.Module
      display_name:
        type: str | None
      origin:
        type: str | None
    """

    key: ModuleKey
    ast: astx.Module
    display_name: str | None = None
    origin: str | None = None


@public
@runtime_checkable
class ImportResolver(Protocol):
    """
    title: Host callback that resolves imports to parsed modules.
    """

    def __call__(
        self,
        requesting_module_key: ModuleKey,
        import_node: astx.ImportStmt | astx.ImportFromStmt,
        requested_specifier: str,
    ) -> ParsedModule:
        """
        title: Resolve one import request.
        parameters:
          requesting_module_key:
            type: ModuleKey
          import_node:
            type: astx.ImportStmt | astx.ImportFromStmt
          requested_specifier:
            type: str
        returns:
          type: ParsedModule
        """
        _ = requesting_module_key
        _ = import_node
        _ = requested_specifier
        raise NotImplementedError

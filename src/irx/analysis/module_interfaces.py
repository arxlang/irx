"""
title: Host-facing module interfaces for multi-module analysis.
summary: >-
  Define the parser-agnostic types that hosts pass into IRX for multi-module
  compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from public import public

from irx import astx

ModuleKey: TypeAlias = str


@public
@dataclass(frozen=True)
class ParsedModule:
    """
    title: One parsed module supplied by the host compiler.
    summary: >-
      Bundle a host-owned module key with the parsed AST and optional human-
      facing origin metadata.
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
    summary: >-
      Describe the host-owned callback IRX uses to turn import specifiers into
      already-parsed modules.
    """

    def __call__(
        self,
        requesting_module_key: ModuleKey,
        import_node: astx.ImportStmt | astx.ImportFromStmt,
        requested_specifier: str,
    ) -> ParsedModule:
        """
        title: Resolve one import request.
        summary: >-
          Return the parsed module selected by the host for one import edge.
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

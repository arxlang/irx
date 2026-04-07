"""
title: Shared semantic-analysis context.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from public import public

from irx.analysis.diagnostics import DiagnosticBag
from irx.analysis.module_interfaces import ModuleKey
from irx.analysis.resolved_nodes import SemanticFunction, SemanticStruct
from irx.analysis.scopes import ScopeStack


@public
@dataclass
class SemanticContext:
    """
    title: Shared semantic-analysis state.
    attributes:
      scopes:
        type: ScopeStack
      diagnostics:
        type: DiagnosticBag
      functions_by_symbol_id:
        type: dict[str, SemanticFunction]
      functions_by_module_and_name:
        type: dict[tuple[ModuleKey, str], SemanticFunction]
      structs_by_symbol_id:
        type: dict[str, SemanticStruct]
      structs_by_module_and_name:
        type: dict[tuple[ModuleKey, str], SemanticStruct]
      current_function:
        type: SemanticFunction | None
      current_module_key:
        type: ModuleKey | None
      loop_depth:
        type: int
      _symbol_counter:
        type: int
    """

    scopes: ScopeStack = field(default_factory=ScopeStack)
    diagnostics: DiagnosticBag = field(default_factory=DiagnosticBag)
    functions_by_symbol_id: dict[str, SemanticFunction] = field(
        default_factory=dict
    )
    functions_by_module_and_name: dict[
        tuple[ModuleKey, str], SemanticFunction
    ] = field(default_factory=dict)
    structs_by_symbol_id: dict[str, SemanticStruct] = field(
        default_factory=dict
    )
    structs_by_module_and_name: dict[tuple[ModuleKey, str], SemanticStruct] = (
        field(default_factory=dict)
    )
    current_function: SemanticFunction | None = None
    current_module_key: ModuleKey | None = None
    loop_depth: int = 0
    _symbol_counter: int = 0

    def next_symbol_id(self, prefix: str) -> str:
        """
        title: Return a fresh semantic symbol id.
        parameters:
          prefix:
            type: str
        returns:
          type: str
        """
        self._symbol_counter += 1
        return f"{prefix}:{self._symbol_counter}"

    @contextmanager
    def scope(self, kind: str) -> Iterator[None]:
        """
        title: Push/pop a scope around a block of work.
        parameters:
          kind:
            type: str
        returns:
          type: Iterator[None]
        """
        self.scopes.push(kind)
        try:
            yield
        finally:
            self.scopes.pop()

    @contextmanager
    def in_function(self, function: SemanticFunction) -> Iterator[None]:
        """
        title: Temporarily set current_function.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: Iterator[None]
        """
        previous = self.current_function
        self.current_function = function
        try:
            yield
        finally:
            self.current_function = previous

    @contextmanager
    def in_module(self, module_key: ModuleKey) -> Iterator[None]:
        """
        title: Temporarily set the current module key.
        parameters:
          module_key:
            type: ModuleKey
        returns:
          type: Iterator[None]
        """
        previous = self.current_module_key
        previous_diagnostic_key = self.diagnostics.default_module_key
        self.current_module_key = module_key
        self.diagnostics.default_module_key = module_key
        try:
            yield
        finally:
            self.current_module_key = previous
            self.diagnostics.default_module_key = previous_diagnostic_key

    @contextmanager
    def in_loop(self) -> Iterator[None]:
        """
        title: Increase loop depth for loop analysis.
        returns:
          type: Iterator[None]
        """
        self.loop_depth += 1
        try:
            yield
        finally:
            self.loop_depth -= 1

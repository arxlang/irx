"""
title: Shared semantic-analysis context.
summary: >-
  Hold the mutable semantic-analysis state that is shared across node visits
  and, in multi-module mode, across modules in one session.
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
    summary: >-
      Store scopes, diagnostics, module-aware top-level registries, and
      transient current-node context for the analyzer.
    attributes:
      scopes:
        type: ScopeStack
      diagnostics:
        type: DiagnosticBag
      functions:
        type: dict[tuple[ModuleKey, str], SemanticFunction]
      structs:
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
    functions: dict[tuple[ModuleKey, str], SemanticFunction] = field(
        default_factory=dict
    )
    structs: dict[tuple[ModuleKey, str], SemanticStruct] = field(
        default_factory=dict
    )
    current_function: SemanticFunction | None = None
    current_module_key: ModuleKey | None = None
    loop_depth: int = 0
    _symbol_counter: int = 0

    def next_symbol_id(self, prefix: str) -> str:
        """
        title: Return a fresh semantic symbol id.
        summary: >-
          Generate a unique semantic id for locals and declarations that need
          stable identity inside one analysis run.
        parameters:
          prefix:
            type: str
        returns:
          type: str
        """
        self._symbol_counter += 1
        return f"{prefix}:{self._symbol_counter}"

    def register_function(self, function: SemanticFunction) -> None:
        """
        title: Register a top-level function by module and name.
        summary: >-
          Store the canonical semantic function object for one module-qualified
          top-level name.
        parameters:
          function:
            type: SemanticFunction
        """
        self.functions[(function.module_key, function.name)] = function

    def get_function(
        self,
        module_key: ModuleKey,
        name: str,
    ) -> SemanticFunction | None:
        """
        title: Return a top-level function by module and name.
        summary: >-
          Retrieve the canonical semantic function for one module-qualified
          top-level name.
        parameters:
          module_key:
            type: ModuleKey
          name:
            type: str
        returns:
          type: SemanticFunction | None
        """
        return self.functions.get((module_key, name))

    def register_struct(self, struct: SemanticStruct) -> None:
        """
        title: Register a top-level struct by module and name.
        summary: >-
          Store the canonical semantic struct object for one module-qualified
          top-level name.
        parameters:
          struct:
            type: SemanticStruct
        """
        self.structs[(struct.module_key, struct.name)] = struct

    def get_struct(
        self,
        module_key: ModuleKey,
        name: str,
    ) -> SemanticStruct | None:
        """
        title: Return a top-level struct by module and name.
        summary: >-
          Retrieve the canonical semantic struct for one module-qualified top-
          level name.
        parameters:
          module_key:
            type: ModuleKey
          name:
            type: str
        returns:
          type: SemanticStruct | None
        """
        return self.structs.get((module_key, name))

    @contextmanager
    def scope(self, kind: str) -> Iterator[None]:
        """
        title: Push/pop a scope around a block of work.
        summary: >-
          Temporarily add a lexical scope frame while analyzing a block of
          statements.
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
        summary: >-
          Remember which function body is being analyzed so return and control-
          flow rules can consult it.
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
        summary: >-
          Switch the active module identity so registrations and diagnostics
          are attributed to the right module.
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
        summary: >-
          Mark a temporary loop region so break and continue validation can see
          that loop nesting exists.
        returns:
          type: Iterator[None]
        """
        self.loop_depth += 1
        try:
            yield
        finally:
            self.loop_depth -= 1

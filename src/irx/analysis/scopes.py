"""
title: Lexical scope helpers for semantic analysis.
summary: >-
  Define the lexical-scope stack used to resolve locals and detect
  redeclarations during analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from public import public

from irx.analysis.resolved_nodes import SemanticSymbol
from irx.typecheck import typechecked


@public
@typechecked
@dataclass
class Scope:
    """
    title: One lexical scope.
    summary: >-
      Hold the local symbol table for one lexical region such as a module or
      function body.
    attributes:
      kind:
        type: str
      symbols:
        type: dict[str, SemanticSymbol]
    """

    kind: str
    symbols: dict[str, SemanticSymbol] = field(default_factory=dict)


@public
@typechecked
class ScopeStack:
    """
    title: Stack of lexical scopes.
    summary: >-
      Manage nested lexical scopes for local declarations and name lookup.
    attributes:
      _stack:
        type: list[Scope]
    """

    def __init__(self) -> None:
        """
        title: Initialize ScopeStack.
        """
        self._stack: list[Scope] = []

    @property
    def current(self) -> Scope | None:
        """
        title: Return the current scope.
        summary: >-
          Expose the innermost active lexical scope, if one has been pushed.
        returns:
          type: Scope | None
        """
        if not self._stack:
            return None
        return self._stack[-1]

    def push(self, kind: str) -> Scope:
        """
        title: Push a scope.
        summary: Create and activate a new lexical scope of the requested kind.
        parameters:
          kind:
            type: str
        returns:
          type: Scope
        """
        scope = Scope(kind=kind)
        self._stack.append(scope)
        return scope

    def pop(self) -> Scope:
        """
        title: Pop the current scope.
        summary: >-
          Remove and return the innermost lexical scope after a block finishes
          analysis.
        returns:
          type: Scope
        """
        return self._stack.pop()

    def declare(self, symbol: SemanticSymbol) -> bool:
        """
        title: Declare a symbol in the current scope.
        summary: >-
          Add one local symbol to the current scope, reporting whether the name
          was new there.
        parameters:
          symbol:
            type: SemanticSymbol
        returns:
          type: bool
        """
        current = self.current
        if current is None:
            raise RuntimeError("No active scope")
        if symbol.name in current.symbols:
            return False
        current.symbols[symbol.name] = symbol
        return True

    def resolve(self, name: str) -> SemanticSymbol | None:
        """
        title: Resolve a symbol by name.
        summary: >-
          Search outward from the innermost scope to find the visible local
          symbol for one name.
        parameters:
          name:
            type: str
        returns:
          type: SemanticSymbol | None
        """
        for scope in reversed(self._stack):
            symbol = scope.symbols.get(name)
            if symbol is not None:
                return symbol
        return None

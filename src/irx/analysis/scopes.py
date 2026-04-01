"""
title: Lexical scope helpers for semantic analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from public import public

from irx.analysis.resolved_nodes import SemanticSymbol


@public
@dataclass
class Scope:
    """
    title: One lexical scope.
    attributes:
      kind:
        type: str
      symbols:
        type: dict[str, SemanticSymbol]
    """

    kind: str
    symbols: dict[str, SemanticSymbol] = field(default_factory=dict)


@public
class ScopeStack:
    """
    title: Stack of lexical scopes.
    attributes:
      _stack:
        type: list[Scope]
    """

    def __init__(self) -> None:
        self._stack: list[Scope] = []

    @property
    def current(self) -> Scope | None:
        """
        title: Return the current scope.
        returns:
          type: Scope | None
        """
        if not self._stack:
            return None
        return self._stack[-1]

    def push(self, kind: str) -> Scope:
        """
        title: Push a scope.
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
        returns:
          type: Scope
        """
        return self._stack.pop()

    def declare(self, symbol: SemanticSymbol) -> bool:
        """
        title: Declare a symbol in the current scope.
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

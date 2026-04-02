"""
title: Symbol Table classes.
"""

from public import public

from irx.astx.symbol_table import SymbolTable
from irx.typecheck import typechecked

__all__ = ["SymbolTable"]


@public
@typechecked
class RegisterTable:
    # each level in the stack represents a context
    stack: list[int]

    def __init__(self) -> None:
        """
        title: Initialize RegisterTable.
        """
        self.stack: list[int] = []

    def append(self) -> None:
        """
        title: Append.
        """
        self.stack.append(0)

    def increase(self, count: int = 1) -> int:
        """
        title: Increase.
        parameters:
          count:
            type: int
        returns:
          type: int
        """
        self.stack[-1] += count
        return self.stack[-1]

    @property
    def last(self) -> int:
        """
        title: Last.
        returns:
          type: int
        """
        return self.stack[-1]

    def pop(self) -> None:
        """
        title: Pop.
        """
        self.stack.pop()

    def redefine(self, count: int) -> None:
        """
        title: Redefine.
        parameters:
          count:
            type: int
        """
        self.stack[-1] = count

    def reset(self) -> None:
        """
        title: Reset.
        """
        self.stack[-1] = 0

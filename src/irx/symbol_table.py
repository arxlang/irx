"""
title: Symbol Table classes.
"""

from public import public

from irx.astx.symbol_table import SymbolTable
from irx.tools.typing import typechecked

__all__ = ["SymbolTable"]


@public
@typechecked
class RegisterTable:
    # each level in the stack represents a context
    stack: list[int]

    def __init__(self) -> None:
        self.stack: list[int] = []

    def append(self) -> None:
        self.stack.append(0)

    def increase(self, count: int = 1) -> int:
        self.stack[-1] += count
        return self.stack[-1]

    @property
    def last(self) -> int:
        return self.stack[-1]

    def pop(self) -> None:
        self.stack.pop()

    def redefine(self, count: int) -> None:
        self.stack[-1] = count

    def reset(self) -> None:
        self.stack[-1] = 0

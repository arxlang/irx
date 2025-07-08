"""Collection of system classes and functions."""

import itertools

from typing import Any

import astx


class PrintExpr(astx.Expr):
    """
    PrintExpr AST class.

    Note: it would be nice to support more arguments similar to the ones
        supported by Python (*args, sep=' ', end='', file=None, flush=False).
    """

    message: astx.LiteralUTF8String
    _counter = itertools.count()

    def __init__(self, message: astx.LiteralUTF8String) -> None:
        """Initialize the PrintExpr."""
        self.message = message
        self._name = f"print_msg_{next(PrintExpr._counter)}"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """Return the AST structure of the object."""
        key = f"FunctionCall[{self}]"
        value = self.message.get_struct(simplified)

        return self._prepare_struct(key, value, simplified)


class Cast(astx.Expr):
    """
    Cast AST node for type conversions.

    Represents a cast of `value` to a specified `target_type`.
    """

    def __init__(self, value: astx.AST, target_type: Any) -> None:
        self.value = value
        self.target_type = target_type

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """Return the structured representation of the cast expression."""
        key = f"Cast[{self.target_type}]"
        value = self.value.get_struct(simplified)
        return self._prepare_struct(key, value, simplified)

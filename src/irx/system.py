"""Collection of system classes and functions."""

import itertools

import astx


class PrintExpr(astx.Expr):
    """
    PrintExpr AST class.

    Note: it would be nice to support more arguments similar to the ones
        supported by Python (*args, sep=' ', end='', file=None, flush=False).
    """

    message: astx.LiteralUTF8String
    _counter = itertools.count()  # <- ADD THIS LINE

    def __init__(self, message: astx.LiteralUTF8String) -> None:
        """Initialize the PrintExpr."""
        self.message = message
        self._name = f"print_msg_{next(PrintExpr._counter)}"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """Return the AST structure of the object."""
        key = f"FunctionCall[{self}]"
        value = self.message.get_struct(simplified)

        return self._prepare_struct(key, value, simplified)

"""Collection of system classes and functions."""

import astx


class PrintExpr(astx.Expr):
    """
    PrintExpr AST class.

    Note: it would be nice to support more arguments similar to the ones
        supported by Python (*args, sep=' ', end='', file=None, flush=False).
    """

    message: astx.LiteralUTF8String

    def __init__(self, message: astx.LiteralUTF8String) -> None:
        """Initialize the PrintExpr."""
        self.message = message

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """Return the AST structure of the object."""
        key = f"FunctionCall[{self}]"
        value = self.message.get_struct(simplified)

        return self._prepare_struct(key, value, simplified)

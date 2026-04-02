"""
title: IRX-owned system AST nodes.
"""

import itertools

from typing import Any

import astx


class PrintExpr(astx.Expr):
    """
    title: PrintExpr AST class.
    attributes:
      message:
        type: astx.Expr
      _name:
        type: str
    notes: >-
      It would be nice to support more arguments similar to the ones supported
      by Python (*args, sep=' ', end='', file=None, flush=False).
    """

    message: astx.Expr
    _counter = itertools.count()
    _name: str = ""

    def __init__(self, message: astx.Expr) -> None:
        """
        title: Initialize the PrintExpr.
        parameters:
          message:
            type: astx.Expr
        """
        super().__init__()
        self.message = message
        self._name = f"print_msg_{next(PrintExpr._counter)}"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the AST structure of the object.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"FunctionCall[{self}]"
        value = self.message.get_struct(simplified)
        return self._prepare_struct(key, value, simplified)


class Cast(astx.Expr):
    """
    title: Cast AST node for type conversions.
    summary: Represents a cast of `value` to a specified `target_type`.
    attributes:
      value:
        type: astx.AST
      target_type:
        type: Any
    """

    value: astx.AST = astx.LiteralNone()
    target_type: Any = astx.LiteralNone()

    def __init__(self, value: astx.AST, target_type: Any) -> None:
        """
        title: Initialize Cast.
        parameters:
          value:
            type: astx.AST
          target_type:
            type: Any
        """
        super().__init__()
        self.value = value
        self.target_type = target_type

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of the cast expression.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"Cast[{self.target_type}]"
        value = self.value.get_struct(simplified)
        return self._prepare_struct(key, value, simplified)


__all__ = ["Cast", "PrintExpr"]

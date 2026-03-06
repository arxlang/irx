"""
title: Collection of system classes and functions.
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
        key = f"PrintExpr[{self}]"
        value = self.message.get_struct(simplified)

        return self._prepare_struct(key, value, simplified)


    class Cast(astx.Expr):
      value: astx.AST
      target_type: astx.Type
    
    

    def __init__(self, value: astx.AST, target_type: astx.Type) -> None:
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

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
        """
        title: Initialize the PrintExpr.
        parameters:
          message:
            type: astx.Expr
        """
        self.message: astx.Expr = message
        self._name: str = f"print_msg_{next(PrintExpr._counter)}"

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
        self.value: astx.AST = value
        self.target_type: Any = target_type

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


class ListInsertExpr(astx.Expr):
    """
    title: Insert an element into a list expression.
    attributes:
      list_expr:
        type: astx.Expr
      index:
        type: astx.Expr
      value:
        type: astx.Expr
    """

    list_expr: astx.Expr
    index: astx.Expr
    value: astx.Expr

    def __init__(
        self, list_expr: astx.Expr, index: astx.Expr, value: astx.Expr
    ) -> None:
        self.list_expr = list_expr
        self.index = index
        self.value = value

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of list insert.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = "ListInsertExpr"
        value = {
            "list": self.list_expr.get_struct(simplified),
            "index": self.index.get_struct(simplified),
            "value": self.value.get_struct(simplified),
        }
        return self._prepare_struct(key, value, simplified)


class ListRemoveExpr(astx.Expr):
    """
    title: Remove the first matching element from a list expression.
    attributes:
      list_expr:
        type: astx.Expr
      value:
        type: astx.Expr
    """

    list_expr: astx.Expr
    value: astx.Expr

    def __init__(self, list_expr: astx.Expr, value: astx.Expr) -> None:
        self.list_expr = list_expr
        self.value = value

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of list remove.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = "ListRemoveExpr"
        value = {
            "list": self.list_expr.get_struct(simplified),
            "value": self.value.get_struct(simplified),
        }
        return self._prepare_struct(key, value, simplified)


class ListSearchExpr(astx.Expr):
    """
    title: Search an element in a list expression.
    attributes:
      list_expr:
        type: astx.Expr
      value:
        type: astx.Expr
    """

    list_expr: astx.Expr
    value: astx.Expr

    def __init__(self, list_expr: astx.Expr, value: astx.Expr) -> None:
        self.list_expr = list_expr
        self.value = value

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of list search.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = "ListSearchExpr"
        value = {
            "list": self.list_expr.get_struct(simplified),
            "value": self.value.get_struct(simplified),
        }
        return self._prepare_struct(key, value, simplified)


class ListCountExpr(astx.Expr):
    """
    title: Count element occurrences in a list expression.
    attributes:
      list_expr:
        type: astx.Expr
      value:
        type: astx.Expr
    """

    list_expr: astx.Expr
    value: astx.Expr

    def __init__(self, list_expr: astx.Expr, value: astx.Expr) -> None:
        self.list_expr = list_expr
        self.value = value

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of list count.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = "ListCountExpr"
        value = {
            "list": self.list_expr.get_struct(simplified),
            "value": self.value.get_struct(simplified),
        }
        return self._prepare_struct(key, value, simplified)


class ListSliceExpr(astx.Expr):
    """
    title: Slice a list expression using start and end.
    attributes:
      list_expr:
        type: astx.Expr
      start:
        type: astx.Expr
      end:
        type: astx.Expr
    """

    list_expr: astx.Expr
    start: astx.Expr
    end: astx.Expr

    def __init__(
        self, list_expr: astx.Expr, start: astx.Expr, end: astx.Expr
    ) -> None:
        self.list_expr = list_expr
        self.start = start
        self.end = end

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the structured representation of list slice.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = "ListSliceExpr"
        value = {
            "list": self.list_expr.get_struct(simplified),
            "start": self.start.get_struct(simplified),
            "end": self.end.get_struct(simplified),
        }
        return self._prepare_struct(key, value, simplified)

"""
title: Collection of system classes and functions.
"""

import itertools

from dataclasses import dataclass
from typing import Any

import astx


class PrintExpr(astx.Expr):
    """
    title: PrintExpr AST class.
    notes: >-
      It would be nice to support more arguments similar to the ones supported
      by Python (*args, sep=' ', end='', file=None, flush=False).
    attributes:
      message:
        type: astx.Expr
      _name:
        type: str
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
        super().__init__()
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


@dataclass
class ListNew(astx.AST):
    """
    title: Create a new list
    attributes:
      element_type:
        type: astx.DataType
    """

    element_type: astx.DataType


@dataclass
class ListAppend(astx.AST):
    """
    title: Append element to list
    attributes:
      list_ptr:
        type: astx.AST
      value:
        type: astx.AST
    """

    list_ptr: astx.AST
    value: astx.AST


@dataclass
class ListInsert(astx.AST):
    """
    title: Insert element into list
    attributes:
      list_ptr:
        type: astx.AST
      index:
        type: astx.AST
      value:
        type: astx.AST
    """

    list_ptr: astx.AST
    index: astx.AST
    value: astx.AST


@dataclass
class ListRemove(astx.AST):
    """
    title: Remove element from list
    attributes:
      list_ptr:
        type: astx.AST
      value:
        type: astx.AST
    """

    list_ptr: astx.AST
    value: astx.AST


@dataclass
class ListGet(astx.AST):
    """
    title: Get element from list
    attributes:
      list_ptr:
        type: astx.AST
      index:
        type: astx.AST
    """

    list_ptr: astx.AST
    index: astx.AST


@dataclass
class ListLen(astx.AST):
    """
    title: Get list length
    attributes:
      list_ptr:
        type: astx.AST
    """

    list_ptr: astx.AST


@dataclass
class ListContains(astx.AST):
    """
    title: Check if list contains value
    attributes:
      list_ptr:
        type: astx.AST
      value:
        type: astx.AST
    """

    list_ptr: astx.AST
    value: astx.AST


@dataclass
class ListCount(astx.AST):
    """
    title: Count occurrences in list
    attributes:
      list_ptr:
        type: astx.AST
      value:
        type: astx.AST
    """

    list_ptr: astx.AST
    value: astx.AST


@dataclass
class ListSlice(astx.AST):
    """
    title: Slice a list
    attributes:
      list_ptr:
        type: astx.AST
      start:
        type: astx.AST
      end:
        type: astx.AST
    """

    list_ptr: astx.AST
    start: astx.AST
    end: astx.AST

"""
title: IRX-owned system AST nodes.
"""

from __future__ import annotations

import itertools

from typing import Any

import astx

from irx.typecheck import typechecked


@typechecked
class AssertStmt(astx.StatementType):
    """
    title: AssertStmt AST class.
    summary: >-
      Represent one fatal assertion statement with an optional failure message.
    attributes:
      condition:
        type: astx.Expr
      message:
        type: astx.Expr | None
      loc:
        type: astx.SourceLocation
    """

    condition: astx.Expr
    message: astx.Expr | None
    loc: astx.SourceLocation

    def __init__(
        self,
        condition: astx.Expr,
        message: astx.Expr | None = None,
        loc: astx.SourceLocation = astx.base.NO_SOURCE_LOCATION,
        parent: astx.ASTNodes[astx.AST] | None = None,
    ) -> None:
        """
        title: Initialize AssertStmt.
        parameters:
          condition:
            type: astx.Expr
          message:
            type: astx.Expr | None
          loc:
            type: astx.SourceLocation
          parent:
            type: astx.ASTNodes[astx.AST] | None
        """
        super().__init__(loc=loc, parent=parent)
        self.loc: astx.SourceLocation = loc
        self.condition: astx.Expr = condition
        self.message: astx.Expr | None = message

    def __str__(self) -> str:
        """
        title: Render one assertion statement as text.
        returns:
          type: str
        """
        if self.message is None:
            return f"Assert[{self.condition}]"
        return f"Assert[{self.condition}, {self.message}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Return the AST structure of the assertion statement.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"ASSERT[{id(self)}]" if simplified else "ASSERT"
        value: astx.base.DictDataTypesStruct = {
            "condition": self.condition.get_struct(simplified),
        }
        if self.message is not None:
            value["message"] = self.message.get_struct(simplified)
        return self._prepare_struct(key, value, simplified)


@typechecked
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


@typechecked
class Cast(astx.DataType):
    """
    title: Cast AST node for type conversions.
    summary: Represents a cast of `value` to a specified `target_type`.
    attributes:
      value:
        type: astx.DataType
      target_type:
        type: Any
    """

    value: astx.DataType = astx.LiteralNone()
    target_type: Any = astx.LiteralNone()

    def __init__(self, value: astx.DataType, target_type: Any) -> None:
        """
        title: Initialize Cast.
        parameters:
          value:
            type: astx.DataType
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


__all__ = ["AssertStmt", "Cast", "PrintExpr"]

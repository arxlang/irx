"""
title: Type-computation helpers for semantic analysis.
summary: >-
  Compute expression result types for operators without embedding that logic
  directly into the main semantic visitor.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.types import (
    common_numeric_type,
    is_boolean_type,
    is_numeric_type,
    is_string_type,
)
from irx.typecheck import typechecked


@typechecked
def binary_result_type(
    op_code: str,
    lhs_type: astx.DataType | None,
    rhs_type: astx.DataType | None,
) -> astx.DataType | None:
    """
    title: Compute the semantic result type of a binary operator.
    parameters:
      op_code:
        type: str
      lhs_type:
        type: astx.DataType | None
      rhs_type:
        type: astx.DataType | None
    returns:
      type: astx.DataType | None
    """
    if op_code in {"<", ">", "<=", ">=", "==", "!="}:
        return astx.Boolean()

    if op_code in {"&&", "and", "||", "or"}:
        if is_boolean_type(lhs_type) and is_boolean_type(rhs_type):
            return astx.Boolean()
        return None

    if op_code == "=":
        return lhs_type

    if (
        op_code == "+"
        and is_string_type(lhs_type)
        and is_string_type(rhs_type)
    ):
        return lhs_type

    if op_code in {"+", "-", "*", "/", "%"}:
        return common_numeric_type(lhs_type, rhs_type)

    if op_code in {"|", "&", "^"}:
        return lhs_type

    return lhs_type if lhs_type == rhs_type else None


@typechecked
def unary_result_type(
    op_code: str,
    operand_type: astx.DataType | None,
) -> astx.DataType | None:
    """
    title: Compute the semantic result type of a unary operator.
    parameters:
      op_code:
        type: str
      operand_type:
        type: astx.DataType | None
    returns:
      type: astx.DataType | None
    """
    if op_code == "!":
        if is_boolean_type(operand_type):
            return astx.Boolean()
        return None
    if op_code in {"++", "--"} and is_numeric_type(operand_type):
        return operand_type
    return operand_type

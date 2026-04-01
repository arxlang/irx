"""
title: Validation helpers for semantic analysis.
"""

from __future__ import annotations

from datetime import date, datetime, time

import astx

from irx.analysis.diagnostics import DiagnosticBag
from irx.analysis.resolved_nodes import SemanticFunction
from irx.analysis.types import is_assignable

TIME_PARTS_HOUR_MINUTE = 2
TIME_PARTS_HOUR_MINUTE_SECOND = 3


def validate_assignment(
    diagnostics: DiagnosticBag,
    *,
    target_name: str,
    target_type: astx.DataType | None,
    value_type: astx.DataType | None,
    node: astx.AST,
) -> None:
    """
    title: Validate an assignment.
    parameters:
      diagnostics:
        type: DiagnosticBag
      target_name:
        type: str
      target_type:
        type: astx.DataType | None
      value_type:
        type: astx.DataType | None
      node:
        type: astx.AST
    """
    if not is_assignable(target_type, value_type):
        diagnostics.add(
            f"Cannot assign value of type '{value_type}' to '{target_name}'",
            node=node,
        )


def validate_call(
    diagnostics: DiagnosticBag,
    *,
    function: SemanticFunction,
    arg_types: list[astx.DataType | None],
    node: astx.FunctionCall,
) -> None:
    """
    title: Validate a function call.
    parameters:
      diagnostics:
        type: DiagnosticBag
      function:
        type: SemanticFunction
      arg_types:
        type: list[astx.DataType | None]
      node:
        type: astx.FunctionCall
    """
    if len(function.args) != len(list(node.args)):
        diagnostics.add("codegen: Incorrect # arguments passed.", node=node)
        return

    for idx, (param, arg_type) in enumerate(zip(function.args, arg_types)):
        if not is_assignable(param.type_, arg_type):
            diagnostics.add(
                f"Argument {idx} for '{function.name}' has incompatible type",
                node=node,
            )


def validate_cast(
    diagnostics: DiagnosticBag,
    *,
    source_type: astx.DataType | None,
    target_type: astx.DataType | None,
    node: astx.AST,
) -> None:
    """
    title: Validate a cast expression.
    parameters:
      diagnostics:
        type: DiagnosticBag
      source_type:
        type: astx.DataType | None
      target_type:
        type: astx.DataType | None
      node:
        type: astx.AST
    """
    if source_type is None or target_type is None:
        return
    if is_assignable(target_type, source_type):
        return
    if isinstance(target_type, (astx.String, astx.UTF8String)):
        return
    diagnostics.add(
        f"Unsupported cast from {source_type} to {target_type}",
        node=node,
    )


def validate_literal_time(value: str) -> time:
    """
    title: Validate an astx time literal.
    parameters:
      value:
        type: str
    returns:
      type: time
    """
    if "." in value:
        raise ValueError("fractional seconds are not supported")
    parts = value.split(":")
    if len(parts) not in {
        TIME_PARTS_HOUR_MINUTE,
        TIME_PARTS_HOUR_MINUTE_SECOND,
    }:
        raise ValueError("invalid time format")
    parsed = [int(part) for part in parts]
    if len(parsed) == TIME_PARTS_HOUR_MINUTE:
        return time(parsed[0], parsed[1], 0)
    return time(parsed[0], parsed[1], parsed[2])


def validate_literal_timestamp(value: str) -> datetime:
    """
    title: Validate an astx timestamp literal.
    parameters:
      value:
        type: str
    returns:
      type: datetime
    """
    if "." in value:
        raise ValueError("fractional seconds are not supported")
    if "Z" in value or "+" in value[10:] or "-" in value[10:]:
        raise ValueError("timezone offsets are not supported")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def validate_literal_datetime(value: str) -> datetime:
    """
    title: Validate an astx datetime literal.
    parameters:
      value:
        type: str
    returns:
      type: datetime
    """
    return validate_literal_timestamp(value)


def validate_calendar_date(value: str) -> date:
    """
    title: Validate a date component.
    parameters:
      value:
        type: str
    returns:
      type: date
    """
    return date.fromisoformat(value)

"""
title: Validation helpers for semantic analysis.
"""

from __future__ import annotations

from datetime import date, datetime, time

import astx

from irx.analysis.diagnostics import DiagnosticBag
from irx.analysis.resolved_nodes import SemanticFunction
from irx.analysis.types import is_assignable, is_boolean_type, is_numeric_type

TIME_PARTS_HOUR_MINUTE = 2
TIME_PARTS_HOUR_MINUTE_SECOND = 3
MAX_HOUR = 23
MAX_MINUTE_SECOND = 59
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


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
    if _is_numeric_cast_type(source_type) and _is_numeric_cast_type(
        target_type
    ):
        return
    if isinstance(target_type, (astx.String, astx.UTF8String)):
        return
    diagnostics.add(
        f"Unsupported cast from {source_type} to {target_type}",
        node=node,
    )


def _is_numeric_cast_type(type_: astx.DataType | None) -> bool:
    return is_numeric_type(type_) or is_boolean_type(type_)


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
    try:
        parsed = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("invalid time format") from exc

    hour, minute = parsed[0], parsed[1]
    second = parsed[2] if len(parsed) == TIME_PARTS_HOUR_MINUTE_SECOND else 0

    if not (0 <= hour <= MAX_HOUR):
        raise ValueError("hour out of range")
    if not (0 <= minute <= MAX_MINUTE_SECOND):
        raise ValueError("minute out of range")
    if not (0 <= second <= MAX_MINUTE_SECOND):
        raise ValueError("second out of range")

    return time(hour, minute, second)


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
    stripped = value.strip()

    if "T" in stripped:
        date_part, time_part = stripped.split("T", 1)
    elif " " in stripped:
        date_part, time_part = stripped.split(" ", 1)
    else:
        raise ValueError("invalid datetime format")

    if "." in time_part:
        raise ValueError("fractional seconds are not supported")
    if time_part.endswith("Z") or "+" in time_part or "-" in time_part[2:]:
        raise ValueError("timezone offsets are not supported")

    try:
        year_str, month_str, day_str = date_part.split("-")
        year = int(year_str)
        month = int(month_str)
        day = int(day_str)
    except Exception as exc:
        raise ValueError("invalid date part") from exc

    if not (INT32_MIN <= year <= INT32_MAX):
        raise ValueError("year out of 32-bit range")

    try:
        time_parts = time_part.split(":")
        if len(time_parts) not in {
            TIME_PARTS_HOUR_MINUTE,
            TIME_PARTS_HOUR_MINUTE_SECOND,
        }:
            raise ValueError("invalid time part")
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = (
            int(time_parts[2])
            if len(time_parts) == TIME_PARTS_HOUR_MINUTE_SECOND
            else 0
        )
    except Exception as exc:
        raise ValueError("invalid time part") from exc

    if not (0 <= hour <= MAX_HOUR):
        raise ValueError("hour out of range")
    if not (0 <= minute <= MAX_MINUTE_SECOND):
        raise ValueError("minute out of range")
    if not (0 <= second <= MAX_MINUTE_SECOND):
        raise ValueError("second out of range")

    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError as exc:
        raise ValueError("invalid calendar date/time") from exc


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

"""
title: Validation helpers for semantic analysis.
summary: >-
  Collect the focused validation routines that emit diagnostics for
  assignments, calls, casts, and temporal literals.
"""

from __future__ import annotations

from datetime import date, datetime, time

from irx import astx
from irx.analysis.resolved_nodes import (
    CallableResolution,
    CallResolution,
    ImplicitConversion,
    ReturnResolution,
    SemanticFunction,
)
from irx.analysis.types import (
    bit_width,
    display_type_name,
    is_assignable,
    is_boolean_type,
    is_explicitly_castable,
    is_float_type,
    is_integer_type,
    is_none_type,
    same_type,
)
from irx.diagnostics import DiagnosticBag, DiagnosticCodes
from irx.typecheck import typechecked

TIME_PARTS_HOUR_MINUTE = 2
TIME_PARTS_HOUR_MINUTE_SECOND = 3
MAX_HOUR = 23
MAX_MINUTE_SECOND = 59
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
DEFAULT_C_INTEGER_PROMOTION_WIDTH = 32


@typechecked
def _implicit_conversion_note(
    source_type: astx.DataType | None,
    target_type: astx.DataType | None,
) -> tuple[str, ...]:
    """
    title: Describe one rejected implicit conversion when relevant.
    parameters:
      source_type:
        type: astx.DataType | None
      target_type:
        type: astx.DataType | None
    returns:
      type: tuple[str, Ellipsis]
    """
    if source_type is None or target_type is None:
        return ()
    if same_type(source_type, target_type):
        return ()
    return (
        "no implicit conversion is defined from "
        f"{display_type_name(source_type)} to "
        f"{display_type_name(target_type)} in this context",
    )


@typechecked
def _callable_resolution(function: SemanticFunction) -> CallableResolution:
    """
    title: Return the canonical callable wrapper for one semantic function.
    parameters:
      function:
        type: SemanticFunction
    returns:
      type: CallableResolution
    """
    return CallableResolution(function=function, signature=function.signature)


@typechecked
def _is_void_return_sentinel(node: astx.AST | None) -> bool:
    """
    title: Return whether one AST value represents a bare void return.
    parameters:
      node:
        type: astx.AST | None
    returns:
      type: bool
    """
    return node is None or isinstance(node, astx.LiteralNone)


@typechecked
def _is_void_call_value(node: astx.AST | None) -> bool:
    """
    title: Return whether one AST value is the result of a void call.
    parameters:
      node:
        type: astx.AST | None
    returns:
      type: bool
    """
    semantic = getattr(node, "semantic", None)
    resolved_call = getattr(semantic, "resolved_call", None)
    resolved_type = getattr(semantic, "resolved_type", None)
    return resolved_call is not None and is_none_type(resolved_type)


@typechecked
def _variadic_argument_type(
    arg_type: astx.DataType | None,
) -> astx.DataType | None:
    """
    title: Return the normalized type for one C-style variadic argument.
    parameters:
      arg_type:
        type: astx.DataType | None
    returns:
      type: astx.DataType | None
    """
    if arg_type is None or is_none_type(arg_type):
        return None
    if is_boolean_type(arg_type):
        return astx.Int32()
    if is_integer_type(arg_type) and bit_width(arg_type) < (
        DEFAULT_C_INTEGER_PROMOTION_WIDTH
    ):
        return astx.Int32()
    if isinstance(arg_type, astx.Float16 | astx.Float32):
        return astx.Float64()
    if (
        is_integer_type(arg_type)
        or is_float_type(arg_type)
        or isinstance(
            arg_type,
            (
                astx.String,
                astx.UTF8String,
                astx.UTF8Char,
            ),
        )
    ):
        return arg_type
    return None


@typechecked
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
            "cannot assign "
            f"{display_type_name(value_type)} to '{target_name}' of type "
            f"{display_type_name(target_type)}",
            node=node,
            code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            notes=_implicit_conversion_note(value_type, target_type),
        )


@typechecked
def validate_call(
    diagnostics: DiagnosticBag,
    *,
    function: SemanticFunction,
    arg_types: list[astx.DataType | None],
    node: astx.FunctionCall | astx.MethodCall | astx.StaticMethodCall,
) -> CallResolution:
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
        type: astx.FunctionCall | astx.MethodCall | astx.StaticMethodCall
    returns:
      type: CallResolution
    """
    signature = function.signature
    fixed_param_count = len(signature.parameters)
    arg_count = len(arg_types)
    if signature.is_variadic:
        if arg_count < fixed_param_count:
            diagnostics.add(
                f"call to '{function.name}' expects at least "
                f"{fixed_param_count} arguments but got {arg_count}",
                node=node,
                code=DiagnosticCodes.SEMANTIC_CALL_ARITY,
            )
    elif fixed_param_count != arg_count:
        diagnostics.add(
            f"call to '{function.name}' expects {fixed_param_count} "
            f"arguments but got {arg_count}",
            node=node,
            code=DiagnosticCodes.SEMANTIC_CALL_ARITY,
        )

    resolved_argument_types: list[astx.DataType | None] = []
    implicit_conversions: list[ImplicitConversion | None] = []
    call_args = list(node.args)

    for idx, (param, arg_type) in enumerate(
        zip(signature.parameters, arg_types)
    ):
        if not is_assignable(param.type_, arg_type):
            diagnostics.add(
                f"argument {idx + 1} of call to '{function.name}' expects "
                f"{display_type_name(param.type_)} but got "
                f"{display_type_name(arg_type)}",
                node=call_args[idx],
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                notes=_implicit_conversion_note(arg_type, param.type_),
            )
            resolved_argument_types.append(arg_type)
            implicit_conversions.append(None)
            continue
        resolved_argument_types.append(param.type_)
        if same_type(param.type_, arg_type):
            implicit_conversions.append(None)
        else:
            implicit_conversions.append(
                ImplicitConversion(arg_type, param.type_)
            )

    for idx, arg_type in enumerate(
        arg_types[fixed_param_count:],
        start=fixed_param_count,
    ):
        promoted_type = _variadic_argument_type(arg_type)
        if signature.is_variadic and promoted_type is not None:
            resolved_argument_types.append(promoted_type)
            if same_type(promoted_type, arg_type):
                implicit_conversions.append(None)
            else:
                implicit_conversions.append(
                    ImplicitConversion(arg_type, promoted_type)
                )
            continue
        if signature.is_variadic:
            diagnostics.add(
                f"variadic argument {idx + 1} of call to '{function.name}' "
                f"uses unsupported type {display_type_name(arg_type)}",
                node=call_args[idx],
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        resolved_argument_types.append(arg_type)
        implicit_conversions.append(None)

    return CallResolution(
        callee=_callable_resolution(function),
        signature=signature,
        resolved_argument_types=tuple(resolved_argument_types),
        result_type=signature.return_type,
        implicit_conversions=tuple(implicit_conversions),
    )


@typechecked
def resolve_return(
    diagnostics: DiagnosticBag,
    *,
    function: SemanticFunction,
    value: astx.AST | None,
    value_type: astx.DataType | None,
    node: astx.FunctionReturn,
) -> ReturnResolution:
    """
    title: Validate one return statement.
    parameters:
      diagnostics:
        type: DiagnosticBag
      function:
        type: SemanticFunction
      value:
        type: astx.AST | None
      value_type:
        type: astx.DataType | None
      node:
        type: astx.FunctionReturn
    returns:
      type: ReturnResolution
    """
    expected_type = function.signature.return_type
    callable_resolution = _callable_resolution(function)
    if is_none_type(expected_type):
        if _is_void_return_sentinel(value):
            return ReturnResolution(
                callable=callable_resolution,
                expected_type=expected_type,
                value_type=None,
                returns_void=True,
            )
        diagnostics.add(
            f"void function '{function.name}' cannot return a value of type "
            f"{display_type_name(value_type)}",
            node=node,
            code=DiagnosticCodes.SEMANTIC_INVALID_RETURN,
        )
        return ReturnResolution(
            callable=callable_resolution,
            expected_type=expected_type,
            value_type=value_type,
            returns_void=True,
        )

    if _is_void_return_sentinel(value):
        diagnostics.add(
            f"function '{function.name}' must return "
            f"{display_type_name(expected_type)}",
            node=node,
            code=DiagnosticCodes.SEMANTIC_INVALID_RETURN,
        )
        return ReturnResolution(
            callable=callable_resolution,
            expected_type=expected_type,
            value_type=None,
            returns_void=False,
        )

    if _is_void_call_value(value):
        diagnostics.add(
            f"return in '{function.name}' cannot use the result of a void "
            "call as a value",
            node=node,
            code=DiagnosticCodes.SEMANTIC_INVALID_RETURN,
        )
        return ReturnResolution(
            callable=callable_resolution,
            expected_type=expected_type,
            value_type=value_type,
            returns_void=False,
        )

    if not is_assignable(expected_type, value_type):
        diagnostics.add(
            f"return in '{function.name}' expects "
            f"{display_type_name(expected_type)} but got "
            f"{display_type_name(value_type)}",
            node=node,
            code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            notes=_implicit_conversion_note(value_type, expected_type),
        )
        return ReturnResolution(
            callable=callable_resolution,
            expected_type=expected_type,
            value_type=value_type,
            returns_void=False,
        )

    implicit_conversion = None
    if not same_type(expected_type, value_type):
        implicit_conversion = ImplicitConversion(value_type, expected_type)
    return ReturnResolution(
        callable=callable_resolution,
        expected_type=expected_type,
        value_type=value_type,
        returns_void=False,
        implicit_conversion=implicit_conversion,
    )


@typechecked
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
    if is_explicitly_castable(source_type, target_type):
        return
    diagnostics.add(
        f"unsupported cast from {display_type_name(source_type)} to "
        f"{display_type_name(target_type)}",
        node=node,
        code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
    )


@typechecked
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


@typechecked
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


@typechecked
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


@typechecked
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

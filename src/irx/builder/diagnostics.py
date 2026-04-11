"""
title: Shared lowering diagnostics helpers.
summary: >-
  Centralize the small helpers that lowering uses to raise structured
  diagnostic exceptions instead of ad hoc Exception strings.
"""

from __future__ import annotations

from typing import NoReturn, TypeVar

from llvmlite import ir

from irx import astx
from irx.analysis.types import display_type_name
from irx.diagnostics import (
    Diagnostic,
    DiagnosticCodes,
    LoweringError,
    get_node_module_key,
)
from irx.typecheck import typechecked

T = TypeVar("T")


@typechecked
def resolved_ast_type_name(node: astx.AST | None) -> str:
    """
    title: Return one node's best-effort semantic type name.
    parameters:
      node:
        type: astx.AST | None
    returns:
      type: str
    """
    if node is None:
        return "<unknown>"
    semantic = getattr(node, "semantic", None)
    resolved_type = getattr(semantic, "resolved_type", None)
    if isinstance(resolved_type, astx.DataType):
        return display_type_name(resolved_type)
    return display_type_name(getattr(node, "type_", None))


@typechecked
def lowered_value_type_name(value: ir.Value | None) -> str:
    """
    title: Return one lowered LLVM value type name.
    parameters:
      value:
        type: ir.Value | None
    returns:
      type: str
    """
    if value is None:
        return "<missing>"
    return str(value.type)


@typechecked
def raise_lowering_error(
    message: str,
    *,
    code: str,
    node: astx.AST | None = None,
    notes: tuple[str, ...] = (),
    hint: str | None = None,
    cause: Exception | None = None,
) -> NoReturn:
    """
    title: Raise one user-facing lowering diagnostic.
    parameters:
      message:
        type: str
      code:
        type: str
      node:
        type: astx.AST | None
      notes:
        type: tuple[str, Ellipsis]
      hint:
        type: str | None
      cause:
        type: Exception | None
    returns:
      type: NoReturn
    """
    raise LoweringError(
        Diagnostic(
            message=message,
            node=node,
            module_key=get_node_module_key(node),
            code=code,
            phase="lowering",
            notes=notes,
            hint=hint,
            cause=cause,
        )
    )


@typechecked
def raise_lowering_internal_error(
    message: str,
    *,
    node: astx.AST | None = None,
    code: str = DiagnosticCodes.LOWERING_MISSING_SEMANTIC_METADATA,
    notes: tuple[str, ...] = (),
    hint: str | None = None,
    cause: Exception | None = None,
) -> NoReturn:
    """
    title: Raise one internal lowering consistency diagnostic.
    parameters:
      message:
        type: str
      node:
        type: astx.AST | None
      code:
        type: str
      notes:
        type: tuple[str, Ellipsis]
      hint:
        type: str | None
      cause:
        type: Exception | None
    returns:
      type: NoReturn
    """
    raise_lowering_error(
        f"internal compiler error: {message}",
        code=code,
        node=node,
        notes=notes,
        hint=hint,
        cause=cause,
    )


@typechecked
def require_semantic_metadata(
    value: T | None,
    *,
    node: astx.AST,
    metadata: str,
    context: str,
) -> T:
    """
    title: Require one semantic sidecar field during lowering.
    parameters:
      value:
        type: T | None
      node:
        type: astx.AST
      metadata:
        type: str
      context:
        type: str
    returns:
      type: T
    """
    if value is None:
        raise_lowering_internal_error(
            f"missing {metadata} metadata during {context}",
            node=node,
            notes=(f"semantic type: {resolved_ast_type_name(node)}",),
        )
    return value


@typechecked
def require_lowered_value(
    value: ir.Value | None,
    *,
    node: astx.AST,
    context: str,
) -> ir.Value:
    """
    title: Require one lowered LLVM value during lowering.
    parameters:
      value:
        type: ir.Value | None
      node:
        type: astx.AST
      context:
        type: str
    returns:
      type: ir.Value
    """
    if value is None:
        raise_lowering_internal_error(
            f"{context} did not lower to a value",
            node=node,
            notes=(f"semantic type: {resolved_ast_type_name(node)}",),
        )
    return value

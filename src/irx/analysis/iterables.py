"""
title: Iterable capability helpers for semantic analysis.
summary: >-
  Resolve known IRx collection types into backend-neutral iteration sidecars so
  loop and comprehension analysis share one semantic authority.
"""

from __future__ import annotations

from public import public

from irx import astx
from irx.analysis.resolved_nodes import (
    IterationKind,
    IterationOrder,
    ResolvedIteration,
)
from irx.builtins.collections.list import list_element_type
from irx.typecheck import typechecked


@public
@typechecked
def resolve_iteration_capability(
    iterable_node: astx.AST,
    iterable_type: astx.DataType | None,
) -> ResolvedIteration | None:
    """
    title: Resolve one iterable semantic capability.
    summary: >-
      Return the canonical iteration sidecar for known concrete iterable types,
      or None when the expression is not iterable in the current IRx contract.
    parameters:
      iterable_node:
        type: astx.AST
      iterable_type:
        type: astx.DataType | None
    returns:
      type: ResolvedIteration | None
    """
    if iterable_type is None:
        return None

    list_type = list_element_type(iterable_type)
    if list_type is not None:
        return ResolvedIteration(
            iterable_node=iterable_node,
            iterable_type=iterable_type,
            element_type=list_type,
            kind=IterationKind.LIST,
            order=IterationOrder.INDEX,
        )

    if isinstance(iterable_type, astx.DictType) and isinstance(
        iterable_type.key_type,
        astx.DataType,
    ):
        return ResolvedIteration(
            iterable_node=iterable_node,
            iterable_type=iterable_type,
            element_type=iterable_type.key_type,
            kind=IterationKind.DICT_KEYS,
            order=IterationOrder.INSERTION,
        )

    if isinstance(iterable_type, astx.SetType) and isinstance(
        iterable_type.element_type,
        astx.DataType,
    ):
        return ResolvedIteration(
            iterable_node=iterable_node,
            iterable_type=iterable_type,
            element_type=iterable_type.element_type,
            kind=IterationKind.SET,
            order=IterationOrder.UNSPECIFIED,
        )

    if isinstance(iterable_type, astx.GeneratorType):
        return ResolvedIteration(
            iterable_node=iterable_node,
            iterable_type=iterable_type,
            element_type=iterable_type.yield_type,
            kind=IterationKind.GENERATOR,
            is_reiterable=False,
            order=IterationOrder.STABLE,
        )

    return None


__all__ = ["resolve_iteration_capability"]

"""
title: IRx-owned template AST helpers.
summary: >-
  Provide semantic-facing template metadata for compile-time specialization
  without requiring parser-level syntax support inside IRx.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import astx

from irx.astx.types import TemplateTypeVar, UnionType
from irx.typecheck import typechecked

_TEMPLATE_PARAMS_ATTR = "irx_template_params"
_TEMPLATE_ARGS_ATTR = "irx_template_args"
_TEMPLATE_SPECIALIZATION_ATTR = "irx_template_specialization_name"
_GENERATED_TEMPLATE_NODES_ATTR = "irx_generated_template_nodes"


@typechecked
@dataclass(frozen=True)
class TemplateParam:
    """
    title: One compile-time template parameter.
    summary: >-
      Describe one bounded template variable attached to a function or method
      declaration.
    attributes:
      name:
        type: str
      bound:
        type: astx.DataType
      loc:
        type: astx.SourceLocation
    """

    name: str
    bound: astx.DataType
    loc: astx.SourceLocation = astx.base.NO_SOURCE_LOCATION


@typechecked
def set_template_params(
    node: astx.AST,
    params: Iterable[TemplateParam],
) -> None:
    """
    title: Attach template parameters to one AST node.
    parameters:
      node:
        type: astx.AST
      params:
        type: Iterable[TemplateParam]
    """
    setattr(node, _TEMPLATE_PARAMS_ATTR, tuple(params))


@typechecked
def get_template_params(node: astx.AST) -> tuple[TemplateParam, ...]:
    """
    title: Return the template parameters attached to one AST node.
    parameters:
      node:
        type: astx.AST
    returns:
      type: tuple[TemplateParam, Ellipsis]
    """
    value = getattr(node, _TEMPLATE_PARAMS_ATTR, ())
    return tuple(value)


@typechecked
def is_template_node(node: astx.AST) -> bool:
    """
    title: Return whether one AST node carries template parameters.
    parameters:
      node:
        type: astx.AST
    returns:
      type: bool
    """
    return bool(get_template_params(node))


@typechecked
def set_template_args(
    node: astx.AST,
    args: Iterable[astx.DataType] | None,
) -> None:
    """
    title: Attach explicit template arguments to one call-like AST node.
    parameters:
      node:
        type: astx.AST
      args:
        type: Iterable[astx.DataType] | None
    """
    if args is None:
        setattr(node, _TEMPLATE_ARGS_ATTR, None)
        return
    setattr(node, _TEMPLATE_ARGS_ATTR, tuple(args))


@typechecked
def get_template_args(node: astx.AST) -> tuple[astx.DataType, ...] | None:
    """
    title: Return the explicit template arguments attached to one AST node.
    parameters:
      node:
        type: astx.AST
    returns:
      type: tuple[astx.DataType, Ellipsis] | None
    """
    value = getattr(node, _TEMPLATE_ARGS_ATTR, None)
    if value is None:
        return None
    return tuple(value)


@typechecked
def mark_template_specialization(
    node: astx.AST,
    specialization_name: str,
) -> None:
    """
    title: Mark one AST node as a generated template specialization.
    parameters:
      node:
        type: astx.AST
      specialization_name:
        type: str
    """
    setattr(node, _TEMPLATE_SPECIALIZATION_ATTR, specialization_name)


@typechecked
def template_specialization_name(node: astx.AST) -> str | None:
    """
    title: Return the generated specialization name for one AST node.
    parameters:
      node:
        type: astx.AST
    returns:
      type: str | None
    """
    value = getattr(node, _TEMPLATE_SPECIALIZATION_ATTR, None)
    return value if isinstance(value, str) else None


@typechecked
def is_template_specialization(node: astx.AST) -> bool:
    """
    title: Return whether one AST node is a generated specialization.
    parameters:
      node:
        type: astx.AST
    returns:
      type: bool
    """
    return template_specialization_name(node) is not None


@typechecked
def add_generated_template_node(
    module: astx.Module,
    node: astx.AST,
) -> None:
    """
    title: Attach one generated template node to a module.
    parameters:
      module:
        type: astx.Module
      node:
        type: astx.AST
    """
    nodes = list(getattr(module, _GENERATED_TEMPLATE_NODES_ATTR, ()))
    nodes.append(node)
    setattr(module, _GENERATED_TEMPLATE_NODES_ATTR, tuple(nodes))


@typechecked
def clear_generated_template_nodes(module: astx.Module) -> None:
    """
    title: Remove generated template nodes attached to one module.
    parameters:
      module:
        type: astx.Module
    """
    setattr(module, _GENERATED_TEMPLATE_NODES_ATTR, ())


@typechecked
def generated_template_nodes(module: astx.Module) -> tuple[astx.AST, ...]:
    """
    title: Return the generated template nodes attached to a module.
    parameters:
      module:
        type: astx.Module
    returns:
      type: tuple[astx.AST, Ellipsis]
    """
    value = getattr(module, _GENERATED_TEMPLATE_NODES_ATTR, ())
    return tuple(value)


__all__ = [
    "TemplateParam",
    "TemplateTypeVar",
    "UnionType",
    "add_generated_template_node",
    "clear_generated_template_nodes",
    "generated_template_nodes",
    "get_template_args",
    "get_template_params",
    "is_template_node",
    "is_template_specialization",
    "mark_template_specialization",
    "set_template_args",
    "set_template_params",
    "template_specialization_name",
]

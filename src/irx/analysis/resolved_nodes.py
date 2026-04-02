"""
title: Sidecar semantic dataclasses attached to AST nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import astx

from public import public

SPECIALIZED_BINARY_OP_EXTRA = "specialized_binary_op"


@public
@dataclass(frozen=True)
class SemanticSymbol:
    """
    title: Resolved symbol information.
    attributes:
      symbol_id:
        type: str
      name:
        type: str
      type_:
        type: astx.DataType
      is_mutable:
        type: bool
      kind:
        type: str
      declaration:
        type: astx.AST | None
    """

    symbol_id: str
    name: str
    type_: astx.DataType
    is_mutable: bool
    kind: str
    declaration: astx.AST | None = None


@public
@dataclass(frozen=True)
class SemanticFunction:
    """
    title: Resolved function information.
    attributes:
      symbol_id:
        type: str
      name:
        type: str
      return_type:
        type: astx.DataType
      args:
        type: tuple[SemanticSymbol, Ellipsis]
      prototype:
        type: astx.FunctionPrototype
      definition:
        type: astx.FunctionDef | None
    """

    symbol_id: str
    name: str
    return_type: astx.DataType
    args: tuple[SemanticSymbol, ...]
    prototype: astx.FunctionPrototype
    definition: astx.FunctionDef | None = None


@public
@dataclass(frozen=True)
class SemanticFlags:
    """
    title: Normalized semantic flags.
    attributes:
      unsigned:
        type: bool
      fast_math:
        type: bool
      fma:
        type: bool
      fma_rhs:
        type: astx.AST | None
    """

    unsigned: bool = False
    fast_math: bool = False
    fma: bool = False
    fma_rhs: astx.AST | None = None


@public
@dataclass(frozen=True)
class ResolvedOperator:
    """
    title: Normalized operator meaning.
    attributes:
      op_code:
        type: str
      result_type:
        type: astx.DataType | None
      lhs_type:
        type: astx.DataType | None
      rhs_type:
        type: astx.DataType | None
      flags:
        type: SemanticFlags
    """

    op_code: str
    result_type: astx.DataType | None = None
    lhs_type: astx.DataType | None = None
    rhs_type: astx.DataType | None = None
    flags: SemanticFlags = field(default_factory=SemanticFlags)


@public
@dataclass(frozen=True)
class ResolvedAssignment:
    """
    title: Resolved assignment target.
    attributes:
      target:
        type: SemanticSymbol
    """

    target: SemanticSymbol


@public
@dataclass
class SemanticInfo:
    """
    title: Sidecar semantic information stored on AST nodes.
    attributes:
      resolved_type:
        type: astx.DataType | None
      resolved_symbol:
        type: SemanticSymbol | None
      resolved_function:
        type: SemanticFunction | None
      resolved_operator:
        type: ResolvedOperator | None
      resolved_assignment:
        type: ResolvedAssignment | None
      semantic_flags:
        type: SemanticFlags
      extras:
        type: dict[str, Any]
    """

    resolved_type: astx.DataType | None = None
    resolved_symbol: SemanticSymbol | None = None
    resolved_function: SemanticFunction | None = None
    resolved_operator: ResolvedOperator | None = None
    resolved_assignment: ResolvedAssignment | None = None
    semantic_flags: SemanticFlags = field(default_factory=SemanticFlags)
    extras: dict[str, Any] = field(default_factory=dict)


class AssignmentBinOp(astx.BinaryOp):
    pass


class AddBinOp(astx.BinaryOp):
    pass


class SubBinOp(astx.BinaryOp):
    pass


class MulBinOp(astx.BinaryOp):
    pass


class DivBinOp(astx.BinaryOp):
    pass


class ModBinOp(astx.BinaryOp):
    pass


class EqBinOp(astx.BinaryOp):
    pass


class NeBinOp(astx.BinaryOp):
    pass


class LtBinOp(astx.BinaryOp):
    pass


class GtBinOp(astx.BinaryOp):
    pass


class LeBinOp(astx.BinaryOp):
    pass


class GeBinOp(astx.BinaryOp):
    pass


class LogicalAndBinOp(astx.BinaryOp):
    pass


class LogicalOrBinOp(astx.BinaryOp):
    pass


class BitOrBinOp(astx.BinaryOp):
    pass


class BitAndBinOp(astx.BinaryOp):
    pass


class BitXorBinOp(astx.BinaryOp):
    pass


_BINARY_OP_TYPES: dict[str, type[astx.BinaryOp]] = {
    "=": AssignmentBinOp,
    "+": AddBinOp,
    "-": SubBinOp,
    "*": MulBinOp,
    "/": DivBinOp,
    "%": ModBinOp,
    "==": EqBinOp,
    "!=": NeBinOp,
    "<": LtBinOp,
    ">": GtBinOp,
    "<=": LeBinOp,
    ">=": GeBinOp,
    "&&": LogicalAndBinOp,
    "and": LogicalAndBinOp,
    "||": LogicalOrBinOp,
    "or": LogicalOrBinOp,
    "|": BitOrBinOp,
    "&": BitAndBinOp,
    "^": BitXorBinOp,
}


def binary_op_type_for_opcode(op_code: str) -> type[astx.BinaryOp]:
    return _BINARY_OP_TYPES.get(op_code, astx.BinaryOp)


def specialize_binary_op(node: astx.BinaryOp) -> astx.BinaryOp:
    target_type = binary_op_type_for_opcode(node.op_code)
    if target_type is astx.BinaryOp or isinstance(node, target_type):
        return node

    specialized = target_type(
        node.op_code,
        node.lhs,
        node.rhs,
        loc=node.loc,
        parent=node.parent,
    )
    specialized.__dict__.update(vars(node))
    return specialized

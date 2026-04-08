"""
title: IRX-owned BinaryOp node specializations.
"""

from __future__ import annotations

import astx

from irx.typecheck import typechecked

SPECIALIZED_BINARY_OP_EXTRA = "specialized_binary_op"


@typechecked
class AssignmentBinOp(astx.BinaryOp):
    """
    title: Specialized assignment binary operation node.
    """


@typechecked
class AddBinOp(astx.BinaryOp):
    """
    title: Specialized addition binary operation node.
    """


@typechecked
class SubBinOp(astx.BinaryOp):
    """
    title: Specialized subtraction binary operation node.
    """


@typechecked
class MulBinOp(astx.BinaryOp):
    """
    title: Specialized multiplication binary operation node.
    """


@typechecked
class DivBinOp(astx.BinaryOp):
    """
    title: Specialized division binary operation node.
    """


@typechecked
class ModBinOp(astx.BinaryOp):
    """
    title: Specialized modulo binary operation node.
    """


@typechecked
class EqBinOp(astx.BinaryOp):
    """
    title: Specialized equality binary operation node.
    """


@typechecked
class NeBinOp(astx.BinaryOp):
    """
    title: Specialized inequality binary operation node.
    """


@typechecked
class LtBinOp(astx.BinaryOp):
    """
    title: Specialized less-than binary operation node.
    """


@typechecked
class GtBinOp(astx.BinaryOp):
    """
    title: Specialized greater-than binary operation node.
    """


@typechecked
class LeBinOp(astx.BinaryOp):
    """
    title: Specialized less-than-or-equal binary operation node.
    """


@typechecked
class GeBinOp(astx.BinaryOp):
    """
    title: Specialized greater-than-or-equal binary operation node.
    """


@typechecked
class LogicalAndBinOp(astx.BinaryOp):
    """
    title: Specialized logical-and binary operation node.
    """


@typechecked
class LogicalOrBinOp(astx.BinaryOp):
    """
    title: Specialized logical-or binary operation node.
    """


@typechecked
class BitOrBinOp(astx.BinaryOp):
    """
    title: Specialized bitwise-or binary operation node.
    """


@typechecked
class BitAndBinOp(astx.BinaryOp):
    """
    title: Specialized bitwise-and binary operation node.
    """


@typechecked
class BitXorBinOp(astx.BinaryOp):
    """
    title: Specialized bitwise-xor binary operation node.
    """


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


@typechecked
def binary_op_type_for_opcode(op_code: str) -> type[astx.BinaryOp]:
    """
    title: Return the specialized BinaryOp subclass for an opcode.
    parameters:
      op_code:
        type: str
    returns:
      type: type[astx.BinaryOp]
    """
    return _BINARY_OP_TYPES.get(op_code, astx.BinaryOp)


@typechecked
def specialize_binary_op(node: astx.BinaryOp) -> astx.BinaryOp:
    """
    title: Return a specialized BinaryOp instance for the given opcode.
    parameters:
      node:
        type: astx.BinaryOp
    returns:
      type: astx.BinaryOp
    """
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


__all__ = [
    "SPECIALIZED_BINARY_OP_EXTRA",
    "AddBinOp",
    "AssignmentBinOp",
    "BitAndBinOp",
    "BitOrBinOp",
    "BitXorBinOp",
    "DivBinOp",
    "EqBinOp",
    "GeBinOp",
    "GtBinOp",
    "LeBinOp",
    "LogicalAndBinOp",
    "LogicalOrBinOp",
    "LtBinOp",
    "ModBinOp",
    "MulBinOp",
    "NeBinOp",
    "SubBinOp",
    "binary_op_type_for_opcode",
    "specialize_binary_op",
]

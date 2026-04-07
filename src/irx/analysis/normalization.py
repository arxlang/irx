"""
title: Semantic normalization helpers.
summary: >-
  Normalize raw AST operator and flag information into the structured semantic
  records used by later analysis and codegen.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.resolved_nodes import ResolvedOperator, SemanticFlags
from irx.analysis.types import is_unsigned_type


def normalize_flags(
    node: astx.AST,
    *,
    lhs_type: astx.DataType | None = None,
    rhs_type: astx.DataType | None = None,
) -> SemanticFlags:
    """
    title: Normalize semantic flags from the raw AST node.
    summary: Normalize semantic flags from the raw AST node.
    parameters:
      node:
        type: astx.AST
      lhs_type:
        type: astx.DataType | None
      rhs_type:
        type: astx.DataType | None
    returns:
      type: SemanticFlags
    """
    explicit_unsigned = getattr(node, "unsigned", None)
    unsigned = (
        bool(explicit_unsigned)
        if explicit_unsigned is not None
        else is_unsigned_type(lhs_type) or is_unsigned_type(rhs_type)
    )
    return SemanticFlags(
        unsigned=unsigned,
        fast_math=bool(getattr(node, "fast_math", False)),
        fma=bool(getattr(node, "fma", False)),
        fma_rhs=getattr(node, "fma_rhs", None),
    )


def normalize_operator(
    op_code: str,
    *,
    result_type: astx.DataType | None,
    lhs_type: astx.DataType | None = None,
    rhs_type: astx.DataType | None = None,
    flags: SemanticFlags | None = None,
) -> ResolvedOperator:
    """
    title: Create a normalized operator record.
    summary: Create a normalized operator record.
    parameters:
      op_code:
        type: str
      result_type:
        type: astx.DataType | None
      lhs_type:
        type: astx.DataType | None
      rhs_type:
        type: astx.DataType | None
      flags:
        type: SemanticFlags | None
    returns:
      type: ResolvedOperator
    """
    return ResolvedOperator(
        op_code=op_code,
        result_type=result_type,
        lhs_type=lhs_type,
        rhs_type=rhs_type,
        flags=flags or SemanticFlags(),
    )

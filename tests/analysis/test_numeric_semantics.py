"""
title: Tests for scalar numeric semantics.
"""

from __future__ import annotations

import pytest

from irx import astx
from irx.analysis import SemanticError, analyze


def _semantic_type(node: astx.AST) -> astx.DataType | None:
    """
    title: Return the resolved semantic type attached to one node.
    parameters:
      node:
        type: astx.AST
    returns:
      type: astx.DataType | None
    """
    return getattr(getattr(node, "semantic", None), "resolved_type", None)


def test_analyze_promotes_int64_and_float32_to_float64() -> None:
    """
    title: Mixed int64 and float32 arithmetic should promote to float64.
    """
    expr = astx.BinaryOp(
        "+",
        astx.LiteralInt64(1),
        astx.LiteralFloat32(2.0),
    )

    analyze(expr)

    resolved_type = _semantic_type(expr)
    assert resolved_type is not None
    assert resolved_type.__class__ is astx.Float64


def test_analyze_marks_signed_result_when_signed_operand_is_wider() -> None:
    """
    title: Wider signed integers should keep signed semantics against unsigned.
    """
    expr = astx.BinaryOp(
        "/",
        astx.LiteralInt32(9),
        astx.LiteralUInt16(2),
    )

    analyze(expr)

    semantic = getattr(expr, "semantic")
    assert semantic.resolved_type.__class__ is astx.Int32
    assert semantic.semantic_flags.unsigned is False


def test_unsigned_result_when_unsigned_operand_is_not_narrower() -> None:
    """
    title: >-
      Equal-or-wider unsigned integers should drive mixed integer semantics.
    """
    expr = astx.BinaryOp(
        ">",
        astx.LiteralInt16(-1),
        astx.LiteralUInt32(1),
    )

    analyze(expr)

    semantic = getattr(expr, "semantic")
    assert semantic.resolved_type.__class__ is astx.Boolean
    assert semantic.semantic_flags.unsigned is True


def test_analyze_rejects_implicit_signed_to_unsigned_assignment() -> None:
    """
    title: >-
      Implicit signed-to-unsigned assignment should require an explicit cast.
    """
    module = astx.Module()
    proto = astx.FunctionPrototype(
        "main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    body = astx.Block()
    body.append(
        astx.VariableDeclaration(
            name="value",
            type_=astx.UInt32(),
            value=astx.LiteralInt32(1),
        )
    )
    body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=proto, body=body))

    with pytest.raises(SemanticError, match="Cannot assign value of type"):
        analyze(module)

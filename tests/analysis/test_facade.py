"""
title: Tests for the semantic-analysis facade.
"""

from __future__ import annotations

from typing import cast

import astx
import pytest

from irx.analysis import DiagnosticBag, SemanticError, analyze
from irx.analysis.resolved_nodes import SemanticInfo
from irx.system import Cast


def _module_with_main(*nodes: astx.AST) -> astx.Module:
    module = astx.Module()
    proto = astx.FunctionPrototype(
        "main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    body = astx.Block()
    for node in nodes:
        body.append(node)
    module.block.append(astx.FunctionDef(prototype=proto, body=body))
    return module


def _semantic(node: astx.AST) -> SemanticInfo:
    return cast(SemanticInfo, getattr(node, "semantic"))


def test_analyze_attaches_symbol_sidecars() -> None:
    decl = astx.VariableDeclaration(
        name="x",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
        value=astx.LiteralInt32(1),
    )
    ident = astx.Identifier("x")
    module = _module_with_main(decl, astx.FunctionReturn(ident))

    analyze(module)

    decl_symbol = _semantic(decl).resolved_symbol
    ident_symbol = _semantic(ident).resolved_symbol
    ident_type = _semantic(ident).resolved_type

    assert decl_symbol is not None
    assert ident_symbol is not None
    assert ident_type is not None
    assert decl_symbol.symbol_id == ident_symbol.symbol_id
    assert ident_type.__class__ is astx.Int32


def test_analyze_rejects_unknown_identifier() -> None:
    module = _module_with_main(astx.FunctionReturn(astx.Identifier("missing")))

    with pytest.raises(SemanticError, match="Unknown variable name"):
        analyze(module)


def test_analyze_rejects_const_write() -> None:
    decl = astx.VariableDeclaration(
        name="x",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.constant,
        value=astx.LiteralInt32(1),
    )
    assign = astx.VariableAssignment("x", astx.LiteralInt32(2))
    module = _module_with_main(
        decl, assign, astx.FunctionReturn(astx.LiteralInt32(0))
    )

    with pytest.raises(SemanticError, match="declared as constant"):
        analyze(module)


def test_analyze_rejects_break_outside_loop() -> None:
    module = _module_with_main(
        astx.BreakStmt(), astx.FunctionReturn(astx.LiteralInt32(0))
    )

    with pytest.raises(SemanticError, match="Break statement outside loop"):
        analyze(module)


def test_analyze_rejects_call_arity_mismatch() -> None:
    add_proto = astx.FunctionPrototype(
        "add",
        args=astx.Arguments(
            astx.Argument("lhs", astx.Int32()),
            astx.Argument("rhs", astx.Int32()),
        ),
        return_type=astx.Int32(),
    )
    add_body = astx.Block()
    add_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    add_fn = astx.FunctionDef(prototype=add_proto, body=add_body)

    module = astx.Module()
    module.block.append(add_fn)
    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_body = astx.Block()
    main_body.append(astx.FunctionCall("add", [astx.LiteralInt32(1)]))
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=main_body))

    with pytest.raises(SemanticError, match="Incorrect # arguments passed"):
        analyze(module)


def test_analyze_rejects_missing_return() -> None:
    proto = astx.FunctionPrototype(
        "value",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    module = astx.Module()
    module.block.append(astx.FunctionDef(prototype=proto, body=astx.Block()))

    with pytest.raises(SemanticError, match="missing a return statement"):
        analyze(module)


def test_analyze_normalizes_binary_flags() -> None:
    expr = astx.BinaryOp(
        "*",
        astx.LiteralFloat32(1.0),
        astx.LiteralFloat32(2.0),
    )
    expr.fast_math = True  # type: ignore[attr-defined]
    expr.fma = True  # type: ignore[attr-defined]
    fma_rhs = astx.LiteralFloat32(3.0)
    expr.fma_rhs = fma_rhs  # type: ignore[attr-defined]

    analyze(expr)

    semantic = _semantic(expr)

    assert semantic.semantic_flags.fast_math is True
    assert semantic.semantic_flags.fma is True
    assert semantic.semantic_flags.fma_rhs is fma_rhs


def test_analyze_allows_numeric_casts() -> None:
    expr = Cast(
        value=astx.LiteralFloat32(7.9),
        target_type=astx.Int32(),
    )

    analyze(expr)

    assert _semantic(expr).resolved_type.__class__ is astx.Int32


def test_analyze_keeps_if_branch_bindings_visible_after_if() -> None:
    branchy_proto = astx.FunctionPrototype(
        "branchy",
        args=astx.Arguments(astx.Argument("x", astx.Int32())),
        return_type=astx.Int32(),
    )
    branchy_body = astx.Block()

    cond = astx.BinaryOp(
        op_code="<",
        lhs=astx.Identifier("x"),
        rhs=astx.LiteralInt32(0),
    )
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(7)))
    else_block = astx.Block()
    else_block.append(
        astx.InlineVariableDeclaration(
            "y", type_=astx.Int32(), value=astx.LiteralInt32(5)
        )
    )
    branchy_body.append(
        astx.IfStmt(condition=cond, then=then_block, else_=else_block)
    )
    return_ident = astx.Identifier("y")
    branchy_body.append(astx.FunctionReturn(return_ident))

    module = astx.Module()
    module.block.append(
        astx.FunctionDef(prototype=branchy_proto, body=branchy_body)
    )

    analyze(module)

    resolved_symbol = _semantic(return_ident).resolved_symbol

    assert resolved_symbol is not None
    assert resolved_symbol.name == "y"


def test_diagnostic_bag_formats_messages() -> None:
    bag = DiagnosticBag()
    bag.add("unknown identifier")

    assert "unknown identifier" in bag.format()

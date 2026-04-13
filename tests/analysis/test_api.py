"""
title: Tests for the semantic-analysis API.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from irx import astx
from irx.analysis import (
    CompilationSession,
    DiagnosticBag,
    SemanticAnalyzer,
    SemanticContract,
    SemanticError,
    analyze,
    analyze_modules,
    get_semantic_contract,
)
from irx.analysis.module_symbols import (
    qualified_function_name,
    qualified_struct_name,
)
from irx.analysis.resolved_nodes import SemanticInfo
from irx.astx.binary_op import (
    SPECIALIZED_BINARY_OP_EXTRA,
    AddBinOp,
)

from tests.conftest import (
    StaticImportResolver,
    make_module,
    make_parsed_module,
)


def _module_with_main(*nodes: astx.AST) -> astx.Module:
    """
    title: Module with main.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
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
    """
    title: Semantic.
    parameters:
      node:
        type: astx.AST
    returns:
      type: SemanticInfo
    """
    return cast(SemanticInfo, getattr(node, "semantic"))


def _block(*nodes: astx.AST) -> astx.Block:
    """
    title: Block.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Block
    """
    block = astx.Block()
    for node in nodes:
        block.append(node)
    return block


def _with_loc(node: astx.AST, line: int, col: int) -> astx.AST:
    """
    title: Attach a simple source location to one AST node.
    parameters:
      node:
        type: astx.AST
      line:
        type: int
      col:
        type: int
    returns:
      type: astx.AST
    """
    node.loc = SimpleNamespace(line=line, col=col)
    return node


def test_analyze_attaches_symbol_sidecars() -> None:
    """
    title: Test analyze attaches symbol sidecars.
    """
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
    """
    title: Test analyze rejects unknown identifier.
    """
    module = _module_with_main(astx.FunctionReturn(astx.Identifier("missing")))

    with pytest.raises(SemanticError, match="cannot resolve name"):
        analyze(module)


def test_unknown_identifier_diagnostic_includes_code_and_location() -> None:
    """
    title: Unknown-name diagnostics should render stable code and source info.
    """
    missing = _with_loc(astx.Identifier("missing"), 12, 8)
    module = _module_with_main(astx.FunctionReturn(missing))

    with pytest.raises(SemanticError) as exc_info:
        analyze(module)

    formatted = str(exc_info.value)

    assert "12:8" in formatted
    assert "IRX-S001" in formatted
    assert "cannot resolve name 'missing'" in formatted


def test_assignment_type_mismatch_reports_expected_and_actual_types() -> None:
    """
    title: >-
      Assignment diagnostics should explain both expected and actual types.
    """
    assignment = _with_loc(
        astx.VariableAssignment("count", astx.LiteralFloat64(1.5)),
        6,
        4,
    )
    module = _module_with_main(
        astx.VariableDeclaration(
            name="count",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
            value=astx.LiteralInt32(1),
        ),
        assignment,
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    with pytest.raises(SemanticError) as exc_info:
        analyze(module)

    formatted = str(exc_info.value)

    assert "IRX-S010" in formatted
    assert "cannot assign Float64 to 'count' of type Int32" in formatted


def test_argument_type_mismatch_reports_argument_position_and_callee() -> None:
    """
    title: Call diagnostics should identify the failing argument and callee.
    """
    helper = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "mix",
            args=astx.Arguments(
                astx.Argument("lhs", astx.Int32()),
                astx.Argument("rhs", astx.Float64()),
            ),
            return_type=astx.Int32(),
        ),
        body=_block(astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    call = _with_loc(
        astx.FunctionCall(
            "mix",
            [astx.LiteralInt32(1), astx.LiteralUTF8String("oops")],
        ),
        9,
        6,
    )
    module = astx.Module()
    module.block.append(helper)
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "main",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(call, astx.FunctionReturn(astx.LiteralInt32(0))),
        )
    )

    with pytest.raises(SemanticError) as exc_info:
        analyze(module)

    formatted = str(exc_info.value)

    assert "IRX-S010" in formatted
    assert (
        "argument 2 of call to 'mix' expects Float64 but got UTF8String"
        in formatted
    )


def test_boolean_condition_diagnostic_reports_actual_type() -> None:
    """
    title: Control-flow condition diagnostics should report the actual type.
    """
    condition = _with_loc(astx.LiteralInt32(1), 4, 5)
    module = _module_with_main(
        astx.IfStmt(
            condition=condition,
            then=_block(astx.FunctionReturn(astx.LiteralInt32(1))),
            else_=_block(astx.FunctionReturn(astx.LiteralInt32(0))),
        )
    )

    with pytest.raises(SemanticError) as exc_info:
        analyze(module)

    formatted = str(exc_info.value)

    assert "IRX-S003" in formatted
    assert "if condition must be Boolean, got Int32" in formatted


def test_analyze_rejects_imports_without_multimodule_resolver() -> None:
    """
    title: Test analyze rejects imports without a resolver-backed session.
    """
    module = make_module(
        "app.main",
        astx.ImportStmt([astx.AliasExpr("lib")]),
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "main",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(astx.FunctionReturn(astx.LiteralInt32(0))),
        ),
    )

    with pytest.raises(
        SemanticError,
        match="Import statements require analyze_modules",
    ):
        analyze(module)


def test_multimodule_semantic_failure_includes_module_attribution() -> None:
    """
    title: Multi-module semantic diagnostics should render the failing module.
    """
    helper_return = astx.FunctionReturn(
        _with_loc(astx.Identifier("missing_value"), 7, 3)
    )
    helper_module = make_parsed_module(
        "lib.math",
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "helper",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(helper_return),
        ),
    )
    app_module = make_parsed_module(
        "app.main",
        astx.ImportStmt([astx.AliasExpr("lib.math")]),
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "main",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(astx.FunctionReturn(astx.LiteralInt32(0))),
        ),
    )
    resolver = StaticImportResolver(
        {
            "app.main": app_module,
            "lib.math": helper_module,
        }
    )

    with pytest.raises(SemanticError) as exc_info:
        analyze_modules(app_module, resolver)

    formatted = str(exc_info.value)

    assert "lib.math:7:3" in formatted
    assert "IRX-S001" in formatted
    assert "cannot resolve name 'missing_value'" in formatted


def test_analyze_rejects_const_write() -> None:
    """
    title: Test analyze rejects const write.
    """
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


def test_analyze_rejects_duplicate_local_variable_declaration() -> None:
    """
    title: Test analyze rejects duplicate local variable declarations.
    """
    module = _module_with_main(
        astx.VariableDeclaration(
            name="x",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
            value=astx.LiteralInt32(1),
        ),
        astx.VariableDeclaration(
            name="x",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
            value=astx.LiteralInt32(2),
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    with pytest.raises(SemanticError, match="Identifier already declared: x"):
        analyze(module)


def test_analyze_rejects_break_outside_loop() -> None:
    """
    title: Test analyze rejects break outside loop.
    """
    module = _module_with_main(
        astx.BreakStmt(), astx.FunctionReturn(astx.LiteralInt32(0))
    )

    with pytest.raises(SemanticError, match="Break statement outside loop"):
        analyze(module)


def test_analyze_rejects_continue_outside_loop() -> None:
    """
    title: Test analyze rejects continue outside loop.
    """
    module = _module_with_main(
        astx.ContinueStmt(), astx.FunctionReturn(astx.LiteralInt32(0))
    )

    with pytest.raises(SemanticError, match="Continue statement outside loop"):
        analyze(module)


def test_analyze_rejects_call_arity_mismatch() -> None:
    """
    title: Test analyze rejects call arity mismatch.
    """
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

    with pytest.raises(
        SemanticError,
        match="call to 'add' expects 2 arguments but got 1",
    ):
        analyze(module)


def test_analyze_rejects_missing_return() -> None:
    """
    title: Test analyze rejects missing return.
    """
    proto = astx.FunctionPrototype(
        "value",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    module = astx.Module()
    module.block.append(astx.FunctionDef(prototype=proto, body=astx.Block()))

    with pytest.raises(SemanticError, match="missing a return statement"):
        analyze(module)


def test_analyze_rejects_duplicate_function_definitions() -> None:
    """
    title: Test analyze rejects duplicate function definitions.
    """
    module = make_module(
        "app.main",
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "helper",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(astx.FunctionReturn(astx.LiteralInt32(1))),
        ),
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "helper",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(astx.FunctionReturn(astx.LiteralInt32(2))),
        ),
    )

    with pytest.raises(
        SemanticError,
        match="Function 'helper' already defined",
    ):
        analyze(module)


def test_analyze_rejects_duplicate_struct_definitions() -> None:
    """
    title: Test analyze rejects duplicate struct definitions.
    """
    module = make_module(
        "app.main",
        astx.StructDefStmt(
            name="Point",
            attributes=[
                astx.VariableDeclaration(name="x", type_=astx.Int32()),
            ],
        ),
        astx.StructDefStmt(
            name="Point",
            attributes=[
                astx.VariableDeclaration(name="y", type_=astx.Int32()),
            ],
        ),
    )

    with pytest.raises(SemanticError, match="Struct 'Point' already defined"):
        analyze(module)


def test_analyze_normalizes_binary_flags() -> None:
    """
    title: Test analyze normalizes binary flags.
    """
    expr = astx.BinaryOp(
        "*",
        astx.LiteralFloat32(1.0),
        astx.LiteralFloat32(2.0),
    )
    expr.fast_math = True
    expr.fma = True
    fma_rhs = astx.LiteralFloat32(3.0)
    expr.fma_rhs = fma_rhs

    analyze(expr)

    semantic = _semantic(expr)

    assert semantic.semantic_flags.fast_math is True
    assert semantic.semantic_flags.fma is True
    assert semantic.semantic_flags.fma_rhs is fma_rhs


def test_analyze_attaches_specialized_binary_op() -> None:
    """
    title: Test analyze attaches specialized binary op.
    """
    expr = astx.BinaryOp(
        "+",
        astx.LiteralInt32(1),
        astx.LiteralInt32(2),
    )

    analyze(expr)

    specialized = _semantic(expr).extras[SPECIALIZED_BINARY_OP_EXTRA]

    assert isinstance(specialized, AddBinOp)
    assert specialized.op_code == "+"
    assert specialized.lhs is expr.lhs
    assert specialized.rhs is expr.rhs


def test_analyze_allows_numeric_casts() -> None:
    """
    title: Test analyze allows numeric casts.
    """
    expr = astx.Cast(
        value=astx.LiteralFloat32(7.9),
        target_type=astx.Int32(),
    )

    analyze(expr)

    assert _semantic(expr).resolved_type.__class__ is astx.Int32


def test_analyze_attaches_argument_symbols_to_parameters() -> None:
    """
    title: Test analyze attaches argument symbols to function parameters.
    """
    arg = astx.Argument("value", astx.Int32())
    return_ident = astx.Identifier("value")
    module = make_module(
        "app.main",
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "echo",
                args=astx.Arguments(arg),
                return_type=astx.Int32(),
            ),
            body=_block(astx.FunctionReturn(return_ident)),
        ),
    )

    analyze(module)

    arg_symbol = _semantic(arg).resolved_symbol
    ident_symbol = _semantic(return_ident).resolved_symbol

    assert arg_symbol is not None
    assert ident_symbol is not None
    assert arg_symbol.kind == "argument"
    assert arg_symbol.symbol_id == ident_symbol.symbol_id


def test_analyze_uses_distinct_lexical_and_visible_name_resolution() -> None:
    """
    title: Test local scope lookup stays distinct from module-visible lookup.
    """
    call = astx.FunctionCall("helper", [])
    return_ident = astx.Identifier("helper")
    module = make_module(
        "app.main",
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "helper",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(astx.FunctionReturn(astx.LiteralInt32(7))),
        ),
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "main",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(
                astx.VariableDeclaration(
                    name="helper",
                    type_=astx.Int32(),
                    mutability=astx.MutabilityKind.mutable,
                    value=astx.LiteralInt32(5),
                ),
                call,
                astx.FunctionReturn(return_ident),
            ),
        ),
    )

    analyze(module)

    resolved_function = _semantic(call).resolved_function
    resolved_symbol = _semantic(return_ident).resolved_symbol

    assert resolved_function is not None
    assert resolved_symbol is not None
    assert resolved_function.name == "helper"
    assert resolved_symbol.name == "helper"
    assert resolved_symbol.kind == "variable"


def test_analyze_prefers_inner_for_scope_symbols() -> None:
    """
    title: Test inner loop scopes can shadow an outer local.
    """
    outer_decl = astx.VariableDeclaration(
        name="x",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
        value=astx.LiteralInt32(99),
    )
    loop_var = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
    )
    inner_ident = astx.Identifier("x")
    outer_ident = astx.Identifier("x")
    module = _module_with_main(
        outer_decl,
        astx.ForRangeLoopStmt(
            variable=loop_var,
            start=astx.LiteralInt32(0),
            end=astx.LiteralInt32(3),
            step=astx.LiteralInt32(1),
            body=_block(inner_ident),
        ),
        astx.FunctionReturn(outer_ident),
    )

    analyze(module)

    outer_symbol = _semantic(outer_decl).resolved_symbol
    loop_symbol = _semantic(loop_var).resolved_symbol
    inner_symbol = _semantic(inner_ident).resolved_symbol
    resolved_outer_ident = _semantic(outer_ident).resolved_symbol

    assert outer_symbol is not None
    assert loop_symbol is not None
    assert inner_symbol is not None
    assert resolved_outer_ident is not None
    assert inner_symbol.symbol_id == loop_symbol.symbol_id
    assert inner_symbol.symbol_id != outer_symbol.symbol_id
    assert resolved_outer_ident.symbol_id == outer_symbol.symbol_id


def test_analyze_prefers_inner_for_count_scope_symbols() -> None:
    """
    title: Test for-count loop initializers can shadow an outer local.
    """
    outer_decl = astx.VariableDeclaration(
        name="x",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
        value=astx.LiteralInt32(99),
    )
    loop_var = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )
    inner_ident = astx.Identifier("x")
    outer_ident = astx.Identifier("x")
    module = _module_with_main(
        outer_decl,
        astx.ForCountLoopStmt(
            initializer=loop_var,
            condition=astx.BinaryOp(
                op_code="<",
                lhs=astx.Identifier("x"),
                rhs=astx.LiteralInt32(1),
            ),
            update=astx.UnaryOp(op_code="++", operand=astx.Identifier("x")),
            body=_block(inner_ident),
        ),
        astx.FunctionReturn(outer_ident),
    )

    analyze(module)

    outer_symbol = _semantic(outer_decl).resolved_symbol
    loop_symbol = _semantic(loop_var).resolved_symbol
    inner_symbol = _semantic(inner_ident).resolved_symbol
    resolved_outer_ident = _semantic(outer_ident).resolved_symbol

    assert outer_symbol is not None
    assert loop_symbol is not None
    assert inner_symbol is not None
    assert resolved_outer_ident is not None
    assert inner_symbol.symbol_id == loop_symbol.symbol_id
    assert inner_symbol.symbol_id != outer_symbol.symbol_id
    assert resolved_outer_ident.symbol_id == outer_symbol.symbol_id


def test_analyze_preserves_module_qualified_function_and_struct_names() -> (
    None
):
    """
    title: Test analyze keeps module-qualified function and struct names.
    """
    struct = astx.StructDefStmt(
        name="Point",
        attributes=[astx.VariableDeclaration(name="x", type_=astx.Int32())],
    )
    function = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=_block(astx.FunctionReturn(astx.LiteralInt32(1))),
    )
    module = make_module("pkg.tools", struct, function)

    analyze(module)

    resolved_struct = _semantic(struct).resolved_struct
    resolved_function = _semantic(function).resolved_function

    assert resolved_struct is not None
    assert resolved_function is not None
    assert resolved_struct.qualified_name == qualified_struct_name(
        "pkg.tools",
        "Point",
    )
    assert resolved_function.qualified_name == qualified_function_name(
        "pkg.tools",
        "helper",
    )


def test_analyze_keeps_if_branch_bindings_visible_after_if() -> None:
    """
    title: Test analyze keeps if branch bindings visible after if.
    """
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
    """
    title: Test diagnostic bag formats messages.
    """
    bag = DiagnosticBag()
    bag.add("unknown identifier")

    assert "unknown identifier" in bag.format()


def test_public_analysis_contract_is_stable() -> None:
    """
    title: Test public analysis contract is stable.
    """
    contract = get_semantic_contract()

    assert SemanticAnalyzer.__name__ == "SemanticAnalyzer"
    assert isinstance(contract, SemanticContract)
    assert tuple(phase.name for phase in contract.stable_phases) == (
        "module_graph_expansion",
        "top_level_predeclaration",
        "top_level_import_resolution",
        "semantic_validation",
    )
    assert contract.required_node_semantic_fields == (
        "resolved_type",
        "resolved_symbol",
        "resolved_function",
        "resolved_callable",
        "resolved_call",
        "resolved_return",
        "resolved_struct",
        "resolved_module",
        "resolved_imports",
        "resolved_operator",
        "resolved_assignment",
        "resolved_field_access",
        "semantic_flags",
        "extras",
    )
    assert contract.required_session_fields == (
        "root",
        "modules",
        "graph",
        "load_order",
        "visible_bindings",
    )
    assert set(contract.required_node_semantic_fields) <= set(
        SemanticInfo.__dataclass_fields__
    )
    assert set(contract.required_session_fields) <= set(
        CompilationSession.__dataclass_fields__
    )
    assert contract.allowed_host_entrypoints == (
        "irx.analysis.analyze",
        "irx.analysis.analyze_module",
        "irx.analysis.analyze_modules",
    )
    semantic_boundary = contract.phase_error_boundaries[0]
    assert semantic_boundary.phase == "semantic"
    assert semantic_boundary.raises == "SemanticError"

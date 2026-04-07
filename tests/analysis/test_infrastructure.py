"""
title: Tests for extracted semantic infrastructure.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.bindings import VisibleBindings
from irx.analysis.context import SemanticContext
from irx.analysis.factories import SemanticEntityFactory
from irx.analysis.module_symbols import qualified_function_name
from irx.analysis.registry import SemanticRegistry


def test_semantic_registry_rejects_duplicate_locals_in_one_scope() -> None:
    """
    title: SemanticRegistry rejects duplicate lexical declarations.
    """
    context = SemanticContext()
    registry = SemanticRegistry(context, SemanticEntityFactory(context))

    with context.in_module("app.main"):
        with context.scope("function"):
            registry.declare_local(
                "value",
                astx.Int32(),
                is_mutable=True,
                declaration=astx.VariableDeclaration(
                    name="value",
                    type_=astx.Int32(),
                ),
            )
            registry.declare_local(
                "value",
                astx.Int32(),
                is_mutable=True,
                declaration=astx.VariableDeclaration(
                    name="value",
                    type_=astx.Int32(),
                ),
            )

    assert "Identifier already declared: value" in context.diagnostics.format()


def test_semantic_registry_builds_parameter_symbols_consistently() -> None:
    """
    title: SemanticRegistry creates argument symbols with function identity.
    """
    context = SemanticContext()
    registry = SemanticRegistry(context, SemanticEntityFactory(context))
    prototype = astx.FunctionPrototype(
        "echo",
        args=astx.Arguments(astx.Argument("value", astx.Int32())),
        return_type=astx.Int32(),
    )

    with context.in_module("pkg.tools"):
        function = registry.register_function(prototype)

    assert function.qualified_name == qualified_function_name(
        "pkg.tools",
        "echo",
    )
    assert len(function.args) == 1
    assert function.args[0].kind == "argument"
    assert function.args[0].name == "value"


def test_visible_bindings_report_conflicts_independently() -> None:
    """
    title: VisibleBindings reports namespace conflicts directly.
    """
    context = SemanticContext()
    factory = SemanticEntityFactory(context)
    bindings = VisibleBindings(context=context, factory=factory)
    prototype = astx.FunctionPrototype(
        "shared",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    struct_node = astx.StructDefStmt(
        name="shared",
        attributes=[astx.VariableDeclaration(name="x", type_=astx.Int32())],
    )

    with context.in_module("app.main"):
        function = factory.make_function("app.main", prototype)
        struct = factory.make_struct("app.main", struct_node)
        bindings.bind_function("shared", function, node=prototype)
        bindings.bind_struct("shared", struct, node=struct_node)

    assert "Conflicting binding for 'shared'" in context.diagnostics.format()

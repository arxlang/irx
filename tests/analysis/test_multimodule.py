"""
title: Tests for multi-module semantic analysis.
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.analysis import (
    ModuleKey,
    ParsedModule,
    SemanticError,
    analyze_modules,
)
from irx.analysis.module_symbols import qualified_function_name
from irx.analysis.resolved_nodes import SemanticInfo

from tests.conftest import StaticImportResolver, make_parsed_module


def _semantic(node: astx.AST) -> SemanticInfo:
    """
    title: Return semantic sidecar information for a node.
    parameters:
      node:
        type: astx.AST
    returns:
      type: SemanticInfo
    """
    return cast(SemanticInfo, getattr(node, "semantic"))


def _int_function(
    name: str,
    *body_nodes: astx.AST,
) -> astx.FunctionDef:
    """
    title: Build a small int32-returning function.
    parameters:
      name:
        type: str
      body_nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.FunctionDef
    """
    body = astx.Block()
    for node in body_nodes:
        body.append(node)
    return astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name,
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=body,
    )


def _point_struct(name: str = "Point") -> astx.StructDefStmt:
    """
    title: Build a simple struct definition.
    parameters:
      name:
        type: str
    returns:
      type: astx.StructDefStmt
    """
    return astx.StructDefStmt(
        name=name,
        attributes=[
            astx.VariableDeclaration(name="x", type_=astx.Int32()),
            astx.VariableDeclaration(name="y", type_=astx.Int32()),
        ],
    )


def test_analyze_modules_resolves_imported_function_identity() -> None:
    """
    title: Imported functions resolve to their original defining module.
    """
    import_stmt = astx.ImportFromStmt(
        module="lib",
        names=[astx.AliasExpr("foo")],
    )
    call = astx.FunctionCall("foo", [])
    root = make_parsed_module(
        "app.main",
        import_stmt,
        _int_function("main", astx.FunctionReturn(call)),
    )
    lib = make_parsed_module(
        "lib",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(7))),
    )

    analyze_modules(root, StaticImportResolver({"lib": lib}))

    resolved_function = _semantic(call).resolved_function

    assert resolved_function is not None
    assert resolved_function.module_key == ModuleKey("lib")
    assert resolved_function.qualified_name == qualified_function_name(
        ModuleKey("lib"),
        "foo",
    )


def test_analyze_modules_resolves_import_alias_to_original_function() -> None:
    """
    title: Function aliases keep the original semantic identity.
    """
    import_stmt = astx.ImportFromStmt(
        module="lib",
        names=[astx.AliasExpr("foo", asname="alias")],
    )
    call = astx.FunctionCall("alias", [])
    root = make_parsed_module(
        "app.main",
        import_stmt,
        _int_function("main", astx.FunctionReturn(call)),
    )
    lib = make_parsed_module(
        "lib",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(1))),
    )

    analyze_modules(root, StaticImportResolver({"lib": lib}))

    resolved_function = _semantic(call).resolved_function
    resolved_import = _semantic(import_stmt).resolved_imports[0]

    assert resolved_function is not None
    assert resolved_import.local_name == "alias"
    assert resolved_import.binding.function is resolved_function


def test_analyze_modules_registers_plain_module_import_binding() -> None:
    """
    title: Plain imports create semantic module bindings.
    """
    import_stmt = astx.ImportStmt([astx.AliasExpr("lib")])
    root = make_parsed_module(
        "app.main",
        import_stmt,
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    lib = make_parsed_module(
        "lib",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(2))),
    )

    session = analyze_modules(root, StaticImportResolver({"lib": lib}))

    alias_semantic = _semantic(import_stmt.names[0])

    assert alias_semantic.resolved_module is not None
    assert alias_semantic.resolved_module.module_key == ModuleKey("lib")
    assert session.visible_bindings[root.key]["lib"].kind == "module"


def test_analyze_modules_resolves_imported_struct_binding() -> None:
    """
    title: Imported structs keep the original defining module.
    """
    import_stmt = astx.ImportFromStmt(
        module="models",
        names=[astx.AliasExpr("Point", asname="UserPoint")],
    )
    root = make_parsed_module(
        "app.main",
        import_stmt,
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    models = make_parsed_module("models", _point_struct("Point"))

    analyze_modules(root, StaticImportResolver({"models": models}))

    resolved_import = _semantic(import_stmt).resolved_imports[0]

    assert resolved_import.binding.struct is not None
    assert resolved_import.binding.struct.module_key == ModuleKey("models")
    assert resolved_import.local_name == "UserPoint"


def test_analyze_modules_reports_missing_module() -> None:
    """
    title: Missing modules produce a semantic diagnostic.
    """
    root = make_parsed_module(
        "app.main",
        astx.ImportStmt([astx.AliasExpr("missing")]),
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )

    with pytest.raises(
        SemanticError, match="Unable to resolve module 'missing'"
    ):
        analyze_modules(root, StaticImportResolver({}))


def test_analyze_modules_reports_missing_imported_symbol() -> None:
    """
    title: Missing imported symbols produce a semantic diagnostic.
    """
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(module="lib", names=[astx.AliasExpr("missing")]),
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    lib = make_parsed_module(
        "lib",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(0))),
    )

    with pytest.raises(SemanticError, match="Imported symbol 'missing'"):
        analyze_modules(root, StaticImportResolver({"lib": lib}))


def test_analyze_modules_reports_conflicting_alias_bindings() -> None:
    """
    title: Conflicting aliases are rejected.
    """
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(
            module="a",
            names=[astx.AliasExpr("foo", asname="shared")],
        ),
        astx.ImportFromStmt(
            module="b",
            names=[astx.AliasExpr("foo", asname="shared")],
        ),
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    module_a = make_parsed_module(
        "a",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(1))),
    )
    module_b = make_parsed_module(
        "b",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(2))),
    )

    with pytest.raises(
        SemanticError, match="Conflicting binding for 'shared'"
    ):
        analyze_modules(
            root,
            StaticImportResolver({"a": module_a, "b": module_b}),
        )


def test_analyze_modules_rejects_import_cycles() -> None:
    """
    title: Import cycles are rejected with a concrete cycle path.
    """
    module_a = make_parsed_module(
        "a",
        astx.ImportStmt([astx.AliasExpr("b")]),
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    module_b = make_parsed_module(
        "b",
        astx.ImportStmt([astx.AliasExpr("a")]),
        _int_function("helper", astx.FunctionReturn(astx.LiteralInt32(0))),
    )

    with pytest.raises(
        SemanticError, match="Cyclic import detected: a -> b -> a"
    ):
        analyze_modules(
            module_a,
            StaticImportResolver({"a": module_a, "b": module_b}),
        )


def test_analyze_modules_keeps_same_bare_function_names_distinct() -> None:
    """
    title: Same bare function names remain distinct across modules.
    """
    call_a = astx.FunctionCall("foo_a", [])
    call_b = astx.FunctionCall("foo_b", [])
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(
            module="a",
            names=[astx.AliasExpr("foo", asname="foo_a")],
        ),
        astx.ImportFromStmt(
            module="b",
            names=[astx.AliasExpr("foo", asname="foo_b")],
        ),
        _int_function(
            "main",
            call_a,
            astx.FunctionReturn(call_b),
        ),
    )
    module_a = make_parsed_module(
        "a",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(1))),
    )
    module_b = make_parsed_module(
        "b",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(2))),
    )

    analyze_modules(
        root,
        StaticImportResolver({"a": module_a, "b": module_b}),
    )

    function_a = _semantic(call_a).resolved_function
    function_b = _semantic(call_b).resolved_function

    assert function_a is not None
    assert function_b is not None
    assert function_a.qualified_name != function_b.qualified_name


@pytest.mark.parametrize(
    ("root", "pattern"),
    [
        (
            make_parsed_module(
                "app.main",
                _int_function(
                    "main",
                    astx.ImportStmt([astx.AliasExpr("lib")]),
                    astx.FunctionReturn(astx.LiteralInt32(0)),
                ),
            ),
            "module top level",
        ),
        (
            make_parsed_module(
                "app.main",
                _int_function(
                    "main",
                    astx.ImportExpr([astx.AliasExpr("lib")]),
                    astx.FunctionReturn(astx.LiteralInt32(0)),
                ),
            ),
            "Import expressions are not supported",
        ),
    ],
)
def test_analyze_modules_rejects_unsupported_import_forms(
    root: ParsedModule,
    pattern: str,
) -> None:
    """
    title: Nested and expression-form imports are rejected.
    parameters:
      root:
        type: ParsedModule
      pattern:
        type: str
    """
    lib = make_parsed_module(
        "lib",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(0))),
    )

    with pytest.raises(SemanticError, match=pattern):
        analyze_modules(root, StaticImportResolver({"lib": lib}))

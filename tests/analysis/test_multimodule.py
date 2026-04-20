"""
title: Tests for multi-module semantic analysis.
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.analysis import (
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


def _sum2_function(name: str = "sum2") -> astx.FunctionDef:
    """
    title: Build a small float64 helper with two parameters.
    parameters:
      name:
        type: str
    returns:
      type: astx.FunctionDef
    """
    body = astx.Block()
    body.append(astx.FunctionReturn(astx.Identifier("lhs")))
    return astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name,
            args=astx.Arguments(
                astx.Argument("lhs", astx.Float64()),
                astx.Argument("rhs", astx.Float64()),
            ),
            return_type=astx.Float64(),
        ),
        body=body,
    )


def _shape_class(name: str = "Shape") -> astx.ClassDefStmt:
    """
    title: Build a simple class definition.
    parameters:
      name:
        type: str
    returns:
      type: astx.ClassDefStmt
    """
    return astx.ClassDefStmt(
        name=name,
        attributes=[
            astx.VariableDeclaration(
                name="rank",
                type_=astx.Int32(),
                mutability=astx.MutabilityKind.mutable,
            )
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
    assert resolved_function.module_key == "lib"
    assert resolved_function.qualified_name == qualified_function_name(
        "lib",
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
    assert alias_semantic.resolved_module.module_key == "lib"
    assert session.visible_bindings[root.key]["lib"].kind == "module"


def test_analyze_modules_resolves_child_module_from_import() -> None:
    """
    title: Child modules bind as namespaces through import-from sugar.
    """
    namespace_call = astx.MethodCall(
        astx.Identifier("stats"),
        "sum2",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    import_stmt = astx.ImportFromStmt(
        module="sciarx",
        names=[astx.AliasExpr("stats")],
    )
    root = make_parsed_module(
        "app.main",
        import_stmt,
        _int_function(
            "main",
            namespace_call,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    sciarx = make_parsed_module("sciarx")
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    session = analyze_modules(
        root,
        StaticImportResolver(
            {
                "sciarx": sciarx,
                "sciarx.stats": stats_module,
            }
        ),
    )

    resolved_import = _semantic(import_stmt).resolved_imports[0]
    alias_semantic = _semantic(import_stmt.names[0])
    resolved_member = _semantic(namespace_call).resolved_module_member_access

    assert resolved_import.source_module_key == "sciarx.stats"
    assert resolved_import.binding.module is not None
    assert alias_semantic.resolved_module is not None
    assert alias_semantic.resolved_module.module_key == "sciarx.stats"
    assert session.visible_bindings[root.key]["stats"].kind == "module"
    assert session.graph == {
        "app.main": {"sciarx", "sciarx.stats"},
        "sciarx": set(),
        "sciarx.stats": set(),
    }
    assert resolved_member is not None
    assert resolved_member.module.module_key == "sciarx.stats"


def test_analyze_modules_resolves_grouped_child_module_from_imports() -> None:
    """
    title: Grouped import-from statements resolve child modules independently.
    """
    stats_call = astx.MethodCall(
        astx.Identifier("stats"),
        "sum2",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    linalg_call = astx.MethodCall(
        astx.Identifier("linalg"),
        "norm2",
        [astx.LiteralFloat64(3.0), astx.LiteralFloat64(4.0)],
    )
    import_stmt = astx.ImportFromStmt(
        module="sciarx",
        names=[astx.AliasExpr("stats"), astx.AliasExpr("linalg")],
    )
    root = make_parsed_module(
        "app.main",
        import_stmt,
        _int_function(
            "main",
            stats_call,
            linalg_call,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    sciarx = make_parsed_module("sciarx")
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())
    linalg_module = make_parsed_module(
        "sciarx.linalg",
        _sum2_function("norm2"),
    )

    session = analyze_modules(
        root,
        StaticImportResolver(
            {
                "sciarx": sciarx,
                "sciarx.stats": stats_module,
                "sciarx.linalg": linalg_module,
            }
        ),
    )

    stats_import, linalg_import = _semantic(import_stmt).resolved_imports
    resolved_stats = _semantic(stats_call).resolved_function
    resolved_linalg = _semantic(linalg_call).resolved_function

    assert stats_import.source_module_key == "sciarx.stats"
    assert linalg_import.source_module_key == "sciarx.linalg"
    assert resolved_stats is not None
    assert resolved_linalg is not None
    assert resolved_stats.module_key == "sciarx.stats"
    assert resolved_linalg.module_key == "sciarx.linalg"
    assert session.graph["app.main"] == {
        "sciarx",
        "sciarx.linalg",
        "sciarx.stats",
    }


def test_analyze_modules_resolves_mixed_from_import_targets() -> None:
    """
    title: Import-from statements may mix direct symbols and child modules.
    """
    direct_call = astx.FunctionCall(
        "sum2",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    namespace_call = astx.MethodCall(
        astx.Identifier("stats"),
        "sum2",
        [astx.LiteralFloat64(3.0), astx.LiteralFloat64(4.0)],
    )
    import_stmt = astx.ImportFromStmt(
        module="sciarx",
        names=[astx.AliasExpr("sum2"), astx.AliasExpr("stats")],
    )
    root = make_parsed_module(
        "app.main",
        import_stmt,
        _int_function(
            "main",
            direct_call,
            namespace_call,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    sciarx = make_parsed_module("sciarx", _sum2_function())
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    session = analyze_modules(
        root,
        StaticImportResolver(
            {
                "sciarx": sciarx,
                "sciarx.stats": stats_module,
            }
        ),
    )

    direct_import, module_import = _semantic(import_stmt).resolved_imports
    resolved_direct = _semantic(direct_call).resolved_function
    resolved_namespace = _semantic(namespace_call).resolved_function

    assert direct_import.source_module_key == "sciarx"
    assert direct_import.binding.function is not None
    assert module_import.source_module_key == "sciarx.stats"
    assert module_import.binding.module is not None
    assert resolved_direct is not None
    assert resolved_namespace is not None
    assert resolved_direct.module_key == "sciarx"
    assert resolved_namespace.module_key == "sciarx.stats"
    assert session.graph["app.main"] == {"sciarx", "sciarx.stats"}


def test_analyze_modules_prefers_direct_symbol_for_from_import() -> None:
    """
    title: Direct importable symbols take precedence over child modules.
    """
    import_stmt = astx.ImportFromStmt(
        module="sciarx",
        names=[astx.AliasExpr("stats")],
    )
    call = astx.FunctionCall("stats", [])
    root = make_parsed_module(
        "app.main",
        import_stmt,
        _int_function("main", astx.FunctionReturn(call)),
    )
    sciarx = make_parsed_module(
        "sciarx",
        _int_function("stats", astx.FunctionReturn(astx.LiteralInt32(5))),
    )
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    session = analyze_modules(
        root,
        StaticImportResolver(
            {
                "sciarx": sciarx,
                "sciarx.stats": stats_module,
            }
        ),
    )

    resolved_import = _semantic(import_stmt).resolved_imports[0]
    resolved_function = _semantic(call).resolved_function

    assert resolved_import.source_module_key == "sciarx"
    assert resolved_import.binding.function is not None
    assert resolved_import.binding.module is None
    assert resolved_function is not None
    assert resolved_function.module_key == "sciarx"
    assert session.graph == {
        "app.main": {"sciarx"},
        "sciarx": set(),
    }
    assert "sciarx.stats" not in session.modules


def test_analyze_modules_reports_missing_from_import_name() -> None:
    """
    title: Missing import-from names still report the existing diagnostic.
    """
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(
            module="sciarx", names=[astx.AliasExpr("missing")]
        ),
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    sciarx = make_parsed_module("sciarx")

    with pytest.raises(SemanticError) as exc_info:
        analyze_modules(root, StaticImportResolver({"sciarx": sciarx}))

    message = str(exc_info.value)

    assert "Imported symbol 'missing'" in message
    assert "Unable to resolve module 'sciarx.missing'" not in message


def test_analyze_modules_propagates_unexpected_probe_failures() -> None:
    """
    title: Unexpected child-module probe failures still surface directly.
    """
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(module="sciarx", names=[astx.AliasExpr("stats")]),
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    sciarx = make_parsed_module("sciarx")

    def resolver(
        requesting_module_key: str,
        import_node: astx.ImportStmt | astx.ImportFromStmt,
        requested_specifier: str,
    ) -> ParsedModule:
        """
        title: Resolver that sometimes fails on child module lookups.
        parameters:
          requesting_module_key:
            type: str
          import_node:
            type: astx.ImportStmt | astx.ImportFromStmt
          requested_specifier:
            type: str
        returns:
          type: ParsedModule
        """
        _ = requesting_module_key
        _ = import_node
        if requested_specifier == "sciarx":
            return sciarx
        if requested_specifier == "sciarx.stats":
            raise RuntimeError("resolver boom")
        raise LookupError(requested_specifier)

    with pytest.raises(RuntimeError, match="resolver boom"):
        analyze_modules(root, resolver)


def test_analyze_modules_resolves_module_namespace_call() -> None:
    """
    title: Module alias imports resolve callable namespace member access.
    """
    namespace_call = astx.MethodCall(
        astx.Identifier("stats"),
        "sum2",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    import_stmt = astx.ImportStmt(
        [astx.AliasExpr("sciarx.stats", asname="stats")],
    )
    root = make_parsed_module(
        "app.main",
        import_stmt,
        _int_function(
            "main",
            namespace_call,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    analyze_modules(
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )

    resolved_call = _semantic(namespace_call)
    resolved_member = resolved_call.resolved_module_member_access
    resolved_function = resolved_call.resolved_function

    assert resolved_member is not None
    assert resolved_function is not None
    assert resolved_member.module.module_key == "sciarx.stats"
    assert resolved_member.binding.function is resolved_function
    assert resolved_function.module_key == "sciarx.stats"
    assert resolved_call.resolved_call is not None


def test_analyze_modules_resolves_module_namespace_member_reference() -> None:
    """
    title: Bare namespace member access resolves to the exported declaration.
    """
    member_access = astx.FieldAccess(astx.Identifier("stats"), "sum2")
    root = make_parsed_module(
        "app.main",
        astx.ImportStmt([astx.AliasExpr("sciarx.stats", asname="stats")]),
        _int_function(
            "main",
            member_access,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    analyze_modules(
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )

    resolved_member = _semantic(member_access).resolved_module_member_access
    resolved_function = _semantic(member_access).resolved_function

    assert resolved_member is not None
    assert resolved_function is not None
    assert resolved_member.member_name == "sum2"
    assert resolved_member.binding.function is resolved_function
    assert resolved_function.module_key == "sciarx.stats"


def test_analyze_modules_resolves_local_namespace_variable_calls() -> None:
    """
    title: Local variables typed as namespaces preserve module-member lookup.
    """
    namespace_call = astx.MethodCall(
        astx.Identifier("stats_local"),
        "sum2",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    root = make_parsed_module(
        "app.main",
        astx.ImportStmt([astx.AliasExpr("sciarx.stats", asname="stats")]),
        _int_function(
            "main",
            astx.VariableDeclaration(
                name="stats_local",
                type_=astx.NamespaceType("sciarx.stats"),
                value=astx.Identifier("stats"),
            ),
            namespace_call,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    analyze_modules(
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )

    resolved_function = _semantic(namespace_call).resolved_function
    resolved_module = _semantic(namespace_call.receiver).resolved_module

    assert resolved_function is not None
    assert resolved_function.module_key == "sciarx.stats"
    assert resolved_module is not None
    assert resolved_module.module_key == "sciarx.stats"


def test_analyze_modules_resolves_namespace_return_values() -> None:
    """
    title: Function results typed as namespaces remain callable as receivers.
    """
    get_stats_body = astx.Block()
    get_stats_body.append(astx.FunctionReturn(astx.Identifier("stats")))
    namespace_call = astx.MethodCall(
        astx.FunctionCall("get_stats", []),
        "sum2",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    root = make_parsed_module(
        "app.main",
        astx.ImportStmt([astx.AliasExpr("sciarx.stats", asname="stats")]),
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "get_stats",
                args=astx.Arguments(),
                return_type=astx.NamespaceType("sciarx.stats"),
            ),
            body=get_stats_body,
        ),
        _int_function(
            "main",
            namespace_call,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    analyze_modules(
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )

    resolved_receiver_type = _semantic(namespace_call.receiver).resolved_type
    resolved_function = _semantic(namespace_call).resolved_function

    assert isinstance(resolved_receiver_type, astx.NamespaceType)
    assert resolved_receiver_type.namespace_key == "sciarx.stats"
    assert resolved_function is not None
    assert resolved_function.module_key == "sciarx.stats"


def test_analyze_modules_keeps_namespace_and_direct_calls_equivalent() -> None:
    """
    title: Direct imports and namespace calls share the same callable target.
    """
    direct_call = astx.FunctionCall(
        "sum2_direct",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    namespace_call = astx.MethodCall(
        astx.Identifier("stats"),
        "sum2",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(
            module="sciarx.stats",
            names=[astx.AliasExpr("sum2", asname="sum2_direct")],
        ),
        astx.ImportStmt([astx.AliasExpr("sciarx.stats", asname="stats")]),
        _int_function(
            "main",
            direct_call,
            namespace_call,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    analyze_modules(
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )

    direct_function = _semantic(direct_call).resolved_function
    namespace_function = _semantic(namespace_call).resolved_function

    assert direct_function is not None
    assert namespace_function is not None
    assert direct_function is namespace_function


def test_analyze_modules_reports_missing_module_namespace_member() -> None:
    """
    title: Missing namespace members produce a clear semantic diagnostic.
    """
    bad_call = astx.MethodCall(
        astx.Identifier("stats"),
        "does_not_exist",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    root = make_parsed_module(
        "app.main",
        astx.ImportStmt([astx.AliasExpr("sciarx.stats", asname="stats")]),
        _int_function(
            "main",
            bad_call,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    with pytest.raises(
        SemanticError,
        match="module namespace 'stats' has no member 'does_not_exist'",
    ):
        analyze_modules(
            root,
            StaticImportResolver({"sciarx.stats": stats_module}),
        )


def test_analyze_modules_keeps_struct_field_access_behavior() -> None:
    """
    title: Module namespace support does not disturb struct field access.
    """
    field_access = astx.FieldAccess(astx.Identifier("point"), "x")
    root = make_parsed_module(
        "app.main",
        astx.ImportStmt([astx.AliasExpr("sciarx.stats", asname="stats")]),
        _point_struct(),
        _int_function(
            "main",
            astx.VariableDeclaration(
                name="point",
                type_=astx.StructType("Point"),
            ),
            field_access,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    analyze_modules(
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )

    assert _semantic(field_access).resolved_field_access is not None
    assert _semantic(field_access).resolved_module_member_access is None


def test_analyze_modules_returns_dep_order_and_predeclared_imports() -> None:
    """
    title: Dependency order and imported bindings follow the public contract.
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
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(3))),
    )

    session = analyze_modules(root, StaticImportResolver({"lib": lib}))

    imported_function = (
        _semantic(import_stmt).resolved_imports[0].binding.function
    )
    lib_function = session.visible_bindings["lib"]["foo"].function
    resolved_call = _semantic(call).resolved_function

    assert session.load_order == ["lib", "app.main"]
    assert session.graph == {
        "app.main": {"lib"},
        "lib": set(),
    }
    assert imported_function is not None
    assert lib_function is not None
    assert resolved_call is not None
    assert imported_function is lib_function
    assert resolved_call is imported_function


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
    assert resolved_import.binding.struct.module_key == "models"
    assert resolved_import.local_name == "UserPoint"


def test_analyze_modules_resolves_imported_class_binding() -> None:
    """
    title: Imported classes keep the original defining module.
    """
    import_stmt = astx.ImportFromStmt(
        module="models",
        names=[astx.AliasExpr("Shape", asname="UserShape")],
    )
    root = make_parsed_module(
        "app.main",
        import_stmt,
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    models = make_parsed_module("models", _shape_class("Shape"))

    analyze_modules(root, StaticImportResolver({"models": models}))

    resolved_import = _semantic(import_stmt).resolved_imports[0]

    assert resolved_import.binding.class_ is not None
    assert resolved_import.binding.class_.module_key == "models"
    assert resolved_import.local_name == "UserShape"


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


def test_analyze_modules_rejects_import_collisions_with_local_functions() -> (
    None
):
    """
    title: Imported names cannot collide with local module-visible functions.
    """
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(module="lib", names=[astx.AliasExpr("foo")]),
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(0))),
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    lib = make_parsed_module(
        "lib",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(1))),
    )

    with pytest.raises(SemanticError, match="Conflicting binding for 'foo'"):
        analyze_modules(root, StaticImportResolver({"lib": lib}))


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


def test_analyze_modules_keeps_same_bare_struct_names_distinct() -> None:
    """
    title: Same bare struct names remain distinct across modules.
    """
    import_a = astx.ImportFromStmt(
        module="a",
        names=[astx.AliasExpr("Point", asname="APoint")],
    )
    import_b = astx.ImportFromStmt(
        module="b",
        names=[astx.AliasExpr("Point", asname="BPoint")],
    )
    root = make_parsed_module(
        "app.main",
        import_a,
        import_b,
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    module_a = make_parsed_module("a", _point_struct("Point"))
    module_b = make_parsed_module("b", _point_struct("Point"))

    analyze_modules(
        root,
        StaticImportResolver({"a": module_a, "b": module_b}),
    )

    struct_a = _semantic(import_a).resolved_imports[0].binding.struct
    struct_b = _semantic(import_b).resolved_imports[0].binding.struct

    assert struct_a is not None
    assert struct_b is not None
    assert struct_a.module_key == "a"
    assert struct_b.module_key == "b"
    assert struct_a.qualified_name != struct_b.qualified_name


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

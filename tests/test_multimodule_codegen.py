"""
title: Tests for multi-module LLVM lowering.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.module_symbols import (
    mangle_function_name,
    mangle_namespace_name,
    mangle_struct_name,
)
from irx.builder import Builder

from tests.conftest import (
    StaticImportResolver,
    assert_ir_parses,
    make_parsed_module,
    translate_modules_ir,
)


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


def test_translate_modules_mangles_same_named_functions() -> None:
    """
    title: Same bare function names produce distinct LLVM symbols.
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
        _int_function("main", call_a, astx.FunctionReturn(call_b)),
    )
    module_a = make_parsed_module(
        "a",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(1))),
    )
    module_b = make_parsed_module(
        "b",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(2))),
    )

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({"a": module_a, "b": module_b}),
    )

    assert f'define i32 @"{mangle_function_name("a", "foo")}"()' in ir_text
    assert f'define i32 @"{mangle_function_name("b", "foo")}"()' in ir_text


def test_translate_modules_calls_imported_function_by_defining_symbol() -> (
    None
):
    """
    title: Imported calls target the defining module's mangled symbol.
    """
    call = astx.FunctionCall("foo", [])
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(module="lib", names=[astx.AliasExpr("foo")]),
        _int_function("main", astx.FunctionReturn(call)),
    )
    lib = make_parsed_module(
        "lib",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(7))),
    )

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({"lib": lib}),
    )

    assert f'call i32 @"{mangle_function_name("lib", "foo")}"()' in ir_text


def test_translate_modules_calls_function_through_module_namespace() -> None:
    """
    title: Namespace method-call syntax lowers to the defining module symbol.
    """
    namespace_call = astx.MethodCall(
        astx.Identifier("stats"),
        "sum2",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    root = make_parsed_module(
        "app.main",
        astx.ImportStmt([astx.AliasExpr("sciarx.stats", asname="stats")]),
        _int_function(
            "main",
            namespace_call,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )

    call_text = (
        f'call double @"{mangle_function_name("sciarx.stats", "sum2")}"('
    )

    assert call_text in ir_text


def test_translate_modules_lowers_child_module_from_import() -> None:
    """
    title: Import-from child modules lower like explicit namespace imports.
    """
    namespace_call = astx.MethodCall(
        astx.Identifier("stats"),
        "sum2",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(module="sciarx", names=[astx.AliasExpr("stats")]),
        _int_function(
            "main",
            namespace_call,
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    sciarx = make_parsed_module("sciarx")
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver(
            {
                "sciarx": sciarx,
                "sciarx.stats": stats_module,
            }
        ),
    )

    call_text = (
        f'call double @"{mangle_function_name("sciarx.stats", "sum2")}"('
    )

    assert call_text in ir_text


def test_translate_modules_lowers_grouped_child_module_from_imports() -> None:
    """
    title: >-
      Grouped child-module import-from bindings lower to each module symbol.
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
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(
            module="sciarx",
            names=[astx.AliasExpr("stats"), astx.AliasExpr("linalg")],
        ),
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

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver(
            {
                "sciarx": sciarx,
                "sciarx.stats": stats_module,
                "sciarx.linalg": linalg_module,
            }
        ),
    )

    stats_call_text = (
        f'call double @"{mangle_function_name("sciarx.stats", "sum2")}"('
    )
    linalg_call_text = (
        f'call double @"{mangle_function_name("sciarx.linalg", "norm2")}"('
    )

    assert stats_call_text in ir_text
    assert linalg_call_text in ir_text


def test_translate_modules_keeps_direct_and_namespace_calls_equivalent() -> (
    None
):
    """
    title: Direct imports and namespace calls lower to the same callee symbol.
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

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )
    call_text = (
        f'call double @"{mangle_function_name("sciarx.stats", "sum2")}"('
    )
    expected_count = 2

    assert ir_text.count(call_text) == expected_count


def test_translate_modules_lowers_local_namespace_variables() -> None:
    """
    title: Namespace-typed locals lower as reusable opaque namespace handles.
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

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )

    call_text = (
        f'call double @"{mangle_function_name("sciarx.stats", "sum2")}"('
    )
    namespace_global = mangle_namespace_name("sciarx.stats", "module")

    assert call_text in ir_text
    assert f'@"{namespace_global}" = internal constant i8 0' in ir_text


def test_translate_modules_lowers_returned_namespace_values() -> None:
    """
    title: Namespace-typed returns lower as opaque pointer signatures.
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

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )

    get_stats_name = mangle_function_name("app.main", "get_stats")
    sum2_call = (
        f'call double @"{mangle_function_name("sciarx.stats", "sum2")}"('
    )

    assert f'define i8* @"{get_stats_name}"()' in ir_text
    assert f'call i8* @"{get_stats_name}"()' in ir_text
    assert sum2_call in ir_text


def test_translate_modules_lowers_namespace_parameters() -> None:
    """
    title: Namespace values can flow through function parameters.
    """
    namespace_call = astx.MethodCall(
        astx.Identifier("stats_ns"),
        "sum2",
        [astx.LiteralFloat64(1.0), astx.LiteralFloat64(2.0)],
    )
    use_stats_body = astx.Block()
    use_stats_body.append(astx.FunctionReturn(namespace_call))
    use_stats = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "use_stats",
            args=astx.Arguments(
                astx.Argument(
                    "stats_ns",
                    astx.NamespaceType("sciarx.stats"),
                )
            ),
            return_type=astx.Float64(),
        ),
        body=use_stats_body,
    )
    root = make_parsed_module(
        "app.main",
        astx.ImportStmt([astx.AliasExpr("sciarx.stats", asname="stats")]),
        use_stats,
        _int_function(
            "main",
            astx.FunctionCall("use_stats", [astx.Identifier("stats")]),
            astx.FunctionReturn(astx.LiteralInt32(0)),
        ),
    )
    stats_module = make_parsed_module("sciarx.stats", _sum2_function())

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({"sciarx.stats": stats_module}),
    )

    use_stats_name = mangle_function_name("app.main", "use_stats")
    expected_namespace_call_count = 1

    assert f'define double @"{use_stats_name}"(i8* %"stats_ns")' in ir_text
    assert f'call double @"{use_stats_name}"(i8* @"' in ir_text
    assert (
        ir_text.count(
            f'call double @"{mangle_function_name("sciarx.stats", "sum2")}"('
        )
        == expected_namespace_call_count
    )


def test_translate_modules_emits_imported_definition_once() -> None:
    """
    title: Imported function definitions are emitted once across aliases.
    """
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(
            module="lib",
            names=[
                astx.AliasExpr("foo"),
                astx.AliasExpr("foo", asname="alias"),
            ],
        ),
        _int_function(
            "main",
            astx.FunctionCall("foo", []),
            astx.FunctionReturn(astx.FunctionCall("alias", [])),
        ),
    )
    lib = make_parsed_module(
        "lib",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(5))),
    )

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({"lib": lib}),
    )
    mangled_name = mangle_function_name("lib", "foo")

    assert ir_text.count(f'define i32 @"{mangled_name}"()') == 1


def test_translate_modules_mangles_same_named_structs() -> None:
    """
    title: Same bare struct names produce distinct LLVM identified types.
    """
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(
            module="a",
            names=[astx.AliasExpr("Point", asname="APoint")],
        ),
        astx.ImportFromStmt(
            module="b",
            names=[astx.AliasExpr("Point", asname="BPoint")],
        ),
        _int_function("main", astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    module_a = make_parsed_module("a", _point_struct("Point"))
    module_b = make_parsed_module("b", _point_struct("Point"))

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({"a": module_a, "b": module_b}),
    )

    assert f'%"{mangle_struct_name("a", "Point")}" = type' in ir_text
    assert f'%"{mangle_struct_name("b", "Point")}" = type' in ir_text


def test_translate_modules_emits_parseable_llvm_ir() -> None:
    """
    title: Multi-module translation still emits parseable LLVM IR.
    """
    root = make_parsed_module(
        "app.main",
        astx.ImportFromStmt(module="lib", names=[astx.AliasExpr("foo")]),
        _int_function(
            "main", astx.FunctionReturn(astx.FunctionCall("foo", []))
        ),
    )
    lib = make_parsed_module(
        "lib",
        _int_function("foo", astx.FunctionReturn(astx.LiteralInt32(3))),
    )

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({"lib": lib}),
    )

    assert_ir_parses(ir_text)

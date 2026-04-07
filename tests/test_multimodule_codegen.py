"""
title: Tests for multi-module LLVM lowering.
"""

from __future__ import annotations

from irx import astx
from irx.analysis import ModuleKey
from irx.analysis.module_symbols import (
    mangle_function_name,
    mangle_struct_name,
)
from irx.builders.llvmliteir import Builder

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

    assert (
        f'define i32 @"{mangle_function_name(ModuleKey("a"), "foo")}"()'
        in ir_text
    )
    assert (
        f'define i32 @"{mangle_function_name(ModuleKey("b"), "foo")}"()'
        in ir_text
    )


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

    assert (
        f'call i32 @"{mangle_function_name(ModuleKey("lib"), "foo")}"()'
        in ir_text
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
    mangled_name = mangle_function_name(ModuleKey("lib"), "foo")

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

    assert (
        f'%"{mangle_struct_name(ModuleKey("a"), "Point")}" = type' in ir_text
    )
    assert (
        f'%"{mangle_struct_name(ModuleKey("b"), "Point")}" = type' in ir_text
    )


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

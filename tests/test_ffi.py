"""
title: Tests for the public FFI lowering and build contract.
"""

from __future__ import annotations

import shutil

from pathlib import Path

import pytest

from irx import astx
from irx.builder import Builder

from tests.conftest import assert_ir_parses

HAS_CLANG = shutil.which("clang") is not None
EXPECTED_SQRT_EXIT_CODE = 4


def _extern_prototype(
    name: str,
    *args: astx.Argument,
    return_type: astx.DataType,
    symbol_name: str | None = None,
    runtime_feature: str | None = None,
) -> astx.FunctionPrototype:
    """
    title: Build one explicit extern prototype.
    parameters:
      name:
        type: str
      return_type:
        type: astx.DataType
      symbol_name:
        type: str | None
      runtime_feature:
        type: str | None
      args:
        type: astx.Argument
        variadic: positional
    returns:
      type: astx.FunctionPrototype
    """
    prototype = astx.FunctionPrototype(
        name,
        args=astx.Arguments(*args),
        return_type=return_type,
    )
    prototype.is_extern = True
    prototype.calling_convention = "c"
    prototype.symbol_name = symbol_name or name
    if runtime_feature is not None:
        prototype.runtime_feature = runtime_feature
    return prototype


def _main_module(*body_nodes: astx.AST) -> astx.Module:
    """
    title: Build one small Int32 main module.
    parameters:
      body_nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    module = astx.Module()
    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    for node in body_nodes:
        main.body.append(node)
    if not any(isinstance(node, astx.FunctionReturn) for node in body_nodes):
        main.body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(main)
    return module


def test_plain_extern_symbol_does_not_activate_runtime_feature() -> None:
    """
    title: >-
      Plain extern declarations should rely on the system linker by default.
    """
    builder = Builder()
    module = _main_module(
        astx.FunctionCall("puts", [astx.LiteralUTF8String("hello")]),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )
    module.block.insert(
        0,
        _extern_prototype(
            "puts",
            astx.Argument("message", astx.UTF8String()),
            return_type=astx.Int32(),
        ),
    )

    builder.translate(module)

    assert builder.translator.runtime_features.active_feature_names() == ()


def test_translate_feature_backed_extern_activates_runtime_feature() -> None:
    """
    title: Feature-backed externs should activate the declared runtime feature.
    """
    builder = Builder()
    module = _main_module(
        astx.FunctionReturn(
            astx.Cast(
                astx.FunctionCall("sqrt", [astx.LiteralFloat64(16.0)]),
                astx.Int32(),
            )
        )
    )
    module.block.insert(
        0,
        _extern_prototype(
            "sqrt",
            astx.Argument("value", astx.Float64()),
            return_type=astx.Float64(),
            runtime_feature="libm",
        ),
    )

    ir_text = builder.translate(module)

    assert builder.translator.runtime_features.active_feature_names() == (
        "libm",
    )
    assert 'declare external double @"sqrt"(double %"value")' in ir_text
    assert_ir_parses(ir_text)


def test_translate_pointer_and_opaque_handle_externs() -> None:
    """
    title: >-
      Pointer and opaque-handle externs should lower through one public path.
    """
    builder = Builder()
    module = _main_module(
        astx.VariableDeclaration(
            name="handle",
            type_=astx.OpaqueHandleType("demo_handle"),
            mutability=astx.MutabilityKind.mutable,
            value=astx.FunctionCall("create_handle", []),
        ),
        astx.FunctionCall(
            "consume_handle",
            [astx.Identifier("handle")],
        ),
        astx.VariableDeclaration(
            name="values",
            type_=astx.PointerType(astx.Float64()),
            mutability=astx.MutabilityKind.mutable,
            value=astx.FunctionCall("alloc_values", []),
        ),
        astx.FunctionReturn(
            astx.FunctionCall("consume_values", [astx.Identifier("values")])
        ),
    )
    module.block.insert(
        0,
        _extern_prototype(
            "consume_values",
            astx.Argument("values", astx.PointerType(astx.Float64())),
            return_type=astx.Int32(),
        ),
    )
    module.block.insert(
        0,
        _extern_prototype(
            "alloc_values",
            return_type=astx.PointerType(astx.Float64()),
        ),
    )
    module.block.insert(
        0,
        _extern_prototype(
            "consume_handle",
            astx.Argument("handle", astx.OpaqueHandleType("demo_handle")),
            return_type=astx.Int32(),
        ),
    )
    module.block.insert(
        0,
        _extern_prototype(
            "create_handle",
            return_type=astx.OpaqueHandleType("demo_handle"),
        ),
    )

    ir_text = builder.translate(module)

    assert 'declare external i8* @"create_handle"()' in ir_text
    assert 'declare external i32 @"consume_handle"(i8* %"handle")' in ir_text
    assert 'declare external double* @"alloc_values"()' in ir_text
    assert (
        'declare external i32 @"consume_values"(double* %"values")'
    ) in ir_text
    assert_ir_parses(ir_text)


def test_translate_struct_extern_by_value() -> None:
    """
    title: ABI-safe structs should lower by value across extern boundaries.
    """
    builder = Builder()
    point = astx.StructDefStmt(
        name="Point",
        attributes=[
            astx.VariableDeclaration(name="x", type_=astx.Int32()),
            astx.VariableDeclaration(name="y", type_=astx.Int32()),
        ],
    )
    module = _main_module(
        astx.VariableDeclaration(
            name="point",
            type_=astx.StructType("Point"),
            mutability=astx.MutabilityKind.mutable,
        ),
        astx.FunctionReturn(
            astx.FunctionCall("read_x", [astx.Identifier("point")])
        ),
    )
    module.block.insert(0, point)
    module.block.insert(
        1,
        _extern_prototype(
            "read_x",
            astx.Argument("point", astx.StructType("Point")),
            return_type=astx.Int32(),
        ),
    )

    ir_text = builder.translate(module)

    assert (
        'declare external i32 @"read_x"(%"main__Point" %"point")'
    ) in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.skipif(not HAS_CLANG, reason="clang is not available")
def test_build_feature_backed_libm_extern_call(tmp_path: Path) -> None:
    """
    title: >-
      Feature-backed externs should link with their runtime feature inputs.
    parameters:
      tmp_path:
        type: Path
    """
    builder = Builder()
    module = _main_module(
        astx.FunctionReturn(
            astx.Cast(
                astx.FunctionCall("sqrt", [astx.LiteralFloat64(16.0)]),
                astx.Int32(),
            )
        )
    )
    module.block.insert(
        0,
        _extern_prototype(
            "sqrt",
            astx.Argument("value", astx.Float64()),
            return_type=astx.Float64(),
            runtime_feature="libm",
        ),
    )

    output_file = tmp_path / "ffi_sqrt"
    builder.build(module, output_file=str(output_file))
    result = builder.run(raise_on_error=False)

    assert result.returncode == EXPECTED_SQRT_EXIT_CODE

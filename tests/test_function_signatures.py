"""
title: Tests for hardened function lowering and integration behavior.
"""

from __future__ import annotations

import shutil

from pathlib import Path

import pytest

from irx import astx
from irx.builder import Builder

from .conftest import assert_ir_parses

HAS_CLANG = shutil.which("clang") is not None
EXPECTED_PUTS_CALLS = 2
EXPECTED_HELPER_EXIT = 7


def _extern_prototype(
    name: str,
    *args: astx.Argument,
    return_type: astx.DataType,
    symbol_name: str | None = None,
    is_variadic: bool = False,
) -> astx.FunctionPrototype:
    """
    title: Build one extern prototype with the hardened semantic attributes.
    parameters:
      name:
        type: str
      return_type:
        type: astx.DataType
      symbol_name:
        type: str | None
      is_variadic:
        type: bool
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
    prototype.is_variadic = is_variadic
    return prototype


def test_uses_canonical_function_signature_for_definition_and_call() -> None:
    """
    title: Function declarations and calls should reuse the semantic signature.
    """
    module = astx.Module()
    helper = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(astx.Argument("value", astx.Int32())),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    helper.body.append(astx.FunctionReturn(astx.Identifier("value")))
    module.block.append(helper)

    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(
        astx.FunctionReturn(
            astx.FunctionCall("helper", [astx.LiteralInt16(7)])
        )
    )
    module.block.append(main)

    ir_text = Builder().translate(module)

    assert 'define i32 @"main__helper"(i32 %"value")' in ir_text
    assert 'call i32 @"main__helper"(i32' in ir_text
    assert ir_text.count('define i32 @"main__helper"') == 1
    assert_ir_parses(ir_text)


def test_translate_emits_ret_void_and_typed_ret() -> None:
    """
    title: >-
      Void and non-void returns should lower through semantic return metadata.
    """
    module = astx.Module()
    noop = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "noop",
            args=astx.Arguments(),
            return_type=astx.NoneType(),
        ),
        body=astx.Block(),
    )
    noop.body.append(astx.FunctionReturn(astx.LiteralNone()))
    module.block.append(noop)

    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(astx.FunctionCall("noop", []))
    main.body.append(astx.FunctionReturn(astx.LiteralInt16(7)))
    module.block.append(main)

    ir_text = Builder().translate(module)

    assert "ret void" in ir_text
    assert "sext i16 7 to i32" in ir_text
    assert "ret i32" in ir_text
    assert_ir_parses(ir_text)


def test_translate_emits_extern_declaration_once_with_symbol_name() -> None:
    """
    title: Extern declarations should reuse one stable LLVM declaration.
    """
    module = astx.Module()
    module.block.append(
        _extern_prototype(
            "puts",
            astx.Argument("message", astx.UTF8String()),
            return_type=astx.Int32(),
            symbol_name="puts",
        )
    )

    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(
        astx.FunctionCall("puts", [astx.LiteralUTF8String("one")])
    )
    main.body.append(
        astx.FunctionCall("puts", [astx.LiteralUTF8String("two")])
    )
    main.body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(main)

    ir_text = Builder().translate(module)

    assert 'declare external i32 @"puts"(i8* %"message")' in ir_text
    assert ir_text.count('declare external i32 @"puts"') == 1
    assert ir_text.count('call i32 @"puts"') == EXPECTED_PUTS_CALLS
    assert_ir_parses(ir_text)


def test_translate_supports_narrow_extern_varargs_with_c_promotions() -> None:
    """
    title: Extern varargs should lower with C-style default promotions.
    """
    module = astx.Module()
    module.block.append(
        _extern_prototype(
            "printf",
            astx.Argument("format", astx.UTF8String()),
            return_type=astx.Int32(),
            symbol_name="printf",
            is_variadic=True,
        )
    )

    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(
        astx.FunctionCall(
            "printf",
            [
                astx.LiteralUTF8String("%d %.1f"),
                astx.LiteralInt16(7),
                astx.LiteralFloat32(1.5),
            ],
        )
    )
    main.body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(main)

    ir_text = Builder().translate(module)

    assert 'declare external i32 @"printf"(i8* %"format", ...)' in ir_text
    assert 'call i32 (i8*, ...) @"printf"' in ir_text
    assert "sext i16 7 to i32" in ir_text
    assert "fpext float" in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.skipif(not HAS_CLANG, reason="clang is not available")
def test_build_helper_function_called_from_main(tmp_path: Path) -> None:
    """
    title: A helper call should produce a deterministic Int32 process exit.
    parameters:
      tmp_path:
        type: Path
    """
    builder = Builder()
    module = astx.Module()
    helper = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    helper.body.append(astx.FunctionReturn(astx.LiteralInt32(7)))
    module.block.append(helper)

    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(astx.FunctionReturn(astx.FunctionCall("helper", [])))
    module.block.append(main)

    output_file = tmp_path / "helper_main"
    builder.build(module, output_file=str(output_file))
    result = builder.run(raise_on_error=False)

    assert result.returncode == EXPECTED_HELPER_EXIT


@pytest.mark.skipif(not HAS_CLANG, reason="clang is not available")
def test_build_extern_puts_call(tmp_path: Path) -> None:
    """
    title: Extern calls should build and run through the hardened C path.
    parameters:
      tmp_path:
        type: Path
    """
    builder = Builder()
    module = astx.Module()
    module.block.append(
        _extern_prototype(
            "puts",
            astx.Argument("message", astx.UTF8String()),
            return_type=astx.Int32(),
            symbol_name="puts",
        )
    )

    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(
        astx.FunctionCall("puts", [astx.LiteralUTF8String("hello")])
    )
    main.body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(main)

    output_file = tmp_path / "extern_puts"
    builder.build(module, output_file=str(output_file))
    result = builder.run(raise_on_error=False)

    assert result.returncode == 0
    assert result.stdout.strip() == "hello"

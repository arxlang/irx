"""
title: Tests for scalar numeric lowering and codegen.
"""

from __future__ import annotations

from irx import astx
from irx.builder import Builder
from irx.system import Cast, PrintExpr

from .conftest import assert_ir_parses, make_main_module


def test_translate_implicit_integer_widening_for_variable_initializers() -> (
    None
):
    """
    title: Variable initializers should widen before storing into the alloca.
    """
    module = make_main_module(
        astx.VariableDeclaration(
            name="value",
            type_=astx.Int32(),
            value=astx.LiteralInt16(7),
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    ir_text = Builder().translate(module)
    assert "sext i16 7 to i32" in ir_text
    assert_ir_parses(ir_text)


def test_translate_mixed_int64_and_float32_promotes_to_double() -> None:
    """
    title: Mixed int64 and float32 arithmetic should lower through double.
    """
    module = make_main_module(
        astx.VariableDeclaration(
            name="left",
            type_=astx.Int64(),
            value=astx.LiteralInt64(1),
        ),
        astx.VariableDeclaration(
            name="right",
            type_=astx.Float32(),
            value=astx.LiteralFloat32(2.0),
        ),
        astx.InlineVariableDeclaration(
            name="result",
            type_=astx.Float64(),
            value=astx.BinaryOp(
                "+",
                astx.Identifier("left"),
                astx.Identifier("right"),
            ),
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    ir_text = Builder().translate(module)

    assert "sitofp i64" in ir_text
    assert "fpext float" in ir_text
    assert "fadd double" in ir_text
    assert_ir_parses(ir_text)


def test_translate_implicit_integer_widening_for_reassignment() -> None:
    """
    title: Reassignments should widen values before storing them.
    """
    module = make_main_module(
        astx.VariableDeclaration(
            name="value",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        astx.VariableAssignment(
            name="value",
            value=astx.LiteralInt16(7),
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    ir_text = Builder().translate(module)

    assert "sext i16 7 to i32" in ir_text
    assert_ir_parses(ir_text)


def test_translate_function_call_implicitly_promotes_arguments() -> None:
    """
    title: >-
      Calls should apply the same implicit promotions as semantic analysis.
    """
    module = astx.Module()

    echo_proto = astx.FunctionPrototype(
        "echo",
        args=astx.Arguments(astx.Argument("value", astx.Int32())),
        return_type=astx.Int32(),
    )
    echo_body = astx.Block()
    echo_body.append(astx.FunctionReturn(astx.Identifier("value")))
    module.block.append(astx.FunctionDef(prototype=echo_proto, body=echo_body))

    main_proto = astx.FunctionPrototype(
        "main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_body = astx.Block()
    main_body.append(
        PrintExpr(astx.FunctionCall("echo", [astx.LiteralInt16(7)]))
    )
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=main_body))

    ir_text = Builder().translate(module)

    assert "sext i16 7 to i32" in ir_text
    assert 'call i32 @"main__echo"(i32' in ir_text
    assert_ir_parses(ir_text)


def test_translate_implicit_return_coercion_uses_declared_function_type() -> (
    None
):
    """
    title: Returns should coerce to the declared function return type.
    """
    module = astx.Module()
    proto = astx.FunctionPrototype(
        "main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    body = astx.Block()
    body.append(astx.FunctionReturn(astx.LiteralInt16(7)))
    module.block.append(astx.FunctionDef(prototype=proto, body=body))

    ir_text = Builder().translate(module)

    assert "sext i16 7 to i32" in ir_text
    assert "ret i32" in ir_text
    assert_ir_parses(ir_text)


def test_mixed_signed_and_unsigned_comparison_uses_canonical_promotion() -> (
    None
):
    """
    title: Mixed signed and unsigned comparisons should follow the shared rule.
    """
    module = make_main_module(
        astx.FunctionReturn(
            astx.BinaryOp(
                ">",
                astx.LiteralInt16(-1),
                astx.LiteralUInt32(1),
            )
        ),
        return_type=astx.Boolean(),
    )

    ir_text = Builder().translate(module)

    assert "sext i16 -1 to i32" in ir_text
    assert "icmp ugt i32" in ir_text
    assert_ir_parses(ir_text)


def test_wider_signed_integer_keeps_signed_comparison_semantics() -> None:
    """
    title: Wider signed integers should force signed comparison lowering.
    """
    module = make_main_module(
        astx.FunctionReturn(
            astx.BinaryOp(
                ">",
                astx.LiteralInt64(-1),
                astx.LiteralUInt32(1),
            )
        ),
        return_type=astx.Boolean(),
    )

    ir_text = Builder().translate(module)

    assert "zext i32 1 to i64" in ir_text
    assert "icmp sgt i64" in ir_text
    assert_ir_parses(ir_text)


def test_translate_bool_numeric_casts_use_truthy_zero_one_semantics() -> None:
    """
    title: Bool casts should use 0/1 truthiness instead of sign-extension.
    """
    module = make_main_module(
        astx.VariableDeclaration(
            name="flag",
            type_=astx.Boolean(),
            value=astx.LiteralBoolean(True),
        ),
        astx.VariableDeclaration(
            name="number",
            type_=astx.Int32(),
            value=astx.LiteralInt32(2),
        ),
        PrintExpr(
            Cast(
                value=astx.Identifier("flag"),
                target_type=astx.Int32(),
            )
        ),
        PrintExpr(
            Cast(
                value=astx.Identifier("number"),
                target_type=astx.Boolean(),
            )
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    ir_text = Builder().translate(module)

    assert "zext i1" in ir_text
    assert "icmp ne i32" in ir_text
    assert_ir_parses(ir_text)


def test_translate_cast_uint32_to_string_formats_as_unsigned() -> None:
    """
    title: Unsigned integer string casts should preserve the unsigned value.
    """
    module = make_main_module(
        PrintExpr(
            Cast(
                value=astx.LiteralUInt32(4294967295),
                target_type=astx.String(),
            )
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    ir_text = Builder().translate(module)

    assert "%llu" in ir_text
    assert_ir_parses(ir_text)

"""
title: Tests for numeric and expression-based PrintExpr lowering.
"""

import astx

from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr
from llvmlite import binding as llvm


def _translate_and_validate(module: astx.Module) -> str:
    builder = LLVMLiteIR()
    ir_text = builder.translate(module)
    llvm.parse_assembly(ir_text)
    return ir_text


def _assert_puts_uses_char_ptr(ir_text: str) -> None:
    assert 'call i32 @"puts"(i8*' in ir_text
    assert 'call i32 @"puts"(i32' not in ir_text
    assert 'call i32 @"puts"(float' not in ir_text
    assert 'call i32 @"puts"(double' not in ir_text


def test_print_integer_literal_codegen() -> None:
    """
    title: PrintExpr should format integer literals through snprintf + puts.
    """
    module = astx.Module()

    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_body = astx.Block()
    main_body.append(PrintExpr(astx.LiteralInt32(42)))
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=main_body))

    ir_text = _translate_and_validate(module)
    assert "%lld" in ir_text
    assert '@"snprintf"' in ir_text
    _assert_puts_uses_char_ptr(ir_text)


def test_print_float_literal_codegen() -> None:
    """
    title: PrintExpr should format float literals through snprintf + puts.
    """
    module = astx.Module()

    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_body = astx.Block()
    main_body.append(PrintExpr(astx.LiteralFloat32(3.5)))
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=main_body))

    ir_text = _translate_and_validate(module)
    assert "%.6f" in ir_text
    assert '@"snprintf"' in ir_text
    _assert_puts_uses_char_ptr(ir_text)


def test_print_recursive_function_call_result_codegen() -> None:
    """
    title: PrintExpr should support integer results from recursive calls.
    """
    module = astx.Module()

    fib_proto = astx.FunctionPrototype(
        "fib",
        args=astx.Arguments(astx.Argument("x", astx.Int32())),
        return_type=astx.Int32(),
    )
    fib_body = astx.Block()
    fib_then = astx.Block()
    fib_then.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    fib_else = astx.Block()
    fib_sum = astx.BinaryOp(
        op_code="+",
        lhs=astx.FunctionCall(
            "fib",
            [
                astx.BinaryOp(
                    op_code="-",
                    lhs=astx.Identifier("x"),
                    rhs=astx.LiteralInt32(1),
                )
            ],
        ),
        rhs=astx.FunctionCall(
            "fib",
            [
                astx.BinaryOp(
                    op_code="-",
                    lhs=astx.Identifier("x"),
                    rhs=astx.LiteralInt32(2),
                )
            ],
        ),
    )
    fib_else.append(astx.FunctionReturn(fib_sum))
    fib_cond = astx.BinaryOp(
        op_code="<",
        lhs=astx.Identifier("x"),
        rhs=astx.LiteralInt32(3),
    )
    fib_body.append(
        astx.IfStmt(condition=fib_cond, then=fib_then, else_=fib_else)
    )
    module.block.append(astx.FunctionDef(prototype=fib_proto, body=fib_body))

    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_body = astx.Block()
    main_body.append(
        PrintExpr(astx.FunctionCall("fib", [astx.LiteralInt32(10)]))
    )
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=main_body))

    ir_text = _translate_and_validate(module)
    assert 'define i32 @"fib"' in ir_text
    assert 'call i32 @"fib"(i32 10)' in ir_text
    assert "%lld" in ir_text
    _assert_puts_uses_char_ptr(ir_text)


def test_print_float_function_call_result_codegen() -> None:
    """
    title: PrintExpr should support floating-point function call results.
    """
    module = astx.Module()

    avg_proto = astx.FunctionPrototype(
        "average",
        args=astx.Arguments(
            astx.Argument("a", astx.Float32()),
            astx.Argument("b", astx.Float32()),
        ),
        return_type=astx.Float32(),
    )
    avg_body = astx.Block()
    avg_sum = astx.BinaryOp(
        op_code="+",
        lhs=astx.Identifier("a"),
        rhs=astx.Identifier("b"),
    )
    avg_value = astx.BinaryOp(
        op_code="/",
        lhs=avg_sum,
        rhs=astx.LiteralFloat32(2.0),
    )
    avg_body.append(astx.FunctionReturn(avg_value))
    module.block.append(astx.FunctionDef(prototype=avg_proto, body=avg_body))

    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_body = astx.Block()
    main_body.append(
        PrintExpr(
            astx.FunctionCall(
                "average",
                [astx.LiteralFloat32(10.0), astx.LiteralFloat32(20.0)],
            )
        )
    )
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=main_body))

    ir_text = _translate_and_validate(module)
    assert 'define float @"average"' in ir_text
    assert 'call float @"average"' in ir_text
    assert "%.6f" in ir_text
    _assert_puts_uses_char_ptr(ir_text)

from irx.system import PrintExpr
from .conftest import check_result


def test_print_integer() -> None:
    """Test PrintExpr with integer value (line 2744)."""
    builder = LLVMLiteIR()
    module = builder.module()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(PrintExpr(astx.LiteralInt32(42)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="42")


def test_print_float() -> None:
    """Test PrintExpr with float value (line 2751-2758)."""
    builder = LLVMLiteIR()
    module = builder.module()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(PrintExpr(astx.LiteralFloat32(3.14)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)


def test_format_global_reuse() -> None:
    """Test _get_or_create_format_global reuse (line 2613)."""
    builder = LLVMLiteIR()
    module = builder.module()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(PrintExpr(astx.LiteralInt32(1)))
    block.append(PrintExpr(astx.LiteralInt32(2)))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module)

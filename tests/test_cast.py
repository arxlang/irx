"""
title: Tests for the Casting.
"""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import Cast, PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type_from, literal_type_from, int_type_to",
    [
        (astx.Int8, astx.LiteralInt8, astx.Int32),
        (astx.Int32, astx.LiteralInt32, astx.Int16),
        (astx.Int16, astx.LiteralInt16, astx.Int8),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_cast_basic.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_cast_basic(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type_from: type,
    literal_type_from: type,
    int_type_to: type,
) -> None:
    """
    title: Test casting int types between different widths.
    parameters:
      action:
        type: str
      expected_file:
        type: str
      builder_class:
        type: Type[Builder]
      int_type_from:
        type: type
      literal_type_from:
        type: type
      int_type_to:
        type: type
    """
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a", type_=int_type_from(), value=literal_type_from(42)
    )
    a = astx.Identifier("a")
    cast_expr = Cast(value=a, target_type=int_type_to())

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type_to()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    cast_var = astx.InlineVariableDeclaration(
        name="r", type_=int_type_to(), value=cast_expr
    )
    main_block.append(cast_var)
    main_block.append(astx.FunctionReturn(astx.Identifier("r")))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    expected_output = "42"
    check_result("build", builder, module, expected_output=expected_output)


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_cast_int_to_float_and_back(builder_class: Type[Builder]) -> None:
    """
    title: Test casting int -> float -> int, returning int result.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    # a: i32 = 42
    decl_a = astx.VariableDeclaration(
        name="a", type_=astx.Int32(), value=astx.LiteralInt32(42)
    )
    a_ident = astx.Identifier("a")

    # r: float32 = cast(a)
    cast_to_float = Cast(value=a_ident, target_type=astx.Float32())
    cast_var_float = astx.InlineVariableDeclaration(
        name="r", type_=astx.Float32(), value=cast_to_float
    )

    # s: int32 = cast(r)
    r_ident = astx.Identifier("r")
    cast_back_to_int = Cast(value=r_ident, target_type=astx.Int32())
    cast_var_int = astx.InlineVariableDeclaration(
        name="s", type_=astx.Int32(), value=cast_back_to_int
    )

    # main returns int32
    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(cast_var_float)
    main_block.append(cast_var_int)
    main_block.append(astx.FunctionReturn(astx.Identifier("s")))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    # expected output: program exit code "42"
    expected_output = "42"
    check_result("build", builder, module, expected_output=expected_output)


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_cast_int_to_string(builder_class: Type[Builder]) -> None:
    """
    title: Cast an integer to a string, print it, and return 0.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    # a: i32 = 42
    decl_a = astx.VariableDeclaration(
        name="a", type_=astx.Int32(), value=astx.LiteralInt32(42)
    )
    a_ident = astx.Identifier("a")

    # r: string = cast(a)
    cast_to_str = Cast(value=a_ident, target_type=astx.String())
    cast_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.String(), value=cast_to_str
    )

    # print(r)
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    # main returns int32 (exit code 0)
    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(cast_var)
    main_block.append(print_stmt)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    expected_output = "42"
    check_result("build", builder, module, expected_output=expected_output)


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_cast_float_to_string(builder_class: Type[Builder]) -> None:
    """
    title: Cast a float to a string, print it, and return 0.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    # a: float32 = 42.0
    decl_a = astx.VariableDeclaration(
        name="a", type_=astx.Float32(), value=astx.LiteralFloat32(42.0)
    )
    a_ident = astx.Identifier("a")

    # r: string = cast(a)
    cast_to_str = Cast(value=a_ident, target_type=astx.String())
    cast_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.String(), value=cast_to_str
    )

    # print(r)
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    # main returns int32 (exit code 0)
    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(cast_var)
    main_block.append(print_stmt)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    expected_output = "42.000000"
    check_result("build", builder, module, expected_output=expected_output)


@pytest.mark.parametrize(
    "boolean_value, expected_output",
    [
        (True, "1"),
        (False, "0"),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_cast_boolean_to_string(
    builder_class: Type[Builder],
    boolean_value: bool,
    expected_output: str,
) -> None:
    """
    title: Cast a boolean to a string, verify it prints as 1/0 not -1/0.
    parameters:
      builder_class:
        type: Type[Builder]
      boolean_value:
        type: bool
      expected_output:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    # a: boolean = True/False
    decl_a = astx.VariableDeclaration(
        name="a",
        type_=astx.Boolean(),
        value=astx.LiteralBoolean(boolean_value),
    )
    a_ident = astx.Identifier("a")

    # r: string = cast(a)
    cast_to_str = Cast(value=a_ident, target_type=astx.String())
    cast_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.String(), value=cast_to_str
    )

    # print(r)
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    # main returns int32 (exit code 0)
    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(cast_var)
    main_block.append(print_stmt)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result("build", builder, module, expected_output=expected_output)


def test_cast_int_to_float() -> None:
    """
    title: Test Cast from int to float (line 2654).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(42),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("x"), target_type=astx.Float32())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)

    cast_var = astx.InlineVariableDeclaration(
        name="cast_res", type_=astx.Float32(), value=cast_expr
    )

    cast_to_str = Cast(
        value=astx.Identifier("cast_res"), target_type=astx.String()
    )
    str_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.String(), value=cast_to_str
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    block.append(cast_var)
    block.append(str_var)
    block.append(print_stmt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="42.000000")


def test_cast_float_to_int() -> None:
    """
    title: Test Cast from float to int (line 2659).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="f",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.14),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("f"), target_type=astx.Int32())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)

    cast_var = astx.InlineVariableDeclaration(
        name="cast_res", type_=astx.Int32(), value=cast_expr
    )

    cast_to_str = Cast(
        value=astx.Identifier("cast_res"), target_type=astx.String()
    )
    str_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.String(), value=cast_to_str
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    block.append(cast_var)
    block.append(str_var)
    block.append(print_stmt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="3")


def test_cast_int_widening() -> None:
    """
    title: Test Cast from int8 to int32 (line 2646).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int8(),
        value=astx.LiteralInt8(10),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("x"), target_type=astx.Int32())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)

    cast_var = astx.InlineVariableDeclaration(
        name="cast_res", type_=astx.Int32(), value=cast_expr
    )

    cast_to_str = Cast(
        value=astx.Identifier("cast_res"), target_type=astx.String()
    )
    str_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.String(), value=cast_to_str
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    block.append(cast_var)
    block.append(str_var)
    block.append(print_stmt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="10")


def test_cast_int_narrowing() -> None:
    """
    title: Test Cast from int32 to int8 (line 2650).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(10),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("x"), target_type=astx.Int8())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)

    cast_var = astx.InlineVariableDeclaration(
        name="cast_res", type_=astx.Int8(), value=cast_expr
    )

    cast_to_str = Cast(
        value=astx.Identifier("cast_res"), target_type=astx.String()
    )
    str_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.String(), value=cast_to_str
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    block.append(cast_var)
    block.append(str_var)
    block.append(print_stmt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="10")


def test_cast_same_type_noop() -> None:
    """
    title: Test Cast with same source and target type is a no-op (line 2639).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(5),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("x"), target_type=astx.Int32())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)

    cast_var = astx.InlineVariableDeclaration(
        name="cast_res", type_=astx.Int32(), value=cast_expr
    )

    cast_to_str = Cast(
        value=astx.Identifier("cast_res"), target_type=astx.String()
    )
    str_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.String(), value=cast_to_str
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    block.append(cast_var)
    block.append(str_var)
    block.append(print_stmt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="5")


def test_cast_float_to_half() -> None:
    """
    title: Test Cast from float32 to float16 (line 2666).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="f",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(1.5),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("f"), target_type=astx.Float16())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)

    cast_var = astx.InlineVariableDeclaration(
        name="cast_res", type_=astx.Float16(), value=cast_expr
    )

    cast_to_str = Cast(
        value=astx.Identifier("cast_res"), target_type=astx.String()
    )
    str_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.String(), value=cast_to_str
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    block.append(cast_var)
    block.append(str_var)
    block.append(print_stmt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1.500000")


def test_cast_half_to_float() -> None:
    """
    title: Test Cast from float16 to float32 (line 2673).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="h",
        type_=astx.Float16(),
        value=astx.LiteralFloat16(1.5),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("h"), target_type=astx.Float32())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)

    cast_var = astx.InlineVariableDeclaration(
        name="cast_res", type_=astx.Float32(), value=cast_expr
    )

    cast_to_str = Cast(
        value=astx.Identifier("cast_res"), target_type=astx.String()
    )
    str_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.String(), value=cast_to_str
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    block.append(cast_var)
    block.append(str_var)
    block.append(print_stmt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1.500000")


def test_cast_int_to_string_extra() -> None:
    """
    title: Test Cast from int to string (line 2694+).
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(42),
        mutability=astx.MutabilityKind.mutable,
    )
    cast_expr = Cast(value=astx.Identifier("x"), target_type=astx.String())

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)

    cast_var = astx.InlineVariableDeclaration(
        name="cast_res", type_=astx.String(), value=cast_expr
    )

    print_stmt = PrintExpr(message=astx.Identifier("cast_res"))
    block.append(cast_var)
    block.append(print_stmt)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="42")

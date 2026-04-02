"""
title: Tests for the Casting.
"""

import pytest

from irx import astx
from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder
from irx.system import Cast, PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type_from, literal_type_from, value, int_type_to, expected_output",
    [
        # widening: 42 fits fine, result is still "42"
        (astx.Int8, astx.LiteralInt8, 42, astx.Int32, "42"),
        # narrowing with truncation: 300 -> i8 = 44 (300 % 256)
        (astx.Int32, astx.LiteralInt32, 300, astx.Int8, "44"),
        # narrowing that fits: 42 -> i8 still "42"
        (astx.Int16, astx.LiteralInt16, 42, astx.Int8, "42"),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_cast_basic(
    builder_class: type[Builder],
    int_type_from: type,
    literal_type_from: type,
    value: int,
    int_type_to: type,
    expected_output: str,
) -> None:
    """
    title: Test casting int types between different widths.
    parameters:
      builder_class:
        type: type[Builder]
      int_type_from:
        type: type
      literal_type_from:
        type: type
      value:
        type: int
      int_type_to:
        type: type
      expected_output:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a", type_=int_type_from(), value=literal_type_from(value)
    )
    cast_expr = Cast(value=astx.Identifier("a"), target_type=int_type_to())
    cast_var = astx.InlineVariableDeclaration(
        name="r", type_=int_type_to(), value=cast_expr
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

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


@pytest.mark.parametrize(
    "astx_type, literal_type, value, expected_output",
    [
        (astx.Int32, astx.LiteralInt32, 42, "42"),
        (astx.Float32, astx.LiteralFloat32, 42.0, "42.000000"),
        (astx.Boolean, astx.LiteralBoolean, True, "1"),
        (astx.Boolean, astx.LiteralBoolean, False, "0"),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_cast_to_string(
    builder_class: type[Builder],
    astx_type: type,
    literal_type: type,
    value: object,
    expected_output: str,
) -> None:
    """
    title: Cast various types to string and verify printed output.
    parameters:
      builder_class:
        type: type[Builder]
      astx_type:
        type: type
      literal_type:
        type: type
      value:
        type: object
      expected_output:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a", type_=astx_type(), value=literal_type(value)
    )
    cast_var = astx.InlineVariableDeclaration(
        name="r",
        type_=astx.String(),
        value=Cast(value=astx.Identifier("a"), target_type=astx.String()),
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(cast_var)
    main_block.append(print_stmt)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(
        astx.FunctionDef(prototype=main_proto, body=main_block)
    )

    check_result("build", builder, module, expected_output=expected_output)


@pytest.mark.parametrize(
    "int_type, literal_type, expected_str",
    [
        (astx.Int32, astx.LiteralInt32, "7.000000"),
        (astx.Int8, astx.LiteralInt8, "7.000000"),
        (astx.Int64, astx.LiteralInt64, "7.000000"),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_cast_int_to_float(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
    expected_str: str,
) -> None:
    """
    title: Cast int to float.
    parameters:
      builder_class:
        type: type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
      expected_str:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    # a: int = 7
    decl_a = astx.VariableDeclaration(
        name="a", type_=int_type(), value=literal_type(7)
    )

    # r: float32 = cast(a)
    cast_expr = Cast(value=astx.Identifier("a"), target_type=astx.Float32())
    cast_var = astx.InlineVariableDeclaration(
        name="r", type_=astx.Float32(), value=cast_expr
    )

    # print(r)  -- will show "7.000000" proving the cast actually happened
    print_stmt = PrintExpr(message=astx.Identifier("r"))

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

    check_result("build", builder, module, expected_output=expected_str)


@pytest.mark.parametrize(
    "float_type, literal_type, value, int_type, expected_output",
    [
        # truncates toward zero, not rounds
        (astx.Float32, astx.LiteralFloat32, 7.9, astx.Int32, "7"),
        (astx.Float32, astx.LiteralFloat32, 7.1, astx.Int32, "7"),
        # negative truncation
        (astx.Float32, astx.LiteralFloat32, -3.9, astx.Int32, "-3"),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_cast_float_to_int(
    builder_class: type[Builder],
    float_type: type,
    literal_type: type,
    value: float,
    int_type: type,
    expected_output: str,
) -> None:
    """
    title: Test float -> int cast truncates toward zero.
    parameters:
      builder_class:
        type: type[Builder]
      float_type:
        type: type
      literal_type:
        type: type
      value:
        type: float
      int_type:
        type: type
      expected_output:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a", type_=float_type(), value=literal_type(value)
    )
    cast_var = astx.InlineVariableDeclaration(
        name="r",
        type_=int_type(),
        value=Cast(value=astx.Identifier("a"), target_type=int_type()),
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(cast_var)
    main_block.append(print_stmt)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(
        astx.FunctionDef(prototype=main_proto, body=main_block)
    )

    check_result("build", builder, module, expected_output=expected_output)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_cast_same_type_noop(builder_class: type[Builder]) -> None:
    """
    title: Cast a value to its own type should be a no-op.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a", type_=astx.Int32(), value=astx.LiteralInt32(42)
    )
    cast_var = astx.InlineVariableDeclaration(
        name="r",
        type_=astx.Int32(),
        value=Cast(value=astx.Identifier("a"), target_type=astx.Int32()),
    )
    print_stmt = PrintExpr(message=astx.Identifier("r"))

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(cast_var)
    main_block.append(print_stmt)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(
        astx.FunctionDef(prototype=main_proto, body=main_block)
    )

    check_result("build", builder, module, expected_output="42")

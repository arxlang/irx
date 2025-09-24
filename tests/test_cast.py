"""Tests for the Casting."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import Cast

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
    """Test casting int types between different widths."""
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
    """Test casting int -> float -> int, returning int result."""
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

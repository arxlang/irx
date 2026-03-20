"""
title: Tests for Bitwise AND (&) operator.
"""

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr
from .conftest import check_result

@pytest.mark.parametrize(
    "op,lhs,rhs,expected",
    [
        ("&", 6, 3, lambda a, b: str(a & b)),
        ("|", 6, 3, lambda a, b: str(a | b)),
        ("^", 6, 3, lambda a, b: str(a ^ b)),
        ("<<", 3, 2, lambda a, b: str(a << b)),
        (">>", 12, 2, lambda a, b: str(a >> b)),
    ],
)
@pytest.mark.parametrize(
    "int_type,literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_bitwise_binary_ops(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
    op: str,
    lhs: int,
    rhs: int,
    expected,
) -> None:
    builder = builder_class()
    module = builder.module()
    left = literal_type(lhs)
    right = literal_type(rhs)
    expr = astx.BinaryOp(op, left, right)
    decl = astx.VariableDeclaration(
        name="result", type_=int_type(), value=expr, mutability=astx.MutabilityKind.mutable
    )
    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl)
    main_block.append(PrintExpr(astx.Identifier("result")))
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)
    check_result("build", builder, module, expected_output=expected(lhs, rhs))

@pytest.mark.parametrize(
    "val",
    [15, 0, 1, 255],
)
@pytest.mark.parametrize(
    "int_type,literal_type,bitwidth",
    [
        (astx.Int8, astx.LiteralInt8, 8),
        (astx.Int16, astx.LiteralInt16, 16),
        (astx.Int32, astx.LiteralInt32, 32),
        (astx.Int64, astx.LiteralInt64, 64),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_bitwise_not(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
    bitwidth: int,
    val: int,
) -> None:
    builder = builder_class()
    module = builder.module()
    expr = astx.UnaryOp("~", literal_type(val))
    decl = astx.VariableDeclaration(
        name="result", type_=int_type(), value=expr, mutability=astx.MutabilityKind.mutable
    )
    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl)
    main_block.append(PrintExpr(astx.Identifier("result")))
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)
    # Compute the expected result as a signed integer of the correct width
    mask = (1 << bitwidth) - 1
    unsigned = (~val) & mask
    sign_bit = 1 << (bitwidth - 1)
    if unsigned & sign_bit:
        expected = str(unsigned - (1 << bitwidth))
    else:
        expected = str(unsigned)
    check_result("build", builder, module, expected_output=expected)

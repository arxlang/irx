"""
title: Tests for the UnaryOp.
"""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_unary_op.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_unary_op_increment_decrement(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test ASTx UnaryOp for increment and decrement operations.
    parameters:
      action:
        type: str
      expected_file:
        type: str
      builder_class:
        type: Type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a",
        type_=int_type(),
        value=literal_type(5),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.VariableDeclaration(
        name="b",
        type_=int_type(),
        value=literal_type(10),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_c = astx.VariableDeclaration(
        name="c",
        type_=int_type(),
        value=literal_type(0),
        mutability=astx.MutabilityKind.mutable,
    )

    var_a = astx.Identifier("a")
    var_b = astx.Identifier("b")
    var_c = astx.Identifier("c")

    incr_a = astx.UnaryOp(op_code="++", operand=var_a)
    incr_a.type_ = int_type()
    decr_b = astx.UnaryOp(op_code="--", operand=var_b)
    decr_b.type_ = int_type()
    not_c = astx.UnaryOp(op_code="!", operand=var_c)
    not_c.type_ = int_type()

    final_expr = incr_a + decr_b + not_c

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(decl_b)
    main_block.append(decl_c)
    main_block.append(final_expr)
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result(action, builder, module, expected_file)


def test_decrement_operator() -> None:
    """
    title: Test standalone decrement operator.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(10),
        mutability=astx.MutabilityKind.mutable,
    )

    decr = astx.UnaryOp(op_code="--", operand=astx.Identifier("x"))
    decr.type_ = astx.Int32()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(decr)
    block.append(astx.FunctionReturn(astx.Identifier("x")))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="9")


def test_not_operator() -> None:
    """
    title: Test standalone NOT operator.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="flag",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )

    not_op = astx.UnaryOp(
        op_code="!", operand=astx.Identifier("flag")
    )
    not_op.type_ = astx.Int32()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(not_op)
    block.append(astx.FunctionReturn(astx.Identifier("flag")))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output="1")


def test_increment_const_error() -> None:
    """Test ++ on const variable raises error (line 686)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="c", type_=astx.Int32(),
        value=astx.LiteralInt32(5),
        mutability=astx.MutabilityKind.constant,
    )
    incr = astx.UnaryOp(op_code="++", operand=astx.Identifier("c"))
    incr.type_ = astx.Int32()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(incr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="Cannot mutate"):
        check_result("build", builder, module)


def test_decrement_const_error() -> None:
    """Test -- on const variable raises error (line 705)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="c", type_=astx.Int32(),
        value=astx.LiteralInt32(5),
        mutability=astx.MutabilityKind.constant,
    )
    decr = astx.UnaryOp(op_code="--", operand=astx.Identifier("c"))
    decr.type_ = astx.Int32()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(decr)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="Cannot mutate"):
        check_result("build", builder, module)


def test_not_const_error() -> None:
    """Test ! on const variable raises error (line 725)."""
    builder = LLVMLiteIR()
    module = builder.module()

    decl = astx.InlineVariableDeclaration(
        name="c", type_=astx.Int32(),
        value=astx.LiteralInt32(1),
        mutability=astx.MutabilityKind.constant,
    )
    not_op = astx.UnaryOp(op_code="!", operand=astx.Identifier("c"))
    not_op.type_ = astx.Int32()

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl)
    block.append(not_op)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="Cannot mutate"):
        check_result("build", builder, module)

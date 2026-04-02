"""
title: Tests for the UnaryOp.
"""

import pytest

from irx import astx
from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
        (astx.UInt32, astx.LiteralUInt32),
        (astx.UInt16, astx.LiteralUInt16),
        (astx.UInt8, astx.LiteralUInt8),
        (astx.UInt64, astx.LiteralUInt64),
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
        LLVMBuilder,
    ],
)
def test_unary_op_increment_decrement(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
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
        type: type[Builder]
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


@pytest.mark.parametrize(
    "float_type, literal_type",
    [
        (astx.Float32, astx.LiteralFloat32),
        (astx.Float64, astx.LiteralFloat64),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_unary_op_increment_decrement_float(
    builder_class: type[Builder],
    float_type: type,
    literal_type: type,
) -> None:
    """
    title: Test ASTx UnaryOp increment and decrement for float types.
    parameters:
      builder_class:
        type: type[Builder]
      float_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a",
        type_=float_type(),
        value=literal_type(5.0),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.VariableDeclaration(
        name="b",
        type_=float_type(),
        value=literal_type(10.0),
        mutability=astx.MutabilityKind.mutable,
    )

    incr_a = astx.UnaryOp(op_code="++", operand=astx.Identifier("a"))
    incr_a.type_ = float_type()
    decr_b = astx.UnaryOp(op_code="--", operand=astx.Identifier("b"))
    decr_b.type_ = float_type()

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(decl_b)
    main_block.append(incr_a)
    main_block.append(decr_b)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result("build", builder, module, "")


@pytest.mark.parametrize(
    "int_type, literal_type, value, expected_output",
    [
        (astx.Int32, astx.LiteralInt32, 0, "1"),
        (astx.Int32, astx.LiteralInt32, 5, "0"),
        (astx.Int32, astx.LiteralInt32, -3, "0"),
        (astx.Int16, astx.LiteralInt16, 0, "1"),
        (astx.Int16, astx.LiteralInt16, 7, "0"),
        (astx.UInt32, astx.LiteralUInt32, 0, "1"),
        (astx.UInt32, astx.LiteralUInt32, 10, "0"),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_unary_op_logical_not_int(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
    value: int,
    expected_output: str,
) -> None:
    """
    title: Test logical NOT (!) for integer types.
    parameters:
      builder_class:
        type: type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
      value:
        type: int
      expected_output:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a",
        type_=int_type(),
        value=literal_type(value),
        mutability=astx.MutabilityKind.mutable,
    )

    var_a = astx.Identifier("a")

    not_a = astx.UnaryOp(op_code="!", operand=var_a)
    not_a.type_ = int_type()

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(not_a)
    main_block.append(astx.FunctionReturn(astx.Identifier("a")))

    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result("build", builder, module, expected_output=expected_output)


@pytest.mark.parametrize(
    "value, expected_output",
    [
        (False, "1"),
        (True, "0"),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_unary_op_logical_not_boolean(
    builder_class: type[Builder],
    value: bool,
    expected_output: str,
) -> None:
    """
    title: Test logical NOT (!) for boolean type.
    parameters:
      builder_class:
        type: type[Builder]
      value:
        type: bool
      expected_output:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a",
        type_=astx.Boolean(),
        value=astx.LiteralBoolean(value),
        mutability=astx.MutabilityKind.mutable,
    )

    not_a = astx.UnaryOp(op_code="!", operand=astx.Identifier("a"))
    not_a.type_ = astx.Boolean()

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Boolean(),
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(not_a)
    main_block.append(astx.FunctionReturn(astx.Identifier("a")))

    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result("build", builder, module, expected_output=expected_output)

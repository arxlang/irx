"""
title: Test While Loop statements.
"""

import pytest

from irx import astx
from irx.astx import PrintExpr
from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder
from llvmlite import binding as llvm

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
        (astx.Float32, astx.LiteralFloat32),
        (astx.Float64, astx.LiteralFloat64),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_while_expr.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_while_expr(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test the While expression translation.
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

    # Identifier declaration and initialization: int a = 0
    init_var = astx.InlineVariableDeclaration(
        "a",
        type_=int_type(),
        value=literal_type(0),
        mutability=astx.MutabilityKind.mutable,
    )

    # Condition: a < 5
    var_a = astx.Identifier("a")
    cond = astx.BinaryOp(op_code="<", lhs=var_a, rhs=literal_type(5))

    # Update: a = a + 1  (works for int and float; ++ only works for int)
    update = astx.VariableAssignment(
        name="a",
        value=astx.BinaryOp(
            op_code="+",
            lhs=astx.Identifier("a"),
            rhs=literal_type(1),
        ),
    )

    # Body
    body = astx.Block()
    body.append(update)
    body.append(literal_type(2))

    while_expr = astx.WhileStmt(condition=cond, body=body)

    # Main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    fn_block = astx.Block()
    fn_block.append(init_var)
    fn_block.append(while_expr)
    fn_block.append(astx.FunctionReturn(literal_type(0)))

    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_while_empty_body_no_crash(
    builder_class: type[Builder],
) -> None:
    """
    title: WhileStmt with empty body must not crash on empty stack.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    init_var = astx.InlineVariableDeclaration(
        "a",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )

    var_a = astx.Identifier("a")
    cond = astx.BinaryOp(
        op_code="<",
        lhs=var_a,
        rhs=astx.LiteralInt32(5),
    )

    # Empty body leaves no value on result_stack.
    body = astx.Block()

    while_expr = astx.WhileStmt(condition=cond, body=body)

    proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    fn_block = astx.Block()
    fn_block.append(init_var)
    fn_block.append(while_expr)
    fn_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    llvm.parse_assembly(builder.translate(module))


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_while_falsy_body_value_no_crash(
    builder_class: type[Builder],
) -> None:
    """
    title: WhileStmt must keep its back-edge when body leaves a falsy value.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    init_var = astx.InlineVariableDeclaration(
        "a",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )

    var_a = astx.Identifier("a")
    cond = astx.BinaryOp(
        op_code="<",
        lhs=var_a,
        rhs=astx.LiteralInt32(5),
    )
    update = astx.UnaryOp(op_code="++", operand=var_a)

    body = astx.Block()
    body.append(update)
    body.append(astx.LiteralInt32(0))

    while_expr = astx.WhileStmt(condition=cond, body=body)

    proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    fn_block = astx.Block()
    fn_block.append(init_var)
    fn_block.append(while_expr)
    fn_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    llvm.parse_assembly(builder.translate(module))


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
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_while_false_condition(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test While loop with a condition that is false from the start.
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

    # Condition is always false: 10 < 0
    cond = astx.BinaryOp(
        op_code="<",
        lhs=literal_type(10),
        rhs=literal_type(0),
    )

    # Body that should never execute
    body = astx.Block()
    body.append(literal_type(1))

    while_expr = astx.WhileStmt(condition=cond, body=body)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    fn_block = astx.Block()
    fn_block.append(while_expr)
    fn_block.append(astx.FunctionReturn(literal_type(0)))

    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_while_float_condition(
    builder_class: type[Builder],
) -> None:
    """
    title: Test WhileStmt with a Float32 condition covers fcmp_ordered path.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    # float a = 3.0
    init_var = astx.InlineVariableDeclaration(
        "a",
        type_=astx.Float32(),
        value=astx.LiteralFloat32(3.0),
        mutability=astx.MutabilityKind.mutable,
    )

    # condition: a (evaluates to float, hits fcmp_ordered 0.0)
    var_a = astx.Identifier("a")
    cond = var_a

    # body: a = a - 1.0
    dec = astx.VariableAssignment(
        name="a",
        value=astx.BinaryOp(
            op_code="-", lhs=var_a, rhs=astx.LiteralFloat32(1.0)
        ),
    )
    body = astx.Block()
    body.append(dec)

    while_expr = astx.WhileStmt(condition=cond, body=body)

    # Print "done" after loop to prove execution completed.
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn_block = astx.Block()
    fn_block.append(init_var)
    fn_block.append(while_expr)
    fn_block.append(PrintExpr(astx.LiteralUTF8String("done")))
    fn_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)
    module = builder.module()
    module.block.append(fn_main)

    check_result("build", builder, module, expected_output="done")

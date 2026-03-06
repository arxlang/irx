"""
title: Test Scoping rules for const variables.
"""


import astx

from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


def test_const_does_not_leak_across_functions() -> None:
    """
    title: Test that a const declared in one function does not prevent a mutable
    variable of the same name from being modified in a subsequent function.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    # Function 1: Declares `const x`
    decl_f1 = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(10),
        mutability=astx.MutabilityKind.constant,
    )
    proto_f1 = astx.FunctionPrototype(
        name="f1", args=astx.Arguments(), return_type=astx.Int32()
    )
    block_f1 = astx.Block()
    block_f1.append(decl_f1)
    block_f1.append(astx.FunctionReturn(astx.Identifier("x")))
    fn_f1 = astx.FunctionDef(prototype=proto_f1, body=block_f1)
    module.block.append(fn_f1)

    # Function 2 (main): Declares `mutable x` and mutates it
    decl_f2 = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(20),
        mutability=astx.MutabilityKind.mutable,
    )
    assign_f2 = astx.VariableAssignment(name="x", value=astx.LiteralInt32(42))
    proto_f2 = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block_f2 = astx.Block()
    block_f2.append(decl_f2)
    block_f2.append(assign_f2)
    block_f2.append(astx.FunctionReturn(astx.Identifier("x")))
    fn_f2 = astx.FunctionDef(prototype=proto_f2, body=block_f2)
    module.block.append(fn_f2)

    expected_output = "42"
    check_result("build", builder, module, expected_output=expected_output)


def test_mutable_loop_variable_shadows_outer_const() -> None:
    """
    title: Test that a mutable loop variable can shadow an outer const variable
    without triggering a const reassignment error during loop update.
    """
    builder = LLVMLiteIR()
    module = builder.module()

    # Outer scope const x = 7
    decl_outer = astx.InlineVariableDeclaration(
        name="x",
        type_=astx.Int32(),
        value=astx.LiteralInt32(7),
        mutability=astx.MutabilityKind.constant,
    )

    var_x = astx.InlineVariableDeclaration(
        "x", type_=astx.Int32(), mutability=astx.MutabilityKind.mutable
    )
    start = astx.LiteralInt32(0)
    end = astx.LiteralInt32(3)
    step = astx.LiteralInt32(1)

    # Internal body mutates x to prove mutability check doesn't trigger
    body = astx.Block()
    assign_inner = astx.VariableAssignment(
        name="x", value=astx.LiteralInt32(42)
    )
    body.append(assign_inner)

    loop = astx.ForRangeLoopStmt(
        variable=var_x,
        start=start,
        end=end,
        step=step,
        body=body,
    )

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    block.append(decl_outer)
    block.append(loop)
    block.append(astx.FunctionReturn(astx.Identifier("x")))
    fn_main = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn_main)

    expected_output = "7"
    check_result("build", builder, module, expected_output=expected_output)

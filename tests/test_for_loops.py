"""
title: Test For Loop statements.
"""

import pytest

from irx import astx
from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder

from .conftest import check_result


def build_int32_main_module(builder: Builder, body: astx.Block) -> astx.Module:
    """
    title: Build a module with a single int32 main function.
    parameters:
      builder:
        type: Builder
      body:
        type: astx.Block
    returns:
      type: astx.Module
    """
    module = builder.module()
    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    main_fn = astx.FunctionDef(prototype=main_proto, body=body)
    module.block.append(main_fn)
    return module


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
        (astx.Float32, astx.LiteralFloat32),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_for_range.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_for_range(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test For Range statement.
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

    # `for` statement
    var_a = astx.InlineVariableDeclaration(
        "a", type_=int_type(), mutability=astx.MutabilityKind.mutable
    )
    start = literal_type(1)
    end = literal_type(10)
    step = literal_type(1)
    body = astx.Block()
    body.append(literal_type(2))
    for_loop = astx.ForRangeLoopStmt(
        variable=var_a,
        start=start,
        end=end,
        step=step,
        body=body,
    )

    # main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    block = astx.Block()
    block.append(for_loop)
    block.append(astx.FunctionReturn(literal_type(0)))
    fn_main = astx.FunctionDef(prototype=proto, body=block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_for_range_empty_body_no_crash(
    builder_class: type[Builder],
) -> None:
    """
    title: ForRangeLoopStmt with empty body must not crash on empty stack.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    variable = astx.InlineVariableDeclaration(
        "a",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
    )
    for_loop = astx.ForRangeLoopStmt(
        variable=variable,
        start=astx.LiteralInt32(1),
        end=astx.LiteralInt32(10),
        step=astx.LiteralInt32(1),
        body=astx.Block(),
    )

    proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    block = astx.Block()
    block.append(for_loop)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn_main = astx.FunctionDef(prototype=proto, body=block)

    module = builder.module()
    module.block.append(fn_main)

    check_result("build", builder, module, "")


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_for_range_break_allows_following_statements(
    builder_class: type[Builder],
) -> None:
    """
    title: ForRangeLoopStmt break must still reach statements after the loop.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    result_decl = astx.InlineVariableDeclaration(
        "result",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )
    loop_var = astx.InlineVariableDeclaration(
        "i",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
    )
    loop_body = astx.Block()
    loop_body.append(astx.BreakStmt())

    loop = astx.ForRangeLoopStmt(
        variable=loop_var,
        start=astx.LiteralInt32(0),
        end=astx.LiteralInt32(4),
        step=astx.LiteralInt32(1),
        body=loop_body,
    )

    main_body = astx.Block()
    main_body.append(result_decl)
    main_body.append(loop)
    main_body.append(astx.VariableAssignment("result", astx.LiteralInt32(7)))
    main_body.append(astx.FunctionReturn(astx.Identifier("result")))

    module = build_int32_main_module(builder, main_body)
    check_result("build", builder, module, "", expected_output="7")


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_for_range_continue_uses_step_block(
    builder_class: type[Builder],
) -> None:
    """
    title: ForRangeLoopStmt continue must branch through the step block.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    loop_var = astx.InlineVariableDeclaration(
        "i",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
    )
    loop_body = astx.Block()
    loop_body.append(astx.ContinueStmt())

    loop = astx.ForRangeLoopStmt(
        variable=loop_var,
        start=astx.LiteralInt32(0),
        end=astx.LiteralInt32(4),
        step=astx.LiteralInt32(1),
        body=loop_body,
    )

    main_body = astx.Block()
    main_body.append(loop)
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    module = build_int32_main_module(builder, main_body)
    ir_result = builder.translate(module)

    assert "for.step" in ir_result
    assert 'br label %"for.step"' in ir_result


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_for_count_break_allows_following_statements(
    builder_class: type[Builder],
) -> None:
    """
    title: ForCountLoopStmt break must still reach statements after the loop.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    result_decl = astx.InlineVariableDeclaration(
        "result",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )
    initializer = astx.InlineVariableDeclaration(
        "i",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )
    loop_var = astx.Identifier("i")
    loop_body = astx.Block()
    loop_body.append(astx.BreakStmt())

    loop = astx.ForCountLoopStmt(
        initializer=initializer,
        condition=astx.BinaryOp(
            op_code="<",
            lhs=loop_var,
            rhs=astx.LiteralInt32(4),
        ),
        update=astx.UnaryOp(op_code="++", operand=loop_var),
        body=loop_body,
    )

    main_body = astx.Block()
    main_body.append(result_decl)
    main_body.append(loop)
    main_body.append(astx.VariableAssignment("result", astx.LiteralInt32(9)))
    main_body.append(astx.FunctionReturn(astx.Identifier("result")))

    module = build_int32_main_module(builder, main_body)
    check_result("build", builder, module, "", expected_output="9")


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_for_count_continue_uses_update_block(
    builder_class: type[Builder],
) -> None:
    """
    title: ForCountLoopStmt continue must branch through the update block.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    initializer = astx.InlineVariableDeclaration(
        "i",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )
    loop_var = astx.Identifier("i")
    loop_body = astx.Block()
    loop_body.append(astx.ContinueStmt())

    loop = astx.ForCountLoopStmt(
        initializer=initializer,
        condition=astx.BinaryOp(
            op_code="<",
            lhs=loop_var,
            rhs=astx.LiteralInt32(4),
        ),
        update=astx.UnaryOp(op_code="++", operand=loop_var),
        body=loop_body,
    )

    main_body = astx.Block()
    main_body.append(loop)
    main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    module = build_int32_main_module(builder, main_body)
    ir_result = builder.translate(module)

    assert "loop.update" in ir_result
    assert 'br label %"loop.update"' in ir_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int64, astx.LiteralInt64),
        (astx.Int8, astx.LiteralInt8),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_for_loops.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
def test_for_count(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test the For Count statement.
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

    init_a = astx.InlineVariableDeclaration(
        "a2",
        type_=int_type(),
        value=literal_type(0),
        mutability=astx.MutabilityKind.mutable,
    )
    var_a = astx.Identifier("a2")
    cond = astx.BinaryOp(op_code="<", lhs=var_a, rhs=literal_type(10))
    update = astx.UnaryOp(op_code="++", operand=var_a)

    for_body = astx.Block()
    for_body.append(literal_type(2))
    for_loop = astx.ForCountLoopStmt(
        initializer=init_a,
        condition=cond,
        update=update,
        body=for_body,
    )

    # main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    fn_block = astx.Block()
    fn_block.append(for_loop)
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
def test_for_count_empty_body_no_crash(
    builder_class: type[Builder],
) -> None:
    """
    title: ForCountLoopStmt with empty body must not crash on empty stack.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()

    initializer = astx.InlineVariableDeclaration(
        "a2",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )
    variable = astx.Identifier("a2")
    for_loop = astx.ForCountLoopStmt(
        initializer=initializer,
        condition=astx.BinaryOp(
            op_code="<",
            lhs=variable,
            rhs=astx.LiteralInt32(10),
        ),
        update=astx.UnaryOp(op_code="++", operand=variable),
        body=astx.Block(),
    )

    proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    fn_block = astx.Block()
    fn_block.append(for_loop)
    fn_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    fn_main = astx.FunctionDef(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    check_result("build", builder, module, "")

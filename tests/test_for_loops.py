"""
title: Test For Loop statements.
"""

import pytest

from irx import astx
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder

from .conftest import assert_ir_parses, assert_jit_int_main_result


def build_int32_main_module(body: astx.Block) -> astx.Module:
    """
    title: Build a module with one Int32 main function.
    parameters:
      body:
        type: astx.Block
    returns:
      type: astx.Module
    """
    module = astx.Module()
    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))
    return module


def block_of(*nodes: astx.AST) -> astx.Block:
    """
    title: Build one block from positional AST nodes.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Block
    """
    block = astx.Block()
    for node in nodes:
        block.append(node)
    return block


def add_assign(name: str, value: astx.AST) -> astx.VariableAssignment:
    """
    title: Build one additive variable assignment.
    parameters:
      name:
        type: str
      value:
        type: astx.AST
    returns:
      type: astx.VariableAssignment
    """
    return astx.VariableAssignment(
        name,
        astx.BinaryOp(op_code="+", lhs=astx.Identifier(name), rhs=value),
    )


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
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_range_lowers_for_numeric_induction_types(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: For-range loops should lower cleanly for supported numeric IV types.
    parameters:
      builder_class:
        type: type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()
    loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "a",
            type_=int_type(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=literal_type(1),
        end=literal_type(10),
        step=literal_type(1),
        body=block_of(literal_type(2)),
    )
    module = build_int32_main_module(
        block_of(loop, astx.FunctionReturn(astx.LiteralInt32(0)))
    )

    assert_jit_int_main_result(builder, module, 0)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_range_empty_body_no_crash(
    builder_class: type[Builder],
) -> None:
    """
    title: ForRangeLoopStmt with empty body must still lower and execute.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    for_loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "a",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=astx.LiteralInt32(1),
        end=astx.LiteralInt32(10),
        step=astx.LiteralInt32(1),
        body=astx.Block(),
    )
    module = build_int32_main_module(
        block_of(for_loop, astx.FunctionReturn(astx.LiteralInt32(0)))
    )

    assert_jit_int_main_result(builder, module, 0)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
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
    loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=astx.LiteralInt32(0),
        end=astx.LiteralInt32(4),
        step=astx.LiteralInt32(1),
        body=block_of(astx.BreakStmt()),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "result",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.VariableAssignment("result", astx.LiteralInt32(7)),
            astx.FunctionReturn(astx.Identifier("result")),
        )
    )

    assert_jit_int_main_result(builder, module, 7)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
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
    loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=astx.LiteralInt32(0),
        end=astx.LiteralInt32(4),
        step=astx.LiteralInt32(1),
        body=block_of(astx.ContinueStmt()),
    )
    module = build_int32_main_module(
        block_of(loop, astx.FunctionReturn(astx.LiteralInt32(0)))
    )

    ir_result = builder.translate(module)

    assert_ir_parses(ir_result)
    assert "for.range.step" in ir_result
    assert 'br label %"for.range.step"' in ir_result
    assert "for.range.exit" in ir_result


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_range_continue_preserves_step_progression(
    builder_class: type[Builder],
) -> None:
    """
    title: For-range continue should still advance through the step path.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=astx.LiteralInt32(0),
        end=astx.LiteralInt32(4),
        step=astx.LiteralInt32(1),
        body=block_of(
            astx.IfStmt(
                condition=astx.BinaryOp(
                    op_code="<",
                    lhs=astx.Identifier("i"),
                    rhs=astx.LiteralInt32(2),
                ),
                then=block_of(astx.ContinueStmt()),
                else_=None,
            ),
            add_assign("sum", astx.Identifier("i")),
        ),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "sum",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("sum")),
        )
    )

    assert_jit_int_main_result(builder, module, 5)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_range_accumulates_with_stable_induction_progression(
    builder_class: type[Builder],
) -> None:
    """
    title: For-range accumulation should observe a stable induction sequence.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=astx.LiteralInt32(0),
        end=astx.LiteralInt32(5),
        step=astx.LiteralInt32(1),
        body=block_of(add_assign("sum", astx.Identifier("i"))),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "sum",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("sum")),
        )
    )

    assert_jit_int_main_result(builder, module, 10)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_range_caches_end_and_step_before_iteration(
    builder_class: type[Builder],
) -> None:
    """
    title: For-range end and step are observed before the first iteration.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=astx.LiteralInt32(0),
        end=astx.Identifier("end"),
        step=astx.Identifier("step"),
        body=block_of(
            add_assign("sum", astx.Identifier("i")),
            astx.VariableAssignment("step", astx.LiteralInt32(1)),
            astx.VariableAssignment("end", astx.LiteralInt32(100)),
        ),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "sum",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.InlineVariableDeclaration(
                "step",
                type_=astx.Int32(),
                value=astx.LiteralInt32(2),
                mutability=astx.MutabilityKind.mutable,
            ),
            astx.InlineVariableDeclaration(
                "end",
                type_=astx.Int32(),
                value=astx.LiteralInt32(6),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("sum")),
        )
    )

    assert_jit_int_main_result(builder, module, 6)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_range_body_mutation_feeds_step(
    builder_class: type[Builder],
) -> None:
    """
    title: >-
      Mutating the range induction variable in the body should feed the step.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=astx.LiteralInt32(0),
        end=astx.LiteralInt32(6),
        step=astx.LiteralInt32(1),
        body=block_of(
            astx.VariableAssignment(
                "i",
                astx.BinaryOp(
                    op_code="+",
                    lhs=astx.Identifier("i"),
                    rhs=astx.LiteralInt32(1),
                ),
            ),
            add_assign("sum", astx.Identifier("i")),
        ),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "sum",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("sum")),
        )
    )

    assert_jit_int_main_result(builder, module, 9)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_range_nested_loops_accumulate_cleanly(
    builder_class: type[Builder],
) -> None:
    """
    title: Nested for-range loops should preserve nearest-loop control flow.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    inner_loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "j",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=astx.LiteralInt32(0),
        end=astx.LiteralInt32(2),
        step=astx.LiteralInt32(1),
        body=block_of(
            astx.UnaryOp(op_code="++", operand=astx.Identifier("count"))
        ),
    )
    outer_loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=astx.LiteralInt32(0),
        end=astx.LiteralInt32(3),
        step=astx.LiteralInt32(1),
        body=block_of(inner_loop),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "count",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            outer_loop,
            astx.FunctionReturn(astx.Identifier("count")),
        )
    )

    assert_jit_int_main_result(builder, module, 6)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_range_zero_iterations_preserve_post_loop_state(
    builder_class: type[Builder],
) -> None:
    """
    title: A zero-trip for-range loop should leave post-loop state untouched.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.ForRangeLoopStmt(
        variable=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
        ),
        start=astx.LiteralInt32(4),
        end=astx.LiteralInt32(4),
        step=astx.LiteralInt32(1),
        body=block_of(astx.VariableAssignment("result", astx.LiteralInt32(0))),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "result",
                type_=astx.Int32(),
                value=astx.LiteralInt32(41),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("result")),
        )
    )

    assert_jit_int_main_result(builder, module, 41)


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int64, astx.LiteralInt64),
        (astx.Int8, astx.LiteralInt8),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_count_lowers_for_numeric_induction_types(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: For-count loops should lower cleanly for supported numeric IV types.
    parameters:
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
    for_loop = astx.ForCountLoopStmt(
        initializer=init_a,
        condition=astx.BinaryOp(op_code="<", lhs=var_a, rhs=literal_type(10)),
        update=astx.UnaryOp(op_code="++", operand=var_a),
        body=block_of(literal_type(2)),
    )
    module = build_int32_main_module(
        block_of(for_loop, astx.FunctionReturn(astx.LiteralInt32(0)))
    )

    assert_jit_int_main_result(builder, module, 0)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_count_empty_body_no_crash(
    builder_class: type[Builder],
) -> None:
    """
    title: ForCountLoopStmt with empty body must still lower and execute.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    variable = astx.Identifier("a2")
    for_loop = astx.ForCountLoopStmt(
        initializer=astx.InlineVariableDeclaration(
            "a2",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        condition=astx.BinaryOp(
            op_code="<",
            lhs=variable,
            rhs=astx.LiteralInt32(10),
        ),
        update=astx.UnaryOp(op_code="++", operand=variable),
        body=astx.Block(),
    )
    module = build_int32_main_module(
        block_of(for_loop, astx.FunctionReturn(astx.LiteralInt32(0)))
    )

    assert_jit_int_main_result(builder, module, 0)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
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
    initializer = astx.InlineVariableDeclaration(
        "i",
        type_=astx.Int32(),
        value=astx.LiteralInt32(0),
        mutability=astx.MutabilityKind.mutable,
    )
    loop_var = astx.Identifier("i")
    loop = astx.ForCountLoopStmt(
        initializer=initializer,
        condition=astx.BinaryOp(
            op_code="<",
            lhs=loop_var,
            rhs=astx.LiteralInt32(4),
        ),
        update=astx.UnaryOp(op_code="++", operand=loop_var),
        body=block_of(astx.BreakStmt()),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "result",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.VariableAssignment("result", astx.LiteralInt32(9)),
            astx.FunctionReturn(astx.Identifier("result")),
        )
    )

    assert_jit_int_main_result(builder, module, 9)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
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
    loop = astx.ForCountLoopStmt(
        initializer=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(4),
        ),
        update=astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
        body=block_of(astx.ContinueStmt()),
    )
    module = build_int32_main_module(
        block_of(loop, astx.FunctionReturn(astx.LiteralInt32(0)))
    )

    ir_result = builder.translate(module)

    assert_ir_parses(ir_result)
    assert "for.count.update" in ir_result
    assert 'br label %"for.count.update"' in ir_result
    assert "for.count.exit" in ir_result


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_count_continue_runs_update_before_rechecking_condition(
    builder_class: type[Builder],
) -> None:
    """
    title: For-count continue should still execute the update expression.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.ForCountLoopStmt(
        initializer=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(5),
        ),
        update=astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
        body=block_of(
            astx.IfStmt(
                condition=astx.BinaryOp(
                    op_code="<",
                    lhs=astx.Identifier("i"),
                    rhs=astx.LiteralInt32(2),
                ),
                then=block_of(astx.ContinueStmt()),
                else_=None,
            ),
            add_assign("sum", astx.Identifier("i")),
        ),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "sum",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("sum")),
        )
    )

    assert_jit_int_main_result(builder, module, 9)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_count_update_expression_result_becomes_next_iteration_value(
    builder_class: type[Builder],
) -> None:
    """
    title: For-count update result should become the next induction value.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.ForCountLoopStmt(
        initializer=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(6),
        ),
        update=astx.BinaryOp(
            op_code="+",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(2),
        ),
        body=block_of(add_assign("sum", astx.Identifier("i"))),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "sum",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("sum")),
        )
    )

    assert_jit_int_main_result(builder, module, 6)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_count_body_mutation_feeds_update(
    builder_class: type[Builder],
) -> None:
    """
    title: Body mutation in for-count should be observed by the update step.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.ForCountLoopStmt(
        initializer=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(6),
        ),
        update=astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
        body=block_of(
            astx.VariableAssignment(
                "i",
                astx.BinaryOp(
                    op_code="+",
                    lhs=astx.Identifier("i"),
                    rhs=astx.LiteralInt32(1),
                ),
            ),
            add_assign("sum", astx.Identifier("i")),
        ),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "sum",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("sum")),
        )
    )

    assert_jit_int_main_result(builder, module, 9)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_count_nested_loops_accumulate_cleanly(
    builder_class: type[Builder],
) -> None:
    """
    title: Nested for-count loops should preserve nearest-loop control flow.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    inner_loop = astx.ForCountLoopStmt(
        initializer=astx.InlineVariableDeclaration(
            "j",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("j"),
            rhs=astx.LiteralInt32(4),
        ),
        update=astx.UnaryOp(op_code="++", operand=astx.Identifier("j")),
        body=block_of(
            astx.UnaryOp(op_code="++", operand=astx.Identifier("count"))
        ),
    )
    outer_loop = astx.ForCountLoopStmt(
        initializer=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(3),
        ),
        update=astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
        body=block_of(inner_loop),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "count",
                type_=astx.Int32(),
                value=astx.LiteralInt32(0),
                mutability=astx.MutabilityKind.mutable,
            ),
            outer_loop,
            astx.FunctionReturn(astx.Identifier("count")),
        )
    )

    assert_jit_int_main_result(builder, module, 12)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_count_uses_single_initializer_alloca(
    builder_class: type[Builder],
) -> None:
    """
    title: For-count lowering should not duplicate the initializer alloca.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.ForCountLoopStmt(
        initializer=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(2),
        ),
        update=astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
        body=astx.Block(),
    )
    module = build_int32_main_module(
        block_of(loop, astx.FunctionReturn(astx.LiteralInt32(0)))
    )

    ir_result = builder.translate(module)

    assert ir_result.count('%"i" = alloca i32') == 1
    assert ir_result.count('store i32 %"inctmp", i32* %"i"') == 1


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_for_count_zero_iterations_preserve_post_loop_state(
    builder_class: type[Builder],
) -> None:
    """
    title: A zero-trip for-count loop should leave post-loop state untouched.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop = astx.ForCountLoopStmt(
        initializer=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            value=astx.LiteralInt32(4),
            mutability=astx.MutabilityKind.mutable,
        ),
        condition=astx.BinaryOp(
            op_code="<",
            lhs=astx.Identifier("i"),
            rhs=astx.LiteralInt32(4),
        ),
        update=astx.UnaryOp(op_code="++", operand=astx.Identifier("i")),
        body=block_of(astx.VariableAssignment("result", astx.LiteralInt32(0))),
    )
    module = build_int32_main_module(
        block_of(
            astx.InlineVariableDeclaration(
                "result",
                type_=astx.Int32(),
                value=astx.LiteralInt32(41),
                mutability=astx.MutabilityKind.mutable,
            ),
            loop,
            astx.FunctionReturn(astx.Identifier("result")),
        )
    )

    assert_jit_int_main_result(builder, module, 41)

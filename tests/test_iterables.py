"""
title: Iterable semantic and lowering tests.
"""

from __future__ import annotations

import shutil

import pytest

from irx import astx
from irx.analysis import SemanticError, analyze
from irx.analysis.resolved_nodes import IterationKind
from irx.builder import Builder
from irx.diagnostics import LoweringError

from .conftest import assert_build_output, assert_ir_parses

HAS_CLANG = shutil.which("clang") is not None


def _block_of(*nodes: astx.AST) -> astx.Block:
    """
    title: Build one block from nodes.
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


def _main_module(*nodes: astx.AST) -> astx.Module:
    """
    title: Build one Int32 main module.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    module = astx.Module()
    body = astx.Block()
    for node in nodes:
        body.append(node)
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "main",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=body,
        )
    )
    return module


def _literal_list(*values: int) -> astx.LiteralList:
    """
    title: Build one Int32 literal list.
    parameters:
      values:
        type: int
        variadic: positional
    returns:
      type: astx.LiteralList
    """
    return astx.LiteralList([astx.LiteralInt32(value) for value in values])


def _sum_for_in_module() -> astx.Module:
    """
    title: Build one module that sums a literal list with for-in.
    returns:
      type: astx.Module
    """
    loop_body = _block_of(
        astx.VariableAssignment(
            "total",
            astx.BinaryOp(
                "+",
                astx.Identifier("total"),
                astx.Identifier("item"),
            ),
        )
    )
    loop = astx.ForInLoopStmt(
        astx.Identifier("item"),
        _literal_list(1, 2, 3),
        loop_body,
    )
    return _main_module(
        astx.VariableDeclaration(
            "total",
            astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
            value=astx.LiteralInt32(0),
        ),
        loop,
        astx.FunctionReturn(astx.Identifier("total")),
    )


def _filtered_comprehension_module() -> astx.Module:
    """
    title: Build one module that returns a filtered list-comprehension length.
    returns:
      type: astx.Module
    """
    comprehension = astx.ListComprehension(
        element=astx.BinaryOp(
            "*",
            astx.Identifier("item"),
            astx.LiteralInt32(2),
        ),
        generators=[
            astx.ComprehensionClause(
                astx.Identifier("item"),
                _literal_list(1, 2, 3),
                [
                    astx.BinaryOp(
                        ">",
                        astx.Identifier("item"),
                        astx.LiteralInt32(1),
                    )
                ],
            )
        ],
    )
    list_type = astx.ListType([astx.Int32()])
    return _main_module(
        astx.VariableDeclaration(
            "values",
            list_type,
            mutability=astx.MutabilityKind.mutable,
            value=comprehension,
        ),
        astx.FunctionReturn(astx.ListLength(astx.Identifier("values"))),
    )


def _dynamic_for_in_module() -> astx.Module:
    """
    title: Build one module that iterates a runtime list variable.
    returns:
      type: astx.Module
    """
    list_type = astx.ListType([astx.Int32()])
    loop_body = _block_of(
        astx.VariableAssignment(
            "total",
            astx.BinaryOp(
                "+",
                astx.Identifier("total"),
                astx.Identifier("item"),
            ),
        )
    )
    return _main_module(
        astx.VariableDeclaration(
            "values",
            list_type,
            mutability=astx.MutabilityKind.mutable,
            value=astx.ListCreate(astx.Int32()),
        ),
        astx.ListAppend(astx.Identifier("values"), astx.LiteralInt32(4)),
        astx.ListAppend(astx.Identifier("values"), astx.LiteralInt32(5)),
        astx.VariableDeclaration(
            "total",
            astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
            value=astx.LiteralInt32(0),
        ),
        astx.ForInLoopStmt(
            astx.Identifier("item"),
            astx.Identifier("values"),
            loop_body,
        ),
        astx.FunctionReturn(astx.Identifier("total")),
    )


def _nested_comprehension_module() -> astx.Module:
    """
    title: Build one module with two comprehension clauses.
    returns:
      type: astx.Module
    """
    comprehension = astx.ListComprehension(
        element=astx.BinaryOp(
            "+",
            astx.Identifier("left"),
            astx.Identifier("right"),
        ),
        generators=[
            astx.ComprehensionClause(
                astx.Identifier("left"),
                _literal_list(1, 2),
            ),
            astx.ComprehensionClause(
                astx.Identifier("right"),
                _literal_list(10, 20, 30),
                [
                    astx.BinaryOp(
                        ">",
                        astx.Identifier("right"),
                        astx.BinaryOp(
                            "*",
                            astx.Identifier("left"),
                            astx.LiteralInt32(10),
                        ),
                    )
                ],
            ),
        ],
    )
    list_type = astx.ListType([astx.Int32()])
    result_sum = astx.BinaryOp(
        "+",
        astx.ListIndex(astx.Identifier("values"), astx.LiteralInt32(0)),
        astx.BinaryOp(
            "+",
            astx.ListIndex(astx.Identifier("values"), astx.LiteralInt32(1)),
            astx.BinaryOp(
                "+",
                astx.ListIndex(
                    astx.Identifier("values"),
                    astx.LiteralInt32(2),
                ),
                astx.ListLength(astx.Identifier("values")),
            ),
        ),
    )
    return _main_module(
        astx.VariableDeclaration(
            "values",
            list_type,
            mutability=astx.MutabilityKind.mutable,
            value=comprehension,
        ),
        astx.FunctionReturn(result_sum),
    )


def test_for_in_resolves_list_iteration() -> None:
    """
    title: For-in analysis should attach list iteration metadata.
    """
    module = _sum_for_in_module()
    loop = module.nodes[0].body.nodes[1]

    analyze(module)

    iteration = loop.semantic.resolved_iteration
    target_symbol = loop.target.semantic.resolved_symbol
    assert iteration is not None
    assert target_symbol is not None
    assert iteration.kind is IterationKind.LIST
    assert isinstance(iteration.element_type, astx.Int32)
    assert iteration.target_symbol == target_symbol


def test_for_in_rejects_non_iterable() -> None:
    """
    title: For-in analysis should reject scalar iterables.
    """
    module = _main_module(
        astx.ForInLoopStmt(
            astx.Identifier("item"),
            astx.LiteralInt32(1),
            astx.Block(),
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    with pytest.raises(SemanticError, match="for-in requires an iterable"):
        analyze(module)


def test_for_in_target_does_not_leak() -> None:
    """
    title: For-in targets should stay scoped to the loop body.
    """
    module = _main_module(
        astx.ForInLoopStmt(
            astx.Identifier("item"),
            _literal_list(1),
            astx.Block(),
        ),
        astx.FunctionReturn(astx.Identifier("item")),
    )

    with pytest.raises(SemanticError, match="cannot resolve name 'item'"):
        analyze(module)


def test_list_comprehension_resolves_iteration_and_result_type() -> None:
    """
    title: List comprehensions should resolve clause and result types.
    """
    module = _filtered_comprehension_module()
    declaration = module.nodes[0].body.nodes[0]
    comprehension = declaration.value

    analyze(module)

    clause = comprehension.generators.nodes[0]
    assert clause.semantic.resolved_iteration is not None
    assert clause.semantic.resolved_iteration.kind is IterationKind.LIST
    assert isinstance(comprehension.semantic.resolved_type, astx.ListType)


def test_comprehension_filter_requires_boolean() -> None:
    """
    title: Comprehension filters should be Boolean expressions.
    """
    comprehension = astx.ListComprehension(
        element=astx.Identifier("item"),
        generators=[
            astx.ComprehensionClause(
                astx.Identifier("item"),
                _literal_list(1),
                [astx.LiteralInt32(1)],
            )
        ],
    )

    with pytest.raises(SemanticError, match="filter must be Boolean"):
        analyze(comprehension)


def test_set_and_dict_comprehensions_analyze_iterables() -> None:
    """
    title: Set and dict comprehensions should resolve semantic result types.
    """
    set_comprehension = astx.SetComprehension(
        element=astx.Identifier("item"),
        generators=[
            astx.ComprehensionClause(
                astx.Identifier("item"),
                astx.LiteralSet({astx.LiteralInt32(1), astx.LiteralInt32(2)}),
            )
        ],
    )
    dict_comprehension = astx.DictComprehension(
        key=astx.Identifier("key"),
        value=astx.Identifier("key"),
        generators=[
            astx.ComprehensionClause(
                astx.Identifier("key"),
                astx.LiteralDict({astx.LiteralInt32(1): astx.LiteralInt32(2)}),
            )
        ],
    )

    analyze(set_comprehension)
    analyze(dict_comprehension)

    set_semantic = getattr(set_comprehension, "semantic")
    dict_semantic = getattr(dict_comprehension, "semantic")
    assert isinstance(set_semantic.resolved_type, astx.SetType)
    assert isinstance(dict_semantic.resolved_type, astx.DictType)


def test_for_in_lowers_to_list_iteration_blocks() -> None:
    """
    title: For-in over lists should lower to explicit iteration blocks.
    """
    builder = Builder()
    ir_text = builder.translate(_sum_for_in_module())

    assert "for.in.cond" in ir_text
    assert "for.in.advance" in ir_text
    assert 'call i8* @"irx_list_at"' in ir_text
    assert_ir_parses(ir_text)


def test_dynamic_list_for_in_lowers() -> None:
    """
    title: For-in over runtime list variables should lower via list length.
    """
    builder = Builder()
    ir_text = builder.translate(_dynamic_for_in_module())

    assert "irx_list_iter_length_i64" in ir_text
    assert "for.in.cond" in ir_text
    assert 'call i8* @"irx_list_at"' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.skipif(
    not HAS_CLANG,
    reason="clang is required for runtime tests",
)
def test_for_in_list_builds_and_runs() -> None:
    """
    title: For-in over a list should execute with the list runtime.
    """
    expected_sum = 6
    assert_build_output(Builder(), _sum_for_in_module(), str(expected_sum))


@pytest.mark.skipif(
    not HAS_CLANG,
    reason="clang is required for runtime tests",
)
def test_dynamic_list_for_in_builds_and_runs() -> None:
    """
    title: For-in over runtime list variables should execute.
    """
    expected_sum = 9
    assert_build_output(
        Builder(),
        _dynamic_for_in_module(),
        str(expected_sum),
    )


def test_list_comprehension_lowers_to_dynamic_list_append() -> None:
    """
    title: List comprehensions should lower through dynamic list append.
    """
    builder = Builder()
    ir_text = builder.translate(_filtered_comprehension_module())

    assert "list.comp.0.cond" in ir_text
    assert 'call i32 @"irx_list_append"' in ir_text
    assert_ir_parses(ir_text)


def test_nested_list_comprehension_lowers() -> None:
    """
    title: Nested list comprehensions should lower nested clause blocks.
    """
    builder = Builder()
    ir_text = builder.translate(_nested_comprehension_module())

    assert "list.comp.0.cond" in ir_text
    assert "list.comp.1.cond" in ir_text
    assert "list.comp.1.filter.0.pass" in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.skipif(
    not HAS_CLANG,
    reason="clang is required for runtime tests",
)
def test_list_comprehension_builds_and_runs() -> None:
    """
    title: Filtered list comprehension should build and execute.
    """
    expected_length = 2
    assert_build_output(
        Builder(),
        _filtered_comprehension_module(),
        str(expected_length),
    )


@pytest.mark.skipif(
    not HAS_CLANG,
    reason="clang is required for runtime tests",
)
def test_nested_list_comprehension_builds_and_runs() -> None:
    """
    title: Nested list comprehensions should preserve loop order.
    """
    expected_first = 21
    expected_second = 31
    expected_third = 32
    expected_length = 3
    expected_result = (
        expected_first + expected_second + expected_third + expected_length
    )
    assert_build_output(
        Builder(),
        _nested_comprehension_module(),
        str(expected_result),
    )


def test_set_comprehension_lowering_is_guarded() -> None:
    """
    title: Set comprehension lowering should report the runtime gap clearly.
    """
    module = _main_module(
        astx.SetComprehension(
            element=astx.Identifier("item"),
            generators=[
                astx.ComprehensionClause(
                    astx.Identifier("item"),
                    astx.LiteralSet({astx.LiteralInt32(1)}),
                )
            ],
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )

    with pytest.raises(LoweringError, match="dynamic set runtime"):
        Builder().translate(module)

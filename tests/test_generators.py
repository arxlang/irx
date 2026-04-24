"""
title: Generator semantic and lowering tests.
"""

from __future__ import annotations

import shutil

import pytest

from irx import astx
from irx.analysis import SemanticError, analyze
from irx.analysis.resolved_nodes import IterationKind
from irx.builder import Builder

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


def _numbers_function() -> astx.FunctionDef:
    """
    title: Build a generator function that yields three integers.
    returns:
      type: astx.FunctionDef
    """
    return astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "numbers",
            args=astx.Arguments(),
            return_type=astx.GeneratorType(astx.Int32()),
        ),
        body=_block_of(
            astx.VariableDeclaration(
                "current",
                astx.Int32(),
                mutability=astx.MutabilityKind.mutable,
                value=astx.LiteralInt32(1),
            ),
            astx.YieldStmt(astx.Identifier("current")),
            astx.VariableAssignment("current", astx.LiteralInt32(2)),
            astx.YieldStmt(astx.Identifier("current")),
            astx.VariableAssignment("current", astx.LiteralInt32(3)),
            astx.YieldStmt(astx.Identifier("current")),
        ),
    )


def _sum_generator_module() -> astx.Module:
    """
    title: Build one module that sums values yielded by a generator.
    returns:
      type: astx.Module
    """
    module = astx.Module()
    module.block.append(_numbers_function())

    loop = astx.ForInLoopStmt(
        astx.Identifier("item"),
        astx.FunctionCall("numbers", []),
        _block_of(
            astx.VariableAssignment(
                "total",
                astx.BinaryOp(
                    "+",
                    astx.Identifier("total"),
                    astx.Identifier("item"),
                ),
            )
        ),
    )
    main_body = _block_of(
        astx.VariableDeclaration(
            "total",
            astx.Int32(),
            mutability=astx.MutabilityKind.mutable,
            value=astx.LiteralInt32(0),
        ),
        loop,
        astx.FunctionReturn(astx.Identifier("total")),
    )
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "main",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=main_body,
        )
    )
    return module


def test_generator_iteration_semantics() -> None:
    """
    title: Generator values should resolve as generator iterables.
    """
    module = analyze(_sum_generator_module())
    assert isinstance(module, astx.Module)
    main_function = module.nodes[1]
    assert isinstance(main_function, astx.FunctionDef)
    loop = main_function.body.nodes[1]
    assert isinstance(loop, astx.ForInLoopStmt)

    semantic = getattr(loop, "semantic", None)
    iteration = getattr(semantic, "resolved_iteration", None)
    assert iteration is not None
    assert iteration.kind is IterationKind.GENERATOR
    assert isinstance(iteration.element_type, astx.Int32)


def test_yield_requires_generator_return_type() -> None:
    """
    title: A function with yield must declare GeneratorType.
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "bad",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block_of(astx.YieldStmt(astx.LiteralInt32(1))),
        )
    )
    with pytest.raises(SemanticError):
        analyze(module)


def test_generator_ir_parses() -> None:
    """
    title: Generator lowering should emit parseable LLVM IR.
    """
    ir_text = Builder().translate(_sum_generator_module())
    assert "numbers.__resume" in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.skipif(not HAS_CLANG, reason="clang is required for build tests")
def test_generator_executes() -> None:
    """
    title: A for-in loop should consume yielded generator values.
    """
    expected_sum = 6
    assert_build_output(Builder(), _sum_generator_module(), str(expected_sum))

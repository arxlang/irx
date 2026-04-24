"""
title: Common collection method tests.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from pathlib import Path

import pytest

from irx import astx
from irx.analysis import SemanticError, analyze
from irx.builder import Builder
from irx.builder.base import CommandResult

from .conftest import (
    assert_ir_parses,
    assert_jit_int_main_result,
    make_main_module,
)

HAS_CLANG = shutil.which("clang") is not None


def _list_i32_type() -> astx.ListType:
    """
    title: Return the canonical List[Int32] type.
    returns:
      type: astx.ListType
    """
    return astx.ListType([astx.Int32()])


def _mutable_decl(
    name: str,
    type_: astx.DataType,
    value: astx.AST,
) -> astx.VariableDeclaration:
    """
    title: Build one mutable variable declaration.
    parameters:
      name:
        type: str
      type_:
        type: astx.DataType
      value:
        type: astx.AST
    returns:
      type: astx.VariableDeclaration
    """
    return astx.VariableDeclaration(
        name=name,
        type_=type_,
        value=value,
        mutability=astx.MutabilityKind.mutable,
    )


def _run_workspace_build(
    builder: Builder,
    module: astx.Module,
) -> CommandResult:
    """
    title: Build and run one module using a workspace-local temporary path.
    parameters:
      builder:
        type: Builder
      module:
        type: astx.Module
    returns:
      type: CommandResult
    """
    temp_root = (Path.cwd() / "tmp").resolve()
    temp_root.mkdir(exist_ok=True)
    original_tmpdir = os.environ.get("TMPDIR")
    output_path = ""
    try:
        os.environ["TMPDIR"] = str(temp_root)
        with tempfile.NamedTemporaryFile(
            suffix=".exe",
            prefix="irx_collection_methods_",
            dir=temp_root,
            delete=False,
        ) as handle:
            output_path = handle.name

        builder.build(module, output_file=output_path)
        return builder.run(raise_on_error=False)
    finally:
        if original_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = original_tmpdir
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)


def test_literal_collection_lengths() -> None:
    """
    title: Literal collection length queries should lower as Int32 values.
    """
    builder = Builder()
    expected_total = 8
    module = make_main_module(
        astx.FunctionReturn(
            astx.BinaryOp(
                "+",
                astx.CollectionLength(
                    astx.LiteralList([astx.LiteralInt32(1)])
                ),
                astx.BinaryOp(
                    "+",
                    astx.CollectionLength(
                        astx.LiteralTuple(
                            (
                                astx.LiteralInt32(1),
                                astx.LiteralInt32(2),
                            )
                        )
                    ),
                    astx.BinaryOp(
                        "+",
                        astx.CollectionLength(
                            astx.LiteralSet(
                                {
                                    astx.LiteralInt32(1),
                                    astx.LiteralInt32(2),
                                }
                            )
                        ),
                        astx.CollectionLength(
                            astx.LiteralDict(
                                {
                                    astx.LiteralInt32(1): astx.LiteralInt32(2),
                                    astx.LiteralInt32(3): astx.LiteralInt32(4),
                                    astx.LiteralInt32(5): astx.LiteralInt32(6),
                                }
                            )
                        ),
                    ),
                ),
            )
        )
    )

    ir_text = builder.translate(module)

    assert_ir_parses(ir_text)
    assert_jit_int_main_result(builder, module, expected_total)


@pytest.mark.parametrize(
    ("base", "value"),
    [
        (
            astx.LiteralList([astx.LiteralInt32(7)]),
            astx.LiteralInt32(7),
        ),
        (
            astx.LiteralTuple((astx.LiteralInt32(7),)),
            astx.LiteralInt32(7),
        ),
        (
            astx.LiteralSet({astx.LiteralInt32(7)}),
            astx.LiteralInt32(7),
        ),
        (
            astx.LiteralDict({astx.LiteralInt32(7): astx.LiteralInt32(10)}),
            astx.LiteralInt32(7),
        ),
    ],
)
def test_literal_collection_contains(
    base: astx.AST,
    value: astx.AST,
) -> None:
    """
    title: Literal collection containment should support common collections.
    parameters:
      base:
        type: astx.AST
      value:
        type: astx.AST
    """
    builder = Builder()
    module = make_main_module(
        astx.FunctionReturn(astx.CollectionContains(base, value)),
        return_type=astx.Boolean(),
    )

    ir_text = builder.translate(module)

    assert 'define i1 @"main__main_body"' in ir_text
    assert "collection_contains" in ir_text
    assert_ir_parses(ir_text)


def test_collection_is_empty() -> None:
    """
    title: Collection emptiness should lower to a Boolean value.
    """
    builder = Builder()
    module = make_main_module(
        astx.FunctionReturn(
            astx.CollectionIsEmpty(astx.LiteralTuple(tuple()))
        ),
        return_type=astx.Boolean(),
    )

    ir_text = builder.translate(module)

    assert "collection_is_empty" in ir_text
    assert_ir_parses(ir_text)


def test_heterogeneous_tuple_contains_skips_incompatible_entries() -> None:
    """
    title: Heterogeneous tuple containment should skip incompatible members.
    """
    builder = Builder()
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module = make_main_module(
        astx.IfStmt(
            condition=astx.CollectionContains(
                astx.LiteralTuple(
                    (
                        astx.LiteralInt32(1),
                        astx.LiteralString("x"),
                    )
                ),
                astx.LiteralInt32(1),
            ),
            then=then_block,
            else_=else_block,
        )
    )

    assert_jit_int_main_result(builder, module, 1)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (
            astx.CollectionIndex(
                astx.LiteralTuple(
                    (
                        astx.LiteralInt32(4),
                        astx.LiteralInt32(9),
                        astx.LiteralInt32(4),
                    )
                ),
                astx.LiteralInt32(9),
            ),
            1,
        ),
        (
            astx.CollectionCount(
                astx.LiteralList(
                    [
                        astx.LiteralInt32(4),
                        astx.LiteralInt32(9),
                        astx.LiteralInt32(4),
                    ]
                ),
                astx.LiteralInt32(4),
            ),
            2,
        ),
        (
            astx.CollectionIndex(
                astx.LiteralTuple(
                    (
                        astx.LiteralString("x"),
                        astx.LiteralInt32(1),
                    )
                ),
                astx.LiteralInt32(1),
            ),
            1,
        ),
        (
            astx.CollectionCount(
                astx.LiteralTuple(
                    (
                        astx.LiteralInt32(1),
                        astx.LiteralString("x"),
                        astx.LiteralInt32(1),
                    )
                ),
                astx.LiteralInt32(1),
            ),
            2,
        ),
    ],
)
def test_literal_sequence_search(expression: astx.AST, expected: int) -> None:
    """
    title: Literal sequence index and count should return Int32 results.
    parameters:
      expression:
        type: astx.AST
      expected:
        type: int
    """
    builder = Builder()
    module = make_main_module(astx.FunctionReturn(expression))

    assert_jit_int_main_result(builder, module, expected)


def test_dynamic_list_search_lowers_to_runtime_loop() -> None:
    """
    title: Dynamic list contains, index, and count should lower through loops.
    """
    builder = Builder()
    list_type = _list_i32_type()
    module = make_main_module(
        _mutable_decl("out", list_type, astx.ListCreate(astx.Int32())),
        astx.ListAppend(astx.Identifier("out"), astx.LiteralInt32(4)),
        astx.ListAppend(astx.Identifier("out"), astx.LiteralInt32(9)),
        astx.FunctionReturn(
            astx.BinaryOp(
                "+",
                astx.CollectionIndex(
                    astx.Identifier("out"),
                    astx.LiteralInt32(9),
                ),
                astx.CollectionCount(
                    astx.Identifier("out"),
                    astx.LiteralInt32(4),
                ),
            )
        ),
    )

    ir_text = builder.translate(module)

    assert 'call i8* @"irx_list_at"' in ir_text
    assert "collection.search.cond" in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.skipif(not HAS_CLANG, reason="clang is required for build tests")
def test_dynamic_list_contains_executes() -> None:
    """
    title: Dynamic list contains should execute through the runtime loop.
    """
    builder = Builder()
    list_type = _list_i32_type()
    then_block = astx.Block()
    then_block.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    else_block = astx.Block()
    else_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module = make_main_module(
        _mutable_decl("out", list_type, astx.ListCreate(astx.Int32())),
        astx.ListAppend(astx.Identifier("out"), astx.LiteralInt32(4)),
        astx.ListAppend(astx.Identifier("out"), astx.LiteralInt32(9)),
        astx.IfStmt(
            condition=astx.CollectionContains(
                astx.Identifier("out"),
                astx.LiteralInt32(9),
            ),
            then=then_block,
            else_=else_block,
        ),
    )

    result = _run_workspace_build(builder, module)
    expected_returncode = 1

    assert result.returncode == expected_returncode


def test_collection_contains_rejects_wrong_probe_type() -> None:
    """
    title: Collection containment should reject incompatible probe types.
    """
    module = make_main_module(
        astx.FunctionReturn(
            astx.CollectionContains(
                astx.LiteralList([astx.LiteralInt32(1)]),
                astx.LiteralFloat32(1.5),
            )
        ),
        return_type=astx.Boolean(),
    )

    with pytest.raises(SemanticError, match="containment probe expects"):
        analyze(module)


def test_collection_count_rejects_dict() -> None:
    """
    title: Sequence-only collection methods should reject dictionaries.
    """
    module = make_main_module(
        astx.FunctionReturn(
            astx.CollectionCount(
                astx.LiteralDict({astx.LiteralInt32(1): astx.LiteralInt32(2)}),
                astx.LiteralInt32(1),
            )
        )
    )

    with pytest.raises(SemanticError, match="requires a list or tuple"):
        analyze(module)


@pytest.mark.parametrize(
    ("type_", "value", "expression", "return_type"),
    [
        (
            astx.TupleType([astx.Int32(), astx.Int32()]),
            astx.LiteralTuple((astx.LiteralInt32(1), astx.LiteralInt32(2))),
            astx.CollectionContains(
                astx.Identifier("items"),
                astx.LiteralInt32(1),
            ),
            astx.Boolean(),
        ),
        (
            astx.TupleType([astx.Int32(), astx.Int32()]),
            astx.LiteralTuple((astx.LiteralInt32(1), astx.LiteralInt32(2))),
            astx.CollectionIndex(
                astx.Identifier("items"),
                astx.LiteralInt32(1),
            ),
            astx.Int32(),
        ),
        (
            astx.SetType(astx.Int32()),
            astx.LiteralSet({astx.LiteralInt32(1)}),
            astx.CollectionLength(astx.Identifier("items")),
            astx.Int32(),
        ),
        (
            astx.DictType(astx.Int32(), astx.Int32()),
            astx.LiteralDict({astx.LiteralInt32(1): astx.LiteralInt32(2)}),
            astx.CollectionContains(
                astx.Identifier("items"),
                astx.LiteralInt32(1),
            ),
            astx.Boolean(),
        ),
    ],
)
def test_nonlowerable_collection_forms_reject_semantically(
    type_: astx.DataType,
    value: astx.AST,
    expression: astx.AST,
    return_type: astx.DataType,
) -> None:
    """
    title: Unsupported collection receiver forms should fail in analysis.
    parameters:
      type_:
        type: astx.DataType
      value:
        type: astx.AST
      expression:
        type: astx.AST
      return_type:
        type: astx.DataType
    """
    module = make_main_module(
        _mutable_decl("items", type_, value),
        astx.FunctionReturn(expression),
        return_type=return_type,
    )

    with pytest.raises(SemanticError, match="currently supports"):
        analyze(module)

"""
title: Common collection method tests.
"""

from __future__ import annotations

import pytest

from irx import astx
from irx.analysis import SemanticError, analyze
from irx.builder import Builder

from .conftest import (
    assert_ir_parses,
    assert_jit_int_main_result,
    make_main_module,
)


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

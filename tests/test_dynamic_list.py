"""
title: Dynamic-list construction and indexing tests.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from pathlib import Path

import pytest

from irx import astx
from irx.analysis import SemanticError
from irx.builder import Builder

from .conftest import assert_ir_parses

HAS_CLANG = shutil.which("clang") is not None
EXPECTED_LIST_AT_CALLS = 3


def _list_i32_type() -> astx.ListType:
    """
    title: Return the canonical list[Int32] test type.
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
    title: Build one mutable local variable declaration.
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
        mutability=astx.MutabilityKind.mutable,
        value=value,
    )


def _index(base: astx.AST, index: int) -> astx.SubscriptExpr:
    """
    title: Build one integer list index expression.
    parameters:
      base:
        type: astx.AST
      index:
        type: int
    returns:
      type: astx.SubscriptExpr
    """
    return astx.SubscriptExpr(base, astx.LiteralInt32(index))


def _assert_workspace_build_output(
    builder: Builder,
    module: astx.Module,
    expected_output: str,
) -> None:
    """
    title: Build and run one module using a workspace-local temporary path.
    parameters:
      builder:
        type: Builder
      module:
        type: astx.Module
      expected_output:
        type: str
    """
    temp_root = (Path.cwd() / "tmp").resolve()
    temp_root.mkdir(exist_ok=True)
    original_tmpdir = os.environ.get("TMPDIR")
    output_path = ""
    try:
        os.environ["TMPDIR"] = str(temp_root)
        with tempfile.NamedTemporaryFile(
            suffix=".exe",
            prefix="irx_dynamic_list_",
            dir=temp_root,
            delete=False,
        ) as handle:
            output_path = handle.name

        builder.build(module, output_file=output_path)
        result = builder.run(raise_on_error=False)
        actual_output = result.stdout.strip() or str(result.returncode)
        assert actual_output == expected_output, (
            f"Expected `{expected_output}`, but got `{actual_output}` "
            f"(stderr={result.stderr.strip()!r})"
        )
    finally:
        if original_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = original_tmpdir
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)


def _singleton_module() -> astx.Module:
    """
    title: Build one module that appends a variable value into a list.
    returns:
      type: astx.Module
    """
    list_type = _list_i32_type()
    module = astx.Module()

    singleton = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "singleton",
            args=astx.Arguments(astx.Argument("value", astx.Int32())),
            return_type=list_type,
        ),
        body=astx.Block(),
    )
    singleton.body.append(
        _mutable_decl(
            "out",
            list_type,
            astx.ListCreate(astx.Int32()),
        )
    )
    singleton.body.append(
        astx.ListAppend(astx.Identifier("out"), astx.Identifier("value"))
    )
    singleton.body.append(astx.FunctionReturn(astx.Identifier("out")))
    module.block.append(singleton)

    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(
        _mutable_decl(
            "vals",
            list_type,
            astx.FunctionCall("singleton", [astx.LiteralInt32(7)]),
        )
    )
    main.body.append(
        _mutable_decl(
            "first", astx.Int32(), _index(astx.Identifier("vals"), 0)
        )
    )
    main.body.append(astx.FunctionReturn(astx.Identifier("first")))
    module.block.append(main)
    return module


def _loop_module() -> astx.Module:
    """
    title: Build one module that appends into a list inside a while loop.
    returns:
      type: astx.Module
    """
    list_type = _list_i32_type()
    module = astx.Module()

    make_list = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "make_list",
            args=astx.Arguments(),
            return_type=list_type,
        ),
        body=astx.Block(),
    )
    make_list.body.append(
        _mutable_decl("out", list_type, astx.ListCreate(astx.Int32()))
    )
    make_list.body.append(
        _mutable_decl("current", astx.Int32(), astx.LiteralInt32(1))
    )

    loop_body = astx.Block()
    loop_body.append(
        astx.ListAppend(astx.Identifier("out"), astx.Identifier("current"))
    )
    loop_body.append(
        astx.VariableAssignment(
            "current",
            astx.BinaryOp(
                "+",
                astx.Identifier("current"),
                astx.LiteralInt32(1),
            ),
        )
    )
    make_list.body.append(
        astx.WhileStmt(
            astx.BinaryOp(
                "<",
                astx.Identifier("current"),
                astx.LiteralInt32(4),
            ),
            loop_body,
        )
    )
    make_list.body.append(astx.FunctionReturn(astx.Identifier("out")))
    module.block.append(make_list)

    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(
        _mutable_decl(
            "vals",
            list_type,
            astx.FunctionCall("make_list", []),
        )
    )
    main.body.append(
        _mutable_decl(
            "first", astx.Int32(), _index(astx.Identifier("vals"), 0)
        )
    )
    main.body.append(
        _mutable_decl(
            "second",
            astx.Int32(),
            _index(astx.Identifier("vals"), 1),
        )
    )
    main.body.append(
        _mutable_decl(
            "third", astx.Int32(), _index(astx.Identifier("vals"), 2)
        )
    )
    sum_expr = astx.BinaryOp(
        "+",
        astx.Identifier("first"),
        astx.BinaryOp(
            "+",
            astx.Identifier("second"),
            astx.Identifier("third"),
        ),
    )
    main.body.append(astx.FunctionReturn(sum_expr))
    module.block.append(main)
    return module


def _uninitialized_local_module() -> astx.Module:
    """
    title: >-
      Build one module that appends after an uninitialized list declaration.
    returns:
      type: astx.Module
    """
    list_type = _list_i32_type()
    module = astx.Module()

    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(
        astx.VariableDeclaration(
            name="out",
            type_=list_type,
            mutability=astx.MutabilityKind.mutable,
        )
    )
    main.body.append(
        astx.ListAppend(astx.Identifier("out"), astx.LiteralInt32(11))
    )
    main.body.append(
        _mutable_decl(
            "first",
            astx.Int32(),
            _index(astx.Identifier("out"), 0),
        )
    )
    main.body.append(astx.FunctionReturn(astx.Identifier("first")))
    module.block.append(main)
    return module


def test_dynamic_list_appends_variable_values() -> None:
    """
    title: Dynamic list creation should accept appended variable values.
    """
    builder = Builder()
    ir_text = builder.translate(_singleton_module())

    assert 'call i32 @"irx_list_append"' in ir_text
    assert 'call i8* @"irx_list_at"' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.skipif(not HAS_CLANG, reason="clang is required for build tests")
def test_dynamic_list_loop_build_and_return() -> None:
    """
    title: A function should append in a loop, return the list, and index it.
    """
    builder = Builder()
    module = _loop_module()
    ir_text = builder.translate(module)

    assert 'call i32 @"irx_list_append"' in ir_text
    assert ir_text.count('call i8* @"irx_list_at"') >= EXPECTED_LIST_AT_CALLS
    assert_ir_parses(ir_text)

    EXPECTED_LOOP_SUM = 6
    _assert_workspace_build_output(builder, module, str(EXPECTED_LOOP_SUM))


@pytest.mark.skipif(not HAS_CLANG, reason="clang is required for build tests")
def test_dynamic_list_uninitialized_local_build_and_append() -> None:
    """
    title: Uninitialized mutable list locals should still append correctly.
    """
    builder = Builder()
    module = _uninitialized_local_module()
    ir_text = builder.translate(module)

    assert 'call i32 @"irx_list_append"' in ir_text
    assert 'call i8* @"irx_list_at"' in ir_text
    assert_ir_parses(ir_text)

    EXPECTED_FIRST_VALUE = 11
    _assert_workspace_build_output(builder, module, str(EXPECTED_FIRST_VALUE))


def test_dynamic_list_append_rejects_type_mismatch() -> None:
    """
    title: Dynamic list append should reject incompatible element values.
    """
    list_type = _list_i32_type()
    module = astx.Module()
    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    main.body.append(
        _mutable_decl("out", list_type, astx.ListCreate(astx.Int32()))
    )
    main.body.append(
        astx.ListAppend(astx.Identifier("out"), astx.LiteralFloat32(1.5))
    )
    main.body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(main)

    with pytest.raises(SemanticError, match="cannot assign Float32"):
        Builder().translate(module)

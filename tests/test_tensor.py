"""
title: Tests for the IRx Tensor layer.
"""

from __future__ import annotations

import shutil

from typing import Any, cast

import pytest

from irx import astx
from irx.analysis import SemanticError, analyze
from irx.buffer import buffer_dtype_handle
from irx.builder import Builder
from irx.builtins.collections.tensor import (
    tensor_element_size_bytes_from_dtype,
)

from tests.conftest import assert_ir_parses, build_and_run


def _module_with_main(*nodes: astx.AST) -> astx.Module:
    """
    title: Build an int32 main module.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    module = astx.Module()
    prototype = astx.FunctionPrototype(
        "main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    body = astx.Block()
    for node in nodes:
        body.append(node)
    if not any(isinstance(node, astx.FunctionReturn) for node in nodes):
        body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=prototype, body=body))
    return module


def _int32_tensor(
    values: list[int],
    *,
    shape: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
    offset_bytes: int = 0,
) -> astx.TensorLiteral:
    """
    title: Build one int32 Tensor literal node.
    parameters:
      values:
        type: list[int]
      shape:
        type: tuple[int, Ellipsis]
      strides:
        type: tuple[int, Ellipsis] | None
      offset_bytes:
        type: int
    returns:
      type: astx.TensorLiteral
    """
    return astx.TensorLiteral(
        [astx.LiteralInt32(value) for value in values],
        element_type=astx.Int32(),
        shape=shape,
        strides=strides,
        offset_bytes=offset_bytes,
    )


def test_tensor_literal_get_struct_shapes() -> None:
    """
    title: Tensor literal get_struct should expose shape and stride metadata.
    """
    node = _int32_tensor([1, 2, 3, 4], shape=(2, 2), strides=(8, 4))

    full = node.get_struct()
    assert isinstance(full, dict)
    full_entry = cast(dict[str, Any], full["TensorLiteral"])
    entry = cast(dict[str, Any], full_entry["content"])
    assert entry["shape"] == [2, 2]
    assert entry["strides"] == [8, 4]
    assert entry["offset_bytes"] == 0

    simplified = node.get_struct(simplified=True)
    assert isinstance(simplified, dict)
    simplified_entry = cast(dict[str, Any], simplified["TensorLiteral"])
    assert simplified_entry["shape"] == [2, 2]


def test_tensor_element_size_uses_shared_dtype_metadata() -> None:
    """
    title: Tensor dtype sizes should come from shared primitive metadata.
    """
    int32_dtype = buffer_dtype_handle("int32")
    float64_dtype = buffer_dtype_handle("float64")
    bool_dtype = buffer_dtype_handle("bool")
    int32_size = 4
    float64_size = 8

    assert tensor_element_size_bytes_from_dtype(int32_dtype) == int32_size
    assert tensor_element_size_bytes_from_dtype(float64_dtype) == float64_size
    assert tensor_element_size_bytes_from_dtype(bool_dtype) is None


def test_tensor_rejects_bool_elements() -> None:
    """
    title: Bool Tensors should fail semantic analysis in this phase.
    """
    module = _module_with_main(
        astx.FunctionReturn(
            astx.TensorNDim(
                astx.TensorLiteral(
                    [astx.LiteralBoolean(True), astx.LiteralBoolean(False)],
                    element_type=astx.Boolean(),
                    shape=(2,),
                )
            )
        )
    )

    with pytest.raises(SemanticError, match="bool tensors are not supported"):
        analyze(module)


def test_tensor_rejects_wrong_static_rank_index_count() -> None:
    """
    title: Tensor indexing should validate static rank against index arity.
    """
    module = _module_with_main(
        astx.FunctionReturn(
            astx.TensorIndex(
                _int32_tensor([1, 2, 3, 4], shape=(2, 2)),
                [astx.LiteralInt32(0)],
            )
        )
    )

    with pytest.raises(SemanticError, match="index count must match"):
        analyze(module)


def test_tensor_store_rejects_arrow_backed_readonly_values() -> None:
    """
    title: Arrow-backed Tensor literals should remain readonly in this phase.
    """
    module = _module_with_main(
        astx.TensorStore(
            _int32_tensor([1, 2, 3, 4], shape=(2, 2)),
            [astx.LiteralInt32(0), astx.LiteralInt32(1)],
            astx.LiteralInt32(99),
        )
    )

    with pytest.raises(SemanticError, match="readonly tensor view"):
        analyze(module)


def test_tensor_literal_lowers_through_tensor_runtime_and_owner_bridge() -> (
    None
):
    """
    title: >-
      Tensor literals should use the Arrow tensor runtime plus an owner bridge.
    """
    builder = Builder()
    module = _module_with_main(
        astx.VariableDeclaration(
            name="tensor",
            type_=astx.TensorType(astx.Int32()),
            mutability=astx.MutabilityKind.mutable,
            value=_int32_tensor([1, 2, 3, 4], shape=(2, 2)),
        ),
        astx.FunctionReturn(
            astx.TensorIndex(
                astx.Identifier("tensor"),
                [astx.LiteralInt32(1), astx.LiteralInt32(1)],
            )
        ),
    )

    ir_text = builder.translate(module)

    assert '@"irx_arrow_tensor_builder_new"' in ir_text
    assert '@"irx_arrow_tensor_borrow_buffer_view"' in ir_text
    assert '@"irx_buffer_owner_external_new"' in ir_text
    assert "irx_tensor_shape" in ir_text
    assert "irx_tensor_stride" in ir_text
    assert "irx_tensor_index_load" in ir_text
    assert_ir_parses(ir_text)


def test_tensor_view_lowers_custom_shape_stride_and_offset() -> None:
    """
    title: Tensor views should lower with explicit shape, stride, and offset.
    """
    builder = Builder()
    view = astx.TensorView(
        _int32_tensor([10, 20, 30, 40, 50, 60], shape=(2, 3)),
        shape=(2, 2),
        strides=(12, 4),
        offset_bytes=4,
    )
    module = _module_with_main(
        astx.FunctionReturn(
            astx.TensorIndex(
                view,
                [astx.LiteralInt32(1), astx.LiteralInt32(0)],
            )
        )
    )

    ir_text = builder.translate(module)

    assert "irx_tensor_shape_ptr" in ir_text or "irx_tensor_shape_" in ir_text
    assert (
        "irx_tensor_stride_ptr" in ir_text or "irx_tensor_strides_" in ir_text
    )
    assert "irx_buffer_index_stride_0" in ir_text
    assert "irx_buffer_index_stride_1" in ir_text
    assert (
        "irx_tensor_offset_bytes" in ir_text
        or "irx_buffer_index_offset_1" in ir_text
    )
    assert_ir_parses(ir_text)


def test_tensor_queries_lower_to_shape_stride_metadata() -> None:
    """
    title: Tensor queries should lower to rank, shape, and stride metadata.
    """
    builder = Builder()
    module = _module_with_main(
        astx.VariableDeclaration(
            name="tensor",
            type_=astx.TensorType(astx.Int32()),
            mutability=astx.MutabilityKind.mutable,
            value=_int32_tensor([1, 2, 3, 4], shape=(2, 2)),
        ),
        astx.FunctionReturn(astx.TensorNDim(astx.Identifier("tensor"))),
    )

    ir_text = builder.translate(module)

    assert "irx_tensor_ndim" in ir_text
    assert_ir_parses(ir_text)


def test_tensor_build_returns_indexed_element() -> None:
    """
    title: Built Tensor programs should return indexed element values.
    """
    if shutil.which("clang") is None:
        pytest.skip("builder.build() currently requires clang")

    module = _module_with_main(
        astx.FunctionReturn(
            astx.TensorIndex(
                _int32_tensor([1, 2, 3, 4, 5, 6], shape=(2, 3)),
                [astx.LiteralInt32(1), astx.LiteralInt32(2)],
            )
        )
    )

    result = build_and_run(Builder(), module)

    expected = 6
    assert result.returncode == expected, result.stderr or result.stdout


def test_tensor_view_build_returns_view_element() -> None:
    """
    title: Built Tensor views should return values through view metadata.
    """
    if shutil.which("clang") is None:
        pytest.skip("builder.build() currently requires clang")

    view = astx.TensorView(
        _int32_tensor([10, 20, 30, 40, 50, 60], shape=(2, 3)),
        shape=(2, 2),
        strides=(12, 4),
        offset_bytes=4,
    )
    module = _module_with_main(
        astx.FunctionReturn(
            astx.TensorIndex(
                view,
                [astx.LiteralInt32(1), astx.LiteralInt32(0)],
            )
        )
    )

    result = build_and_run(Builder(), module)

    expected = 50
    assert result.returncode == expected, result.stderr or result.stdout

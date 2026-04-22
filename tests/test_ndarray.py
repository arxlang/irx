"""
title: Tests for the IRx NDArray layer.
"""

from __future__ import annotations

import shutil

from typing import Any, cast

import pytest

from irx import astx
from irx.analysis import SemanticError, analyze
from irx.array import ndarray_element_size_bytes_from_dtype
from irx.buffer import buffer_dtype_handle
from irx.builder import Builder

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


def _int32_ndarray(
    values: list[int],
    *,
    shape: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
    offset_bytes: int = 0,
) -> astx.NDArrayLiteral:
    """
    title: Build one int32 NDArray literal node.
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
      type: astx.NDArrayLiteral
    """
    return astx.NDArrayLiteral(
        [astx.LiteralInt32(value) for value in values],
        element_type=astx.Int32(),
        shape=shape,
        strides=strides,
        offset_bytes=offset_bytes,
    )


def test_ndarray_literal_get_struct_shapes() -> None:
    """
    title: NDArray literal get_struct should expose shape and stride metadata.
    """
    node = _int32_ndarray([1, 2, 3, 4], shape=(2, 2), strides=(8, 4))

    full = node.get_struct()
    assert isinstance(full, dict)
    full_entry = cast(dict[str, Any], full["NDArrayLiteral"])
    entry = cast(dict[str, Any], full_entry["content"])
    assert entry["shape"] == [2, 2]
    assert entry["strides"] == [8, 4]
    assert entry["offset_bytes"] == 0

    simplified = node.get_struct(simplified=True)
    assert isinstance(simplified, dict)
    simplified_entry = cast(dict[str, Any], simplified["NDArrayLiteral"])
    assert simplified_entry["shape"] == [2, 2]


def test_ndarray_element_size_uses_shared_dtype_metadata() -> None:
    """
    title: NDArray dtype sizes should come from shared primitive metadata.
    """
    int32_dtype = buffer_dtype_handle("int32")
    float64_dtype = buffer_dtype_handle("float64")
    bool_dtype = buffer_dtype_handle("bool")
    int32_size = 4
    float64_size = 8

    assert ndarray_element_size_bytes_from_dtype(int32_dtype) == int32_size
    assert ndarray_element_size_bytes_from_dtype(float64_dtype) == float64_size
    assert ndarray_element_size_bytes_from_dtype(bool_dtype) is None


def test_ndarray_rejects_bool_elements() -> None:
    """
    title: Bool NDArrays should fail semantic analysis in this phase.
    """
    module = _module_with_main(
        astx.FunctionReturn(
            astx.NDArrayNDim(
                astx.NDArrayLiteral(
                    [astx.LiteralBoolean(True), astx.LiteralBoolean(False)],
                    element_type=astx.Boolean(),
                    shape=(2,),
                )
            )
        )
    )

    with pytest.raises(SemanticError, match="bool ndarrays are not supported"):
        analyze(module)


def test_ndarray_rejects_wrong_static_rank_index_count() -> None:
    """
    title: NDArray indexing should validate static rank against index arity.
    """
    module = _module_with_main(
        astx.FunctionReturn(
            astx.NDArrayIndex(
                _int32_ndarray([1, 2, 3, 4], shape=(2, 2)),
                [astx.LiteralInt32(0)],
            )
        )
    )

    with pytest.raises(SemanticError, match="index count must match"):
        analyze(module)


def test_ndarray_store_rejects_arrow_backed_readonly_values() -> None:
    """
    title: Arrow-backed NDArray literals should remain readonly in this phase.
    """
    module = _module_with_main(
        astx.NDArrayStore(
            _int32_ndarray([1, 2, 3, 4], shape=(2, 2)),
            [astx.LiteralInt32(0), astx.LiteralInt32(1)],
            astx.LiteralInt32(99),
        )
    )

    with pytest.raises(SemanticError, match="readonly ndarray view"):
        analyze(module)


def test_ndarray_literal_lowers_through_array_runtime_and_owner_bridge() -> (
    None
):
    """
    title: >-
      NDArray literals should use Arrow storage plus a buffer-owner bridge.
    """
    builder = Builder()
    module = _module_with_main(
        astx.VariableDeclaration(
            name="arr",
            type_=astx.NDArrayType(astx.Int32()),
            mutability=astx.MutabilityKind.mutable,
            value=_int32_ndarray([1, 2, 3, 4], shape=(2, 2)),
        ),
        astx.FunctionReturn(
            astx.NDArrayIndex(
                astx.Identifier("arr"),
                [astx.LiteralInt32(1), astx.LiteralInt32(1)],
            )
        ),
    )

    ir_text = builder.translate(module)

    assert '@"irx_arrow_array_builder_new"' in ir_text
    assert '@"irx_arrow_array_borrow_buffer_view"' in ir_text
    assert '@"irx_buffer_owner_external_new"' in ir_text
    assert "irx_ndarray_shape" in ir_text
    assert "irx_ndarray_stride" in ir_text
    assert "irx_ndarray_index_load" in ir_text
    assert_ir_parses(ir_text)


def test_ndarray_view_lowers_custom_shape_stride_and_offset() -> None:
    """
    title: NDArray views should lower with explicit shape, stride, and offset.
    """
    builder = Builder()
    view = astx.NDArrayView(
        _int32_ndarray([10, 20, 30, 40, 50, 60], shape=(2, 3)),
        shape=(2, 2),
        strides=(12, 4),
        offset_bytes=4,
    )
    module = _module_with_main(
        astx.FunctionReturn(
            astx.NDArrayIndex(
                view,
                [astx.LiteralInt32(1), astx.LiteralInt32(0)],
            )
        )
    )

    ir_text = builder.translate(module)

    assert (
        "irx_ndarray_shape_ptr" in ir_text or "irx_ndarray_shape_" in ir_text
    )
    assert (
        "irx_ndarray_stride_ptr" in ir_text
        or "irx_ndarray_strides_" in ir_text
    )
    assert "irx_buffer_index_stride_0" in ir_text
    assert "irx_buffer_index_stride_1" in ir_text
    assert (
        "irx_ndarray_offset_bytes" in ir_text
        or "irx_buffer_index_offset_1" in ir_text
    )
    assert_ir_parses(ir_text)


def test_ndarray_queries_lower_to_shape_stride_metadata() -> None:
    """
    title: NDArray queries should lower to rank, shape, and stride metadata.
    """
    builder = Builder()
    module = _module_with_main(
        astx.VariableDeclaration(
            name="arr",
            type_=astx.NDArrayType(astx.Int32()),
            mutability=astx.MutabilityKind.mutable,
            value=_int32_ndarray([1, 2, 3, 4], shape=(2, 2)),
        ),
        astx.FunctionReturn(astx.NDArrayNDim(astx.Identifier("arr"))),
    )

    ir_text = builder.translate(module)

    assert "irx_ndarray_ndim" in ir_text
    assert_ir_parses(ir_text)


def test_ndarray_build_returns_indexed_element() -> None:
    """
    title: Built NDArray programs should return indexed element values.
    """
    if shutil.which("clang") is None:
        pytest.skip("builder.build() currently requires clang")

    module = _module_with_main(
        astx.FunctionReturn(
            astx.NDArrayIndex(
                _int32_ndarray([1, 2, 3, 4, 5, 6], shape=(2, 3)),
                [astx.LiteralInt32(1), astx.LiteralInt32(2)],
            )
        )
    )

    result = build_and_run(Builder(), module)

    expected = 6
    assert result.returncode == expected, result.stderr or result.stdout


def test_ndarray_view_build_returns_view_element() -> None:
    """
    title: Built NDArray views should return values through view metadata.
    """
    if shutil.which("clang") is None:
        pytest.skip("builder.build() currently requires clang")

    view = astx.NDArrayView(
        _int32_ndarray([10, 20, 30, 40, 50, 60], shape=(2, 3)),
        shape=(2, 2),
        strides=(12, 4),
        offset_bytes=4,
    )
    module = _module_with_main(
        astx.FunctionReturn(
            astx.NDArrayIndex(
                view,
                [astx.LiteralInt32(1), astx.LiteralInt32(0)],
            )
        )
    )

    result = build_and_run(Builder(), module)

    expected = 50
    assert result.returncode == expected, result.stderr or result.stdout

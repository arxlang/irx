"""
title: Tests for the low-level buffer/view substrate.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap

from pathlib import Path

import pytest

from irx import astx
from irx.analysis import SemanticError, analyze
from irx.buffer import (
    BUFFER_FLAG_BORROWED,
    BUFFER_FLAG_OWNED,
    BUFFER_FLAG_READONLY,
    BUFFER_VIEW_FIELD_INDICES,
    BUFFER_VIEW_FIELD_NAMES,
    BufferHandle,
    BufferMutability,
    BufferOwnership,
    BufferViewMetadata,
    buffer_view_flags,
)
from irx.builder import Builder
from irx.builder.runtime.buffer.feature import build_buffer_runtime_feature
from irx.builder.runtime.linking import link_executable

from tests.conftest import assert_ir_parses


def _borrowed_metadata(
    *,
    data: BufferHandle = BufferHandle(),
    dtype: BufferHandle = BufferHandle(1),
    shape: tuple[int, ...] = (0,),
    strides: tuple[int, ...] = (1,),
    mutability: BufferMutability = BufferMutability.READONLY,
) -> BufferViewMetadata:
    """
    title: Build borrowed buffer metadata.
    parameters:
      data:
        type: BufferHandle
      dtype:
        type: BufferHandle
      shape:
        type: tuple[int, Ellipsis]
      strides:
        type: tuple[int, Ellipsis]
      mutability:
        type: BufferMutability
    returns:
      type: BufferViewMetadata
    """
    return BufferViewMetadata(
        data=data,
        owner=BufferHandle(),
        dtype=dtype,
        ndim=len(shape),
        shape=shape,
        strides=strides,
        offset_bytes=0,
        flags=buffer_view_flags(BufferOwnership.BORROWED, mutability),
    )


def _owned_metadata(
    *,
    data: BufferHandle = BufferHandle(),
    owner: BufferHandle = BufferHandle(2),
    dtype: BufferHandle = BufferHandle(1),
    shape: tuple[int, ...] = (0,),
    strides: tuple[int, ...] = (1,),
    mutability: BufferMutability = BufferMutability.READONLY,
) -> BufferViewMetadata:
    """
    title: Build owned buffer metadata.
    parameters:
      data:
        type: BufferHandle
      owner:
        type: BufferHandle
      dtype:
        type: BufferHandle
      shape:
        type: tuple[int, Ellipsis]
      strides:
        type: tuple[int, Ellipsis]
      mutability:
        type: BufferMutability
    returns:
      type: BufferViewMetadata
    """
    return BufferViewMetadata(
        data=data,
        owner=owner,
        dtype=dtype,
        ndim=len(shape),
        shape=shape,
        strides=strides,
        offset_bytes=0,
        flags=buffer_view_flags(BufferOwnership.OWNED, mutability),
    )


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


def _descriptor(metadata: BufferViewMetadata) -> astx.BufferViewDescriptor:
    """
    title: Build one descriptor node.
    parameters:
      metadata:
        type: BufferViewMetadata
    returns:
      type: astx.BufferViewDescriptor
    """
    return astx.BufferViewDescriptor(metadata)


def test_buffer_view_field_order_is_stable() -> None:
    """
    title: Canonical field order should remain ABI-stable.
    """
    assert BUFFER_VIEW_FIELD_NAMES == (
        "data",
        "owner",
        "dtype",
        "ndim",
        "shape",
        "strides",
        "offset_bytes",
        "flags",
    )
    assert BUFFER_VIEW_FIELD_INDICES == {
        "data": 0,
        "owner": 1,
        "dtype": 2,
        "ndim": 3,
        "shape": 4,
        "strides": 5,
        "offset_bytes": 6,
        "flags": 7,
    }


def test_buffer_view_descriptor_lowers_to_stable_struct() -> None:
    """
    title: Buffer views should lower as the canonical ABI struct.
    """
    builder = Builder()
    module = _module_with_main(
        astx.VariableDeclaration(
            name="view",
            type_=astx.BufferViewType(),
            mutability=astx.MutabilityKind.mutable,
            value=_descriptor(_borrowed_metadata()),
        )
    )

    ir_text = builder.translate(module)

    assert (
        '%"irx_buffer_view" = type {i8*, i8*, i8*, i32, i64*, i64*, i64, i32}'
    ) in ir_text
    assert "irx_buffer_view_retain" not in ir_text
    runtime_features = builder.translator.runtime_features
    active_features = runtime_features.active_feature_names()
    assert "buffer" not in active_features
    assert_ir_parses(ir_text)


def test_buffer_view_passes_by_value_to_irx_functions() -> None:
    """
    title: IRx-internal functions should pass view structs by value.
    """
    view_arg = astx.Argument("view", astx.BufferViewType())
    use_view_body = astx.Block()
    use_view_body.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    use_view = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="use_view",
            args=astx.Arguments(view_arg),
            return_type=astx.Int32(),
        ),
        body=use_view_body,
    )
    module = _module_with_main(
        astx.FunctionReturn(
            astx.FunctionCall(
                "use_view",
                [_descriptor(_borrowed_metadata())],
            )
        )
    )
    module.block.insert(0, use_view)

    ir_text = Builder().translate(module)

    assert (
        'define i32 @"main__use_view"(%"irx_buffer_view" %"view")'
    ) in ir_text
    assert 'call i32 @"main__use_view"(%"irx_buffer_view" ' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (
            BufferViewMetadata(
                data=BufferHandle(),
                owner=BufferHandle(),
                dtype=BufferHandle(1),
                ndim=1,
                shape=(0,),
                strides=(1,),
                offset_bytes=0,
                flags=BUFFER_FLAG_READONLY,
            ),
            "exactly one ownership",
        ),
        (
            BufferViewMetadata(
                data=BufferHandle(),
                owner=BufferHandle(),
                dtype=BufferHandle(1),
                ndim=1,
                shape=(0,),
                strides=(1,),
                offset_bytes=0,
                flags=(
                    BUFFER_FLAG_BORROWED
                    | BUFFER_FLAG_OWNED
                    | BUFFER_FLAG_READONLY
                ),
            ),
            "exactly one ownership",
        ),
        (
            BufferViewMetadata(
                data=BufferHandle(),
                owner=BufferHandle(2),
                dtype=BufferHandle(1),
                ndim=1,
                shape=(0,),
                strides=(1,),
                offset_bytes=0,
                flags=buffer_view_flags(
                    BufferOwnership.BORROWED,
                    BufferMutability.READONLY,
                ),
            ),
            "borrowed buffer views must use a null owner",
        ),
        (
            BufferViewMetadata(
                data=BufferHandle(),
                owner=BufferHandle(),
                dtype=BufferHandle(1),
                ndim=1,
                shape=(0,),
                strides=(1,),
                offset_bytes=0,
                flags=buffer_view_flags(
                    BufferOwnership.OWNED,
                    BufferMutability.READONLY,
                ),
            ),
            "owned buffer views must use a non-null owner",
        ),
    ],
)
def test_buffer_view_rejects_invalid_ownership(
    metadata: BufferViewMetadata,
    match: str,
) -> None:
    """
    title: Ownership state should be explicit and non-conflicting.
    parameters:
      metadata:
        type: BufferViewMetadata
      match:
        type: str
    """
    module = _module_with_main(
        astx.FunctionReturn(astx.BufferViewRetain(_descriptor(metadata)))
    )

    with pytest.raises(SemanticError, match=match):
        analyze(module)


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (
            BufferViewMetadata(
                data=BufferHandle(4096),
                owner=BufferHandle(),
                dtype=BufferHandle(1),
                ndim=2,
                shape=(1,),
                strides=(1,),
                offset_bytes=0,
                flags=buffer_view_flags(
                    BufferOwnership.BORROWED,
                    BufferMutability.READONLY,
                ),
            ),
            "shape length must match ndim",
        ),
        (
            BufferViewMetadata(
                data=BufferHandle(4096),
                owner=BufferHandle(),
                dtype=BufferHandle(1),
                ndim=1,
                shape=(-1,),
                strides=(1,),
                offset_bytes=0,
                flags=buffer_view_flags(
                    BufferOwnership.BORROWED,
                    BufferMutability.READONLY,
                ),
            ),
            "shape dimensions must be non-negative",
        ),
        (
            BufferViewMetadata(
                data=BufferHandle(),
                owner=BufferHandle(),
                dtype=BufferHandle(1),
                ndim=1,
                shape=(1,),
                strides=(1,),
                offset_bytes=0,
                flags=buffer_view_flags(
                    BufferOwnership.BORROWED,
                    BufferMutability.READONLY,
                ),
            ),
            "nonzero extent must use non-null data",
        ),
        (
            BufferViewMetadata(
                data=BufferHandle(),
                owner=BufferHandle(),
                dtype=BufferHandle(),
                ndim=1,
                shape=(0,),
                strides=(1,),
                offset_bytes=0,
                flags=buffer_view_flags(
                    BufferOwnership.BORROWED,
                    BufferMutability.READONLY,
                ),
            ),
            "dtype handle must be non-null",
        ),
    ],
)
def test_buffer_view_rejects_invalid_shape_stride_metadata(
    metadata: BufferViewMetadata,
    match: str,
) -> None:
    """
    title: Static shape/stride descriptor errors should be semantic errors.
    parameters:
      metadata:
        type: BufferViewMetadata
      match:
        type: str
    """
    module = _module_with_main(
        astx.FunctionReturn(astx.BufferViewRetain(_descriptor(metadata)))
    )

    with pytest.raises(SemanticError, match=match):
        analyze(module)


def test_buffer_view_rejects_readonly_write() -> None:
    """
    title: Writes through readonly views should be rejected semantically.
    """
    metadata = _borrowed_metadata(
        data=BufferHandle(4096),
        shape=(1,),
        strides=(1,),
        mutability=BufferMutability.READONLY,
    )
    module = _module_with_main(
        astx.BufferViewWrite(
            _descriptor(metadata),
            astx.LiteralUInt8(7),
        )
    )

    with pytest.raises(SemanticError, match="readonly buffer view"):
        analyze(module)


def test_rejects_readonly_write_through_static_identifier() -> None:
    """
    title: Static descriptor metadata should flow to local identifiers.
    """
    module = _module_with_main(
        astx.VariableDeclaration(
            name="view",
            type_=astx.BufferViewType(),
            mutability=astx.MutabilityKind.mutable,
            value=_descriptor(
                _borrowed_metadata(
                    data=BufferHandle(4096),
                    shape=(1,),
                    strides=(1,),
                    mutability=BufferMutability.READONLY,
                )
            ),
        ),
        astx.BufferViewWrite(
            astx.Identifier("view"),
            astx.LiteralUInt8(7),
        ),
    )

    with pytest.raises(SemanticError, match="readonly buffer view"):
        analyze(module)


def test_buffer_view_allows_mutable_borrowed_write() -> None:
    """
    title: Mutable borrowed views should be possible when explicitly flagged.
    """
    metadata = _borrowed_metadata(
        data=BufferHandle(4096),
        shape=(1,),
        strides=(1,),
        mutability=BufferMutability.WRITABLE,
    )
    module = _module_with_main(
        astx.BufferViewWrite(
            _descriptor(metadata),
            astx.LiteralUInt8(7),
        )
    )

    ir_text = Builder().translate(module)

    assert "store i8 7" in ir_text
    assert_ir_parses(ir_text)


def test_buffer_view_descriptor_assignment_is_shallow_copy() -> None:
    """
    title: Descriptor assignment should copy metadata without retaining.
    """
    module = _module_with_main(
        astx.VariableDeclaration(
            name="view",
            type_=astx.BufferViewType(),
            mutability=astx.MutabilityKind.mutable,
            value=_descriptor(_borrowed_metadata()),
        ),
        astx.VariableDeclaration(
            name="copy",
            type_=astx.BufferViewType(),
            mutability=astx.MutabilityKind.mutable,
            value=astx.Identifier("view"),
        ),
    )

    ir_text = Builder().translate(module)

    assert 'load %"irx_buffer_view"' in ir_text
    assert 'store %"irx_buffer_view"' in ir_text
    assert "irx_buffer_view_retain" not in ir_text
    assert_ir_parses(ir_text)


def test_buffer_view_retain_uses_buffer_runtime_feature() -> None:
    """
    title: Explicit retain should route through the buffer runtime feature.
    """
    builder = Builder()
    module = _module_with_main(
        astx.FunctionReturn(
            astx.BufferViewRetain(_descriptor(_owned_metadata()))
        )
    )

    ir_text = builder.translate(module)

    runtime_features = builder.translator.runtime_features
    active_features = runtime_features.active_feature_names()
    assert "buffer" in active_features
    assert "irx_buffer_view_retain" in ir_text
    assert builder.translator.runtime_features.native_artifacts()
    assert_ir_parses(ir_text)


def test_buffer_view_runtime_boundary_uses_descriptor_pointer_abi() -> None:
    """
    title: Runtime helper calls should pass buffer views by descriptor pointer.
    """
    ir_text = Builder().translate(
        _module_with_main(
            astx.FunctionReturn(
                astx.BufferViewRetain(_descriptor(_owned_metadata()))
            )
        )
    )

    assert (
        'declare external i32 @"irx_buffer_view_retain"'
        '(%"irx_buffer_view"* %".1")'
    ) in ir_text
    assert '%"irx_buffer_retain_view" = alloca %"irx_buffer_view"' in ir_text
    assert (
        'call i32 @"irx_buffer_view_retain"'
        '(%"irx_buffer_view"* %"irx_buffer_retain_view")'
    ) in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize(
    "node_factory",
    [
        astx.BufferViewRetain,
        astx.BufferViewRelease,
    ],
)
def test_buffer_lifetime_helpers_reject_borrowed_static_views(
    node_factory: type[astx.BufferViewRetain] | type[astx.BufferViewRelease],
) -> None:
    """
    title: Borrowed views should not allow explicit retain/release helpers.
    parameters:
      node_factory:
        type: type[astx.BufferViewRetain] | type[astx.BufferViewRelease]
    """
    module = _module_with_main(
        astx.FunctionReturn(node_factory(_descriptor(_borrowed_metadata())))
    )

    with pytest.raises(
        SemanticError,
        match="requires an owned or external-owner view",
    ):
        analyze(module)


def test_buffer_lifetime_helpers_reject_borrowed_static_identifiers() -> None:
    """
    title: Borrowed lifetime checks should inspect static local initializers.
    """
    module = _module_with_main(
        astx.VariableDeclaration(
            name="view",
            type_=astx.BufferViewType(),
            mutability=astx.MutabilityKind.mutable,
            value=_descriptor(_borrowed_metadata()),
        ),
        astx.FunctionReturn(astx.BufferViewRetain(astx.Identifier("view"))),
    )

    with pytest.raises(
        SemanticError,
        match="requires an owned or external-owner view",
    ):
        analyze(module)


def test_plain_buffer_descriptor_does_not_activate_runtime_feature() -> None:
    """
    title: Plain descriptor lowering should not pull native helpers in.
    """
    builder = Builder()
    module = _module_with_main(
        astx.VariableDeclaration(
            name="view",
            type_=astx.BufferViewType(),
            mutability=astx.MutabilityKind.mutable,
            value=_descriptor(_borrowed_metadata()),
        )
    )

    ir_text = builder.translate(module)

    runtime_features = builder.translator.runtime_features
    active_features = runtime_features.active_feature_names()
    assert "buffer" not in active_features
    assert "irx_buffer_view_retain" not in ir_text
    assert builder.translator.runtime_features.native_artifacts() == ()


def test_buffer_runtime_owner_retain_release_harness() -> None:
    """
    title: Native buffer runtime should retain/release owner handles.
    """
    clang_binary = shutil.which("clang")
    if clang_binary is None:
        pytest.skip("clang is required for buffer runtime harness tests")

    feature = build_buffer_runtime_feature()
    include_dirs: list[Path] = []
    seen_include_dirs: set[Path] = set()
    for artifact in feature.artifacts:
        for include_dir in artifact.include_dirs:
            if include_dir in seen_include_dirs:
                continue
            seen_include_dirs.add(include_dir)
            include_dirs.append(include_dir)

    source = """
      #include "irx_buffer_runtime.h"

      static int released = 0;

      static void mark_released(void* context) {
        int* counter = (int*)context;
        *counter += 1;
      }

      int main(void) {
        int64_t shape[1] = {1};
        int64_t strides[1] = {1};
        irx_buffer_owner_handle* owner = 0;
        int32_t code = irx_buffer_owner_external_new(
            &released, mark_released, &owner);
        if (code != 0 || owner == 0) return 1;

        irx_buffer_view view = {0};
        view.data = (void*)4096;
        view.owner = owner;
        view.dtype = (void*)1;
        view.ndim = 1;
        view.shape = shape;
        view.strides = strides;
        view.flags = IRX_BUFFER_FLAG_OWNED | IRX_BUFFER_FLAG_WRITABLE;

        if (irx_buffer_view_retain(&view) != 0) return 2;
        if (irx_buffer_view_release(&view) != 0) return 3;
        if (released != 0) return 4;
        if (irx_buffer_owner_release(owner) != 0) return 5;
        if (released != 1) return 6;
        return 0;
      }
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_path = tmp_path / "buffer_harness.c"
        object_path = tmp_path / "buffer_harness.o"
        output_path = tmp_path / "buffer_harness"
        source_path.write_text(textwrap.dedent(source), encoding="utf8")

        subprocess.run(
            [
                clang_binary,
                "-c",
                str(source_path),
                "-o",
                str(object_path),
                *[
                    option
                    for include_dir in include_dirs
                    for option in ("-I", str(include_dir))
                ],
                "-std=c99",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        link_executable(
            primary_object=object_path,
            output_file=output_path,
            artifacts=feature.artifacts,
            linker_flags=feature.linker_flags,
            clang_binary=clang_binary,
        )
        result = subprocess.run(
            [str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0, result.stderr or result.stdout

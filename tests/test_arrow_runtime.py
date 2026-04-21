"""
title: Tests for the Arrow runtime feature and lowering path.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import tempfile
import textwrap

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict, cast

import nanoarrow
import pytest

from arx_nanoarrow_sources import get_include_dir, get_source_files
from irx import astx
from irx.buffer import (
    BUFFER_DTYPE_BOOL,
    BUFFER_DTYPE_INT32,
    BUFFER_DTYPE_UINT64,
    BUFFER_FLAG_BORROWED,
    BUFFER_FLAG_C_CONTIGUOUS,
    BUFFER_FLAG_READONLY,
    BUFFER_FLAG_VALIDITY_BITMAP,
)
from irx.builder import Builder
from irx.builder.runtime.array.feature import (
    ARRAY_PRIMITIVE_TYPE_SPECS,
    IRX_ARROW_TYPE_BOOL,
    IRX_ARROW_TYPE_FLOAT32,
    IRX_ARROW_TYPE_FLOAT64,
    IRX_ARROW_TYPE_INT8,
    IRX_ARROW_TYPE_INT16,
    IRX_ARROW_TYPE_INT32,
    IRX_ARROW_TYPE_INT64,
    IRX_ARROW_TYPE_UINT8,
    IRX_ARROW_TYPE_UINT16,
    IRX_ARROW_TYPE_UINT32,
    IRX_ARROW_TYPE_UINT64,
    build_array_runtime_feature,
)
from irx.builder.runtime.linking import (
    compile_native_artifacts,
    link_executable,
)
from llvmlite import binding as llvm
from nanoarrow import Array
from nanoarrow.c_array import allocate_c_array
from nanoarrow.c_schema import allocate_c_schema


class SupportedPrimitiveMetadata(TypedDict):
    type_id: int
    dtype_token: int
    element_size_bytes: int | None
    buffer_view_compatible: bool


PrimitiveValue = int | float | bool | None
BuilderValue = int | float | None
ArrowSchemaFactory = Callable[[], object]


class ArrowSchemaStruct(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("metadata", ctypes.c_char_p),
        ("flags", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


class ArrowArrayStruct(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_int64),
        ("null_count", ctypes.c_int64),
        ("offset", ctypes.c_int64),
        ("n_buffers", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("buffers", ctypes.c_void_p),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


class BufferViewStruct(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("owner", ctypes.c_void_p),
        ("dtype", ctypes.c_void_p),
        ("ndim", ctypes.c_int32),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("offset_bytes", ctypes.c_int64),
        ("flags", ctypes.c_int32),
    ]


PRIMITIVE_IMPORT_CASES: tuple[
    tuple[
        str,
        ArrowSchemaFactory,
        Sequence[PrimitiveValue],
        Sequence[PrimitiveValue],
    ],
    ...,
] = (
    ("int8", nanoarrow.int8, [1, -2, 127], [1, -2, 127]),
    ("int16", nanoarrow.int16, [1, -32000, 32000], [1, -32000, 32000]),
    (
        "int32",
        nanoarrow.int32,
        [1, -2_000_000_000, 2_000_000_000],
        [1, -2_000_000_000, 2_000_000_000],
    ),
    (
        "int64",
        nanoarrow.int64,
        [1, -(2**40), 2**40],
        [1, -(2**40), 2**40],
    ),
    ("uint8", nanoarrow.uint8, [0, 7, 255], [0, 7, 255]),
    ("uint16", nanoarrow.uint16, [0, 42, 65535], [0, 42, 65535]),
    ("uint32", nanoarrow.uint32, [0, 7, 2**32 - 1], [0, 7, 2**32 - 1]),
    (
        "uint64",
        nanoarrow.uint64,
        [0, 7, 2**63 + 7],
        [0, 7, 2**63 + 7],
    ),
    ("float32", nanoarrow.float32, [1.5, -2.25, 3.75], [1.5, -2.25, 3.75]),
    ("float64", nanoarrow.float64, [1.25, -2.5, 3.75], [1.25, -2.5, 3.75]),
    (
        "bool",
        nanoarrow.bool_,
        [True, False, None, True],
        [True, False, None, True],
    ),
)


BUILDER_CASES: tuple[
    tuple[str, int, str, Sequence[BuilderValue], Sequence[PrimitiveValue]],
    ...,
] = (
    ("int8", IRX_ARROW_TYPE_INT8, "int", [1, -2, 127], [1, -2, 127]),
    (
        "int16",
        IRX_ARROW_TYPE_INT16,
        "int",
        [1, -32000, 32000],
        [1, -32000, 32000],
    ),
    (
        "int32",
        IRX_ARROW_TYPE_INT32,
        "int",
        [1, -2_000_000_000, 2_000_000_000],
        [1, -2_000_000_000, 2_000_000_000],
    ),
    (
        "int64",
        IRX_ARROW_TYPE_INT64,
        "int",
        [1, -(2**40), 2**40],
        [1, -(2**40), 2**40],
    ),
    ("uint8", IRX_ARROW_TYPE_UINT8, "uint", [0, 7, 255], [0, 7, 255]),
    ("uint16", IRX_ARROW_TYPE_UINT16, "uint", [0, 42, 65535], [0, 42, 65535]),
    (
        "uint32",
        IRX_ARROW_TYPE_UINT32,
        "uint",
        [0, 7, 2**32 - 1],
        [0, 7, 2**32 - 1],
    ),
    (
        "uint64",
        IRX_ARROW_TYPE_UINT64,
        "uint",
        [0, 7, 2**63 + 7],
        [0, 7, 2**63 + 7],
    ),
    (
        "float32",
        IRX_ARROW_TYPE_FLOAT32,
        "double",
        [1.5, -2.25, 3.75],
        [1.5, -2.25, 3.75],
    ),
    (
        "float64",
        IRX_ARROW_TYPE_FLOAT64,
        "double",
        [1.25, -2.5, 3.75],
        [1.25, -2.5, 3.75],
    ),
    (
        "bool",
        IRX_ARROW_TYPE_BOOL,
        "int",
        [1, 0, None, 1],
        [True, False, None, True],
    ),
)


def _find_c_compiler() -> str | None:
    """
    title: Find one usable C compiler for runtime tests.
    returns:
      type: str | None
    """
    return shutil.which("clang") or shutil.which("cc")


def _array_length_module(values: list[int]) -> astx.Module:
    """
    title: Array length module.
    parameters:
      values:
        type: list[int]
    returns:
      type: astx.Module
    """
    module = astx.Module()
    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    body = astx.Block()
    body.append(
        astx.FunctionReturn(
            astx.ArrayInt32ArrayLength(
                [astx.LiteralInt32(value) for value in values]
            )
        )
    )
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))
    return module


def _plain_main_module() -> astx.Module:
    """
    title: Plain main module.
    returns:
      type: astx.Module
    """
    module = astx.Module()
    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    body = astx.Block()
    body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))
    return module


def _compile_arrow_harness(source: str) -> subprocess.CompletedProcess[str]:
    """
    title: Compile arrow harness.
    parameters:
      source:
        type: str
    returns:
      type: subprocess.CompletedProcess[str]
    """
    feature = build_array_runtime_feature()
    include_dirs: list[Path] = []
    seen_include_dirs: set[Path] = set()
    for artifact in feature.artifacts:
        for include_dir in artifact.include_dirs:
            if include_dir in seen_include_dirs:
                continue
            seen_include_dirs.add(include_dir)
            include_dirs.append(include_dir)

    c_compiler = _find_c_compiler()
    if c_compiler is None:
        pytest.skip("a C compiler is required for Arrow runtime harness tests")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_path = tmp_path / "arrow_harness.c"
        object_path = tmp_path / "arrow_harness.o"
        output_path = tmp_path / "arrow_harness"

        source_path.write_text(textwrap.dedent(source), encoding="utf8")
        subprocess.run(
            [
                c_compiler,
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
            clang_binary=c_compiler,
        )
        return subprocess.run(
            [str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )


def _shared_library_suffix() -> str:
    """
    title: Shared library suffix.
    returns:
      type: str
    """
    if sys.platform == "darwin":
        return ".dylib"

    return ".so"


@contextmanager
def _load_arrow_runtime_library() -> Iterator[ctypes.CDLL]:
    """
    title: Load arrow runtime library.
    returns:
      type: Iterator[ctypes.CDLL]
    """
    if sys.platform == "win32":
        pytest.skip("nanoarrow interop shared-library tests require Unix")

    feature = build_array_runtime_feature()
    c_compiler = _find_c_compiler()
    if c_compiler is None:
        pytest.skip("a C compiler is required for Arrow runtime interop tests")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_path = (
            tmp_path / f"libirx_arrow_runtime{_shared_library_suffix()}"
        )
        link_inputs = compile_native_artifacts(
            feature.artifacts,
            tmp_path,
            c_compiler,
        )

        command = [c_compiler]
        if sys.platform == "darwin":
            command.append("-dynamiclib")
        else:
            command.append("-shared")

        command.extend(str(obj) for obj in link_inputs.objects)
        command.extend(link_inputs.linker_flags)
        command.extend(["-o", str(output_path)])
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        library = ctypes.CDLL(str(output_path))
        _configure_arrow_runtime_library(library)
        yield library


def _configure_arrow_runtime_library(library: ctypes.CDLL) -> None:
    """
    title: Configure arrow runtime library.
    parameters:
      library:
        type: ctypes.CDLL
    """
    library.irx_arrow_schema_import_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.irx_arrow_schema_import_copy.restype = ctypes.c_int
    library.irx_arrow_schema_export.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.irx_arrow_schema_export.restype = ctypes.c_int
    library.irx_arrow_schema_type_id.argtypes = [ctypes.c_void_p]
    library.irx_arrow_schema_type_id.restype = ctypes.c_int32
    library.irx_arrow_schema_is_nullable.argtypes = [ctypes.c_void_p]
    library.irx_arrow_schema_is_nullable.restype = ctypes.c_int32
    library.irx_arrow_schema_retain.argtypes = [ctypes.c_void_p]
    library.irx_arrow_schema_retain.restype = ctypes.c_int
    library.irx_arrow_schema_release.argtypes = [ctypes.c_void_p]
    library.irx_arrow_schema_release.restype = None
    library.irx_arrow_array_builder_new.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.irx_arrow_array_builder_new.restype = ctypes.c_int
    library.irx_arrow_array_builder_append_null.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    library.irx_arrow_array_builder_append_null.restype = ctypes.c_int
    library.irx_arrow_array_builder_append_int.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
    ]
    library.irx_arrow_array_builder_append_int.restype = ctypes.c_int
    library.irx_arrow_array_builder_append_uint.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]
    library.irx_arrow_array_builder_append_uint.restype = ctypes.c_int
    library.irx_arrow_array_builder_append_double.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
    ]
    library.irx_arrow_array_builder_append_double.restype = ctypes.c_int
    library.irx_arrow_array_builder_int32_new.argtypes = [
        ctypes.POINTER(ctypes.c_void_p)
    ]
    library.irx_arrow_array_builder_int32_new.restype = ctypes.c_int
    library.irx_arrow_array_builder_append_int32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
    ]
    library.irx_arrow_array_builder_append_int32.restype = ctypes.c_int
    library.irx_arrow_array_builder_finish.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.irx_arrow_array_builder_finish.restype = ctypes.c_int
    library.irx_arrow_array_builder_release.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_builder_release.restype = None
    library.irx_arrow_array_length.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_length.restype = ctypes.c_int64
    library.irx_arrow_array_offset.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_offset.restype = ctypes.c_int64
    library.irx_arrow_array_null_count.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_null_count.restype = ctypes.c_int64
    library.irx_arrow_array_type_id.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_type_id.restype = ctypes.c_int32
    library.irx_arrow_array_is_nullable.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_is_nullable.restype = ctypes.c_int32
    library.irx_arrow_array_has_validity_bitmap.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_has_validity_bitmap.restype = ctypes.c_int32
    library.irx_arrow_array_can_borrow_buffer_view.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_can_borrow_buffer_view.restype = ctypes.c_int32
    library.irx_arrow_array_schema_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.irx_arrow_array_schema_copy.restype = ctypes.c_int
    library.irx_arrow_array_export.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.irx_arrow_array_export.restype = ctypes.c_int
    library.irx_arrow_array_import.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.irx_arrow_array_import.restype = ctypes.c_int
    library.irx_arrow_array_import_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.irx_arrow_array_import_copy.restype = ctypes.c_int
    library.irx_arrow_array_import_move.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.irx_arrow_array_import_move.restype = ctypes.c_int
    library.irx_arrow_array_validity_bitmap.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    library.irx_arrow_array_validity_bitmap.restype = ctypes.c_int
    library.irx_arrow_array_borrow_buffer_view.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(BufferViewStruct),
    ]
    library.irx_arrow_array_borrow_buffer_view.restype = ctypes.c_int
    library.irx_arrow_array_retain.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_retain.restype = ctypes.c_int
    library.irx_arrow_array_release.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_release.restype = None
    library.irx_arrow_last_error.argtypes = []
    library.irx_arrow_last_error.restype = ctypes.c_char_p


def _assert_arrow_ok(library: ctypes.CDLL, code: int) -> None:
    """
    title: Assert arrow ok.
    parameters:
      library:
        type: ctypes.CDLL
      code:
        type: int
    """
    assert code == 0, library.irx_arrow_last_error().decode()


def _build_runtime_array(
    library: ctypes.CDLL,
    type_id: int,
    append_kind: str,
    values: Sequence[BuilderValue],
) -> ctypes.c_void_p:
    """
    title: Build runtime array.
    parameters:
      library:
        type: ctypes.CDLL
      type_id:
        type: int
      append_kind:
        type: str
      values:
        type: Sequence[BuilderValue]
    returns:
      type: ctypes.c_void_p
    """
    builder = ctypes.c_void_p()
    array_handle = ctypes.c_void_p()

    _assert_arrow_ok(
        library,
        library.irx_arrow_array_builder_new(type_id, ctypes.byref(builder)),
    )

    try:
        for value in values:
            if value is None:
                _assert_arrow_ok(
                    library,
                    library.irx_arrow_array_builder_append_null(builder, 1),
                )
                continue

            if append_kind == "int":
                _assert_arrow_ok(
                    library,
                    library.irx_arrow_array_builder_append_int(
                        builder,
                        int(value),
                    ),
                )
                continue

            if append_kind == "uint":
                _assert_arrow_ok(
                    library,
                    library.irx_arrow_array_builder_append_uint(
                        builder,
                        int(value),
                    ),
                )
                continue

            if append_kind == "double":
                _assert_arrow_ok(
                    library,
                    library.irx_arrow_array_builder_append_double(
                        builder,
                        float(value),
                    ),
                )
                continue

            raise AssertionError(f"unknown append kind {append_kind!r}")

        _assert_arrow_ok(
            library,
            library.irx_arrow_array_builder_finish(
                builder,
                ctypes.byref(array_handle),
            ),
        )
        return array_handle
    finally:
        if builder.value is not None and array_handle.value is None:
            library.irx_arrow_array_builder_release(builder)


def _arrow_array_struct(addr: int) -> ArrowArrayStruct:
    """
    title: View one ArrowArray at an address.
    parameters:
      addr:
        type: int
    returns:
      type: ArrowArrayStruct
    """
    return ctypes.cast(addr, ctypes.POINTER(ArrowArrayStruct)).contents


def _arrow_schema_struct(addr: int) -> ArrowSchemaStruct:
    """
    title: View one ArrowSchema at an address.
    parameters:
      addr:
        type: int
    returns:
      type: ArrowSchemaStruct
    """
    return ctypes.cast(addr, ctypes.POINTER(ArrowSchemaStruct)).contents


def _primitive_spec(name: str) -> tuple[int, int, bool]:
    """
    title: Return one runtime primitive metadata triple.
    parameters:
      name:
        type: str
    returns:
      type: tuple[int, int, bool]
    """
    spec = ARRAY_PRIMITIVE_TYPE_SPECS[name]
    return (
        spec.type_id,
        spec.dtype_token,
        spec.buffer_view_compatible,
    )


def test_arrow_symbols_absent_when_unused() -> None:
    """
    title: Array runtime declarations should be absent when unused.
    """
    builder = Builder()

    ir_text = builder.translate(_array_length_module([]))
    assert "irx_arrow_array_builder_int32_new" in ir_text

    plain_builder = Builder()
    plain_ir = plain_builder.translate(_plain_main_module())
    assert "irx_arrow_" not in plain_ir


def test_arrow_length_codegen_declares_runtime_symbols() -> None:
    """
    title: Array lowering should declare runtime symbols and parse as LLVM.
    """
    builder = Builder()
    ir_text = builder.translate(_array_length_module([1, 2, 3]))

    llvm.parse_assembly(ir_text)

    active_features = (
        builder.translator.runtime_features.active_feature_names()
    )

    assert "array" in active_features
    assert '@"irx_arrow_array_builder_int32_new"' in ir_text
    assert '@"irx_arrow_array_length"' in ir_text
    assert builder.translator.runtime_features.native_artifacts()


def test_arrow_feature_uses_packaged_nanoarrow_sources() -> None:
    """
    title: Arrow runtime should compile against arx-nanoarrow-sources.
    """
    feature = build_array_runtime_feature()
    native_sources = {
        artifact.path
        for artifact in feature.artifacts
        if artifact.kind == "c_source"
    }

    assert get_source_files()
    assert set(get_source_files()).issubset(native_sources)

    for artifact in feature.artifacts:
        if artifact.kind == "c_source":
            assert get_include_dir() in artifact.include_dirs


def test_arrow_feature_metadata_exposes_supported_primitive_mapping() -> None:
    """
    title: >-
      Arrow feature metadata should publish the explicit primitive mapping.
    """
    feature = build_array_runtime_feature()
    supported = cast(
        dict[str, SupportedPrimitiveMetadata],
        feature.metadata["supported_primitive_types"],
    )

    assert supported["int32"] == {
        "type_id": IRX_ARROW_TYPE_INT32,
        "dtype_token": BUFFER_DTYPE_INT32,
        "element_size_bytes": 4,
        "buffer_view_compatible": True,
    }
    assert supported["uint64"] == {
        "type_id": IRX_ARROW_TYPE_UINT64,
        "dtype_token": BUFFER_DTYPE_UINT64,
        "element_size_bytes": 8,
        "buffer_view_compatible": True,
    }
    assert supported["bool"] == {
        "type_id": IRX_ARROW_TYPE_BOOL,
        "dtype_token": BUFFER_DTYPE_BOOL,
        "element_size_bytes": None,
        "buffer_view_compatible": False,
    }


def test_arrow_length_build_returns_length() -> None:
    """
    title: >-
      Building an array-backed module should link and return the array length.
    """
    if shutil.which("clang") is None:
        pytest.skip("builder.build() currently requires clang")

    builder = Builder()
    module = _array_length_module([10, 20, 30])

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "arrow_len"
        builder.build(module, str(output_path))
        result = subprocess.run(
            [str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 3  # noqa: PLR2004
    assert result.stdout == ""


def test_arrow_runtime_harness_lifecycle() -> None:
    """
    title: >-
      Arrow runtime C ABI should support create append finish inspect release.
    """
    result = _compile_arrow_harness(
        """
        #include "irx_arrow_runtime.h"

        int main(void) {
          irx_arrow_array_builder_handle* builder = NULL;
          irx_arrow_array_handle* array = NULL;

          if (irx_arrow_array_builder_int32_new(&builder) != 0) return 11;
          if (irx_arrow_array_builder_append_int32(builder, 1) != 0) return 12;
          if (irx_arrow_array_builder_append_int32(builder, 2) != 0) return 13;
          if (irx_arrow_array_builder_append_int32(builder, 3) != 0) return 14;
          if (irx_arrow_array_builder_finish(builder, &array) != 0) {
            irx_arrow_array_builder_release(builder);
            return 15;
          }

          if (irx_arrow_array_length(array) != 3) return 16;
          if (irx_arrow_array_null_count(array) != 0) return 17;
          if (irx_arrow_array_type_id(array) != IRX_ARROW_TYPE_INT32) {
            return 18;
          }

          irx_arrow_array_release(array);
          return 0;
        }
        """
    )

    assert result.returncode == 0
    assert result.stderr == ""


def test_arrow_runtime_harness_c_data_roundtrip() -> None:
    """
    title: Arrow runtime should roundtrip int32 arrays through Arrow C Data.
    """
    result = _compile_arrow_harness(
        """
        #include "irx_arrow_runtime.h"

        int main(void) {
          irx_arrow_array_builder_handle* builder = NULL;
          irx_arrow_array_handle* array = NULL;
          irx_arrow_array_handle* imported = NULL;
          struct ArrowArray exported_array;
          struct ArrowSchema exported_schema;

          if (irx_arrow_array_builder_int32_new(&builder) != 0) return 21;
          if (irx_arrow_array_builder_append_int32(builder, 4) != 0) return 22;
          if (irx_arrow_array_builder_append_int32(builder, 5) != 0) return 23;
          if (irx_arrow_array_builder_finish(builder, &array) != 0) {
            irx_arrow_array_builder_release(builder);
            return 24;
          }

          if (
              irx_arrow_array_export(
                  array, &exported_array, &exported_schema) != 0) {
            irx_arrow_array_release(array);
            return 25;
          }

          if (
              irx_arrow_array_import(
                  &exported_array, &exported_schema, &imported) != 0) {
            if (exported_array.release != NULL) {
              exported_array.release(&exported_array);
            }
            if (exported_schema.release != NULL) {
              exported_schema.release(&exported_schema);
            }
            irx_arrow_array_release(array);
            return 26;
          }

          if (exported_array.release != NULL) {
            exported_array.release(&exported_array);
          }
          if (exported_schema.release != NULL) {
            exported_schema.release(&exported_schema);
          }

          if (irx_arrow_array_length(imported) != 2) return 27;
          if (irx_arrow_array_type_id(imported) != IRX_ARROW_TYPE_INT32) {
            return 28;
          }

          irx_arrow_array_release(imported);
          irx_arrow_array_release(array);
          return 0;
        }
        """
    )

    assert result.returncode == 0
    assert result.stderr == ""


def test_arrow_runtime_harness_buffer_view_bridge() -> None:
    """
    title: >-
      Arrow runtime C ABI should expose supported numeric buffers as views.
    """
    result = _compile_arrow_harness(
        """
        #include "irx_arrow_runtime.h"

        int main(void) {
          irx_arrow_array_builder_handle* builder = NULL;
          irx_arrow_array_handle* array = NULL;
          irx_buffer_view view = {0};

          if (
              irx_arrow_array_builder_new(
                  IRX_ARROW_TYPE_INT16,
                  &builder) != 0) {
            return 31;
          }
          if (irx_arrow_array_builder_append_int(builder, 10) != 0) return 32;
          if (irx_arrow_array_builder_append_null(builder, 1) != 0) return 33;
          if (irx_arrow_array_builder_append_int(builder, 30) != 0) return 34;
          if (irx_arrow_array_builder_finish(builder, &array) != 0) {
            irx_arrow_array_builder_release(builder);
            return 35;
          }

          if (irx_arrow_array_has_validity_bitmap(array) != 1) return 36;
          if (irx_arrow_array_borrow_buffer_view(array, &view) != 0) return 37;
          if (view.dtype != (void*)IRX_BUFFER_DTYPE_INT16) return 38;
          if (view.ndim != 1) return 39;
          if (view.shape == NULL || view.shape[0] != 3) return 40;
          if (view.strides == NULL || view.strides[0] != 2) return 41;
          if ((view.flags & IRX_BUFFER_FLAG_VALIDITY_BITMAP) == 0) return 42;

          irx_arrow_array_release(array);
          return 0;
        }
        """
    )

    assert result.returncode == 0
    assert result.stderr == ""


def test_arrow_runtime_imports_python_nanoarrow_array() -> None:
    """
    title: Arrow runtime should import arrays built by Python nanoarrow.
    """
    with _load_arrow_runtime_library() as library:
        source = nanoarrow.c_array([7, 8, 9], nanoarrow.int32())
        array_handle = ctypes.c_void_p()

        try:
            _assert_arrow_ok(
                library,
                library.irx_arrow_array_import(
                    source._addr(),
                    source.schema._addr(),
                    ctypes.byref(array_handle),
                ),
            )

            assert array_handle.value is not None
            assert library.irx_arrow_array_length(array_handle) == 3  # noqa: PLR2004
            assert library.irx_arrow_array_null_count(array_handle) == 0
            assert library.irx_arrow_array_type_id(array_handle) == (
                IRX_ARROW_TYPE_INT32
            )
        finally:
            if array_handle.value is not None:
                library.irx_arrow_array_release(array_handle)


def test_arrow_runtime_exports_to_python_nanoarrow_array() -> None:
    """
    title: Arrow runtime should export arrays consumable by Python nanoarrow.
    """
    with _load_arrow_runtime_library() as library:
        array_handle = _build_runtime_array(
            library,
            IRX_ARROW_TYPE_INT32,
            "int",
            [4, 5, 6],
        )
        try:
            exported_schema = allocate_c_schema()
            exported_array = allocate_c_array(exported_schema)

            _assert_arrow_ok(
                library,
                library.irx_arrow_array_export(
                    array_handle,
                    exported_array._addr(),
                    exported_schema._addr(),
                ),
            )

            exported = Array(exported_array)
            assert len(exported) == 3  # noqa: PLR2004
            assert list(exported.iter_py()) == [4, 5, 6]
        finally:
            library.irx_arrow_array_release(array_handle)


@pytest.mark.parametrize(
    ("name", "schema_factory", "values", "expected"),
    PRIMITIVE_IMPORT_CASES,
    ids=[case[0] for case in PRIMITIVE_IMPORT_CASES],
)
def test_arrow_runtime_import_export_roundtrips_supported_primitives(
    name: str,
    schema_factory: ArrowSchemaFactory,
    values: Sequence[PrimitiveValue],
    expected: Sequence[PrimitiveValue],
) -> None:
    """
    title: Arrow runtime should roundtrip the supported primitive set.
    parameters:
      name:
        type: str
      schema_factory:
        type: ArrowSchemaFactory
      values:
        type: Sequence[PrimitiveValue]
      expected:
        type: Sequence[PrimitiveValue]
    """
    with _load_arrow_runtime_library() as library:
        type_id, _, _ = _primitive_spec(name)
        source = nanoarrow.c_array(values, schema_factory())
        array_handle = ctypes.c_void_p()

        try:
            _assert_arrow_ok(
                library,
                library.irx_arrow_array_import_copy(
                    source._addr(),
                    source.schema._addr(),
                    ctypes.byref(array_handle),
                ),
            )

            assert library.irx_arrow_array_type_id(array_handle) == type_id
            assert library.irx_arrow_array_length(array_handle) == len(
                expected
            )
            assert library.irx_arrow_array_null_count(array_handle) == sum(
                value is None for value in expected
            )

            exported_schema = allocate_c_schema()
            exported_array = allocate_c_array(exported_schema)
            _assert_arrow_ok(
                library,
                library.irx_arrow_array_export(
                    array_handle,
                    exported_array._addr(),
                    exported_schema._addr(),
                ),
            )

            exported = Array(exported_array)
            assert exported.to_pylist() == expected
        finally:
            if array_handle.value is not None:
                library.irx_arrow_array_release(array_handle)


@pytest.mark.parametrize(
    ("name", "type_id", "append_kind", "values", "expected"),
    BUILDER_CASES,
    ids=[case[0] for case in BUILDER_CASES],
)
def test_arrow_runtime_builder_supports_supported_primitives(
    name: str,
    type_id: int,
    append_kind: str,
    values: Sequence[BuilderValue],
    expected: Sequence[PrimitiveValue],
) -> None:
    """
    title: >-
      Arrow runtime builders should produce all supported primitive arrays.
    parameters:
      name:
        type: str
      type_id:
        type: int
      append_kind:
        type: str
      values:
        type: Sequence[BuilderValue]
      expected:
        type: Sequence[PrimitiveValue]
    """
    with _load_arrow_runtime_library() as library:
        array_handle = _build_runtime_array(
            library,
            type_id,
            append_kind,
            values,
        )

        try:
            assert library.irx_arrow_array_type_id(array_handle) == type_id
            assert library.irx_arrow_array_length(array_handle) == len(
                expected
            )
            assert library.irx_arrow_array_null_count(array_handle) == sum(
                value is None for value in expected
            )

            exported_schema = allocate_c_schema()
            exported_array = allocate_c_array(exported_schema)
            _assert_arrow_ok(
                library,
                library.irx_arrow_array_export(
                    array_handle,
                    exported_array._addr(),
                    exported_schema._addr(),
                ),
            )

            exported = Array(exported_array)
            assert exported.to_pylist() == expected
        finally:
            library.irx_arrow_array_release(array_handle)


def test_arrow_runtime_nullable_numeric_bridge_is_explicit() -> None:
    """
    title: Nullable numeric arrays should stay Arrow-aware even when bridged.
    """
    with _load_arrow_runtime_library() as library:
        source = nanoarrow.c_array([10, None, 30], nanoarrow.int32())
        array_handle = ctypes.c_void_p()
        validity_data = ctypes.c_void_p()
        validity_offset_bits = ctypes.c_int64()
        validity_length_bits = ctypes.c_int64()
        view = BufferViewStruct()

        try:
            _assert_arrow_ok(
                library,
                library.irx_arrow_array_import_copy(
                    source._addr(),
                    source.schema._addr(),
                    ctypes.byref(array_handle),
                ),
            )

            assert library.irx_arrow_array_is_nullable(array_handle) == 1
            assert library.irx_arrow_array_null_count(array_handle) == 1
            assert (
                library.irx_arrow_array_has_validity_bitmap(array_handle) == 1
            )
            assert (
                library.irx_arrow_array_can_borrow_buffer_view(array_handle)
                == 1
            )

            _assert_arrow_ok(
                library,
                library.irx_arrow_array_validity_bitmap(
                    array_handle,
                    ctypes.byref(validity_data),
                    ctypes.byref(validity_offset_bits),
                    ctypes.byref(validity_length_bits),
                ),
            )
            assert validity_data.value is not None
            assert validity_offset_bits.value == 0
            assert validity_length_bits.value == 3  # noqa: PLR2004

            _assert_arrow_ok(
                library,
                library.irx_arrow_array_borrow_buffer_view(
                    array_handle,
                    ctypes.byref(view),
                ),
            )

            assert view.owner is None
            assert view.dtype == BUFFER_DTYPE_INT32
            assert view.ndim == 1
            assert view.shape[0] == 3  # noqa: PLR2004
            assert view.strides[0] == 4  # noqa: PLR2004
            assert view.offset_bytes == 0
            assert view.flags == (
                BUFFER_FLAG_BORROWED
                | BUFFER_FLAG_READONLY
                | BUFFER_FLAG_C_CONTIGUOUS
                | BUFFER_FLAG_VALIDITY_BITMAP
            )
        finally:
            if array_handle.value is not None:
                library.irx_arrow_array_release(array_handle)


def test_arrow_runtime_bool_arrays_reject_plain_buffer_view_bridge() -> None:
    """
    title: Bit-packed bool arrays should not masquerade as plain buffer views.
    """
    with _load_arrow_runtime_library() as library:
        source = nanoarrow.c_array([True, False, None], nanoarrow.bool_())
        array_handle = ctypes.c_void_p()
        view = BufferViewStruct()

        try:
            _assert_arrow_ok(
                library,
                library.irx_arrow_array_import_copy(
                    source._addr(),
                    source.schema._addr(),
                    ctypes.byref(array_handle),
                ),
            )

            assert (
                library.irx_arrow_array_type_id(array_handle)
                == IRX_ARROW_TYPE_BOOL
            )
            assert (
                library.irx_arrow_array_can_borrow_buffer_view(array_handle)
                == 0
            )

            code = library.irx_arrow_array_borrow_buffer_view(
                array_handle,
                ctypes.byref(view),
            )
            assert code != 0
            assert (
                "bit-packed values" in library.irx_arrow_last_error().decode()
            )
        finally:
            if array_handle.value is not None:
                library.irx_arrow_array_release(array_handle)


def test_arrow_runtime_import_move_adopts_offset_arrays() -> None:
    """
    title: Move import should adopt external C Data and preserve offsets.
    """
    with _load_arrow_runtime_library() as library:
        source = nanoarrow.c_array([10, 20, 30, 40], nanoarrow.int32())
        raw_array = _arrow_array_struct(source._addr())
        raw_array.length = 2
        raw_array.offset = 1
        raw_array.null_count = 0

        array_handle = ctypes.c_void_p()
        view = BufferViewStruct()

        _assert_arrow_ok(
            library,
            library.irx_arrow_array_import_move(
                source._addr(),
                source.schema._addr(),
                ctypes.byref(array_handle),
            ),
        )

        try:
            moved_array = _arrow_array_struct(source._addr())
            moved_schema = _arrow_schema_struct(source.schema._addr())

            assert moved_array.release is None
            assert moved_schema.release is None
            assert library.irx_arrow_array_length(array_handle) == 2  # noqa: PLR2004
            assert library.irx_arrow_array_offset(array_handle) == 1

            _assert_arrow_ok(
                library,
                library.irx_arrow_array_borrow_buffer_view(
                    array_handle,
                    ctypes.byref(view),
                ),
            )

            assert view.dtype == BUFFER_DTYPE_INT32
            assert view.shape[0] == 2  # noqa: PLR2004
            assert view.strides[0] == 4  # noqa: PLR2004
            assert view.offset_bytes == 4  # noqa: PLR2004
            assert view.flags == (
                BUFFER_FLAG_BORROWED
                | BUFFER_FLAG_READONLY
                | BUFFER_FLAG_C_CONTIGUOUS
            )

            exported_schema = allocate_c_schema()
            exported_array = allocate_c_array(exported_schema)
            _assert_arrow_ok(
                library,
                library.irx_arrow_array_export(
                    array_handle,
                    exported_array._addr(),
                    exported_schema._addr(),
                ),
            )
            assert Array(exported_array).to_pylist() == [20, 30]
        finally:
            if array_handle.value is not None:
                library.irx_arrow_array_release(array_handle)


def test_arrow_runtime_export_copy_survives_source_release() -> None:
    """
    title: Export should return independent C Data copies.
    """
    with _load_arrow_runtime_library() as library:
        array_handle = _build_runtime_array(
            library,
            IRX_ARROW_TYPE_INT32,
            "int",
            [4, 5, 6],
        )
        exported_schema = allocate_c_schema()
        exported_array = allocate_c_array(exported_schema)

        _assert_arrow_ok(
            library,
            library.irx_arrow_array_export(
                array_handle,
                exported_array._addr(),
                exported_schema._addr(),
            ),
        )
        library.irx_arrow_array_release(array_handle)

        exported = Array(exported_array)
        assert exported.to_pylist() == [4, 5, 6]


def test_arrow_runtime_schema_handles_roundtrip_supported_schemas() -> None:
    """
    title: Schema handles should import, retain, export, and reapply schemas.
    """
    with _load_arrow_runtime_library() as library:
        source = nanoarrow.c_array([1, 2, 3], nanoarrow.int16())
        schema_handle = ctypes.c_void_p()
        array_handle = ctypes.c_void_p()

        try:
            _assert_arrow_ok(
                library,
                library.irx_arrow_schema_import_copy(
                    source.schema._addr(),
                    ctypes.byref(schema_handle),
                ),
            )
            assert (
                library.irx_arrow_schema_type_id(schema_handle)
                == IRX_ARROW_TYPE_INT16
            )
            assert library.irx_arrow_schema_is_nullable(schema_handle) == 1

            _assert_arrow_ok(
                library,
                library.irx_arrow_schema_retain(schema_handle),
            )
            library.irx_arrow_schema_release(schema_handle)
            assert (
                library.irx_arrow_schema_type_id(schema_handle)
                == IRX_ARROW_TYPE_INT16
            )

            exported_schema = allocate_c_schema()
            _assert_arrow_ok(
                library,
                library.irx_arrow_schema_export(
                    schema_handle,
                    exported_schema._addr(),
                ),
            )

            _assert_arrow_ok(
                library,
                library.irx_arrow_array_import_copy(
                    source._addr(),
                    exported_schema._addr(),
                    ctypes.byref(array_handle),
                ),
            )
            assert (
                library.irx_arrow_array_type_id(array_handle)
                == IRX_ARROW_TYPE_INT16
            )
        finally:
            if array_handle.value is not None:
                library.irx_arrow_array_release(array_handle)
            if schema_handle.value is not None:
                library.irx_arrow_schema_release(schema_handle)


def test_arrow_runtime_rejects_unsupported_string_arrays() -> None:
    """
    title: Unsupported variable-width layouts should fail clearly.
    """
    with _load_arrow_runtime_library() as library:
        source = nanoarrow.c_array(["a", "b"], nanoarrow.string())
        array_handle = ctypes.c_void_p()

        code = library.irx_arrow_array_import_copy(
            source._addr(),
            source.schema._addr(),
            ctypes.byref(array_handle),
        )

        assert code != 0
        assert array_handle.value is None
        assert "Unsupported Arrow storage type" in (
            library.irx_arrow_last_error().decode()
        )

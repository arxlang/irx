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

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import astx
import nanoarrow
import pytest

from arx_nanoarrow_sources import get_include_dir, get_source_files
from irx.arrow import ArrowInt32ArrayLength
from irx.builders.llvmliteir import Builder
from irx.runtime.arrow.feature import (
    IRX_ARROW_TYPE_INT32,
    build_arrow_runtime_feature,
)
from irx.runtime.linking import compile_native_artifacts, link_executable
from llvmlite import binding as llvm
from nanoarrow import Array
from nanoarrow.c_array import allocate_c_array
from nanoarrow.c_schema import allocate_c_schema


def _arrow_length_module(values: list[int]) -> astx.Module:
    module = astx.Module()
    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    body = astx.Block()
    body.append(
        astx.FunctionReturn(
            ArrowInt32ArrayLength(
                [astx.LiteralInt32(value) for value in values]
            )
        )
    )
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))
    return module


def _plain_main_module() -> astx.Module:
    module = astx.Module()
    main_proto = astx.FunctionPrototype(
        "main", args=astx.Arguments(), return_type=astx.Int32()
    )
    body = astx.Block()
    body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.append(astx.FunctionDef(prototype=main_proto, body=body))
    return module


def _compile_arrow_harness(source: str) -> subprocess.CompletedProcess[str]:
    feature = build_arrow_runtime_feature()
    native_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "irx"
        / "runtime"
        / "arrow"
        / "native"
    )
    clang_binary = shutil.which("clang")
    if clang_binary is None:
        pytest.skip("clang is required for Arrow runtime harness tests")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_path = tmp_path / "arrow_harness.c"
        object_path = tmp_path / "arrow_harness.o"
        output_path = tmp_path / "arrow_harness"

        source_path.write_text(textwrap.dedent(source), encoding="utf8")
        subprocess.run(
            [
                clang_binary,
                "-c",
                str(source_path),
                "-o",
                str(object_path),
                "-I",
                str(native_root),
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
        return subprocess.run(
            [str(output_path)],
            check=False,
            capture_output=True,
            text=True,
        )


def _shared_library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"

    return ".so"


@contextmanager
def _load_arrow_runtime_library() -> Iterator[ctypes.CDLL]:
    if sys.platform == "win32":
        pytest.skip("nanoarrow interop shared-library tests require Unix")

    feature = build_arrow_runtime_feature()
    clang_binary = shutil.which("clang")
    if clang_binary is None:
        pytest.skip("clang is required for Arrow runtime interop tests")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_path = (
            tmp_path / f"libirx_arrow_runtime{_shared_library_suffix()}"
        )
        link_inputs = compile_native_artifacts(
            feature.artifacts,
            tmp_path,
            clang_binary,
        )

        command = [clang_binary]
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
    library.irx_arrow_array_null_count.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_null_count.restype = ctypes.c_int64
    library.irx_arrow_array_type_id.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_type_id.restype = ctypes.c_int32
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
    library.irx_arrow_array_release.argtypes = [ctypes.c_void_p]
    library.irx_arrow_array_release.restype = None
    library.irx_arrow_last_error.argtypes = []
    library.irx_arrow_last_error.restype = ctypes.c_char_p


def _assert_arrow_ok(library: ctypes.CDLL, code: int) -> None:
    assert code == 0, library.irx_arrow_last_error().decode()


def _build_runtime_array(
    library: ctypes.CDLL, values: list[int]
) -> ctypes.c_void_p:
    builder = ctypes.c_void_p()
    array_handle = ctypes.c_void_p()

    _assert_arrow_ok(
        library,
        library.irx_arrow_array_builder_int32_new(ctypes.byref(builder)),
    )

    try:
        for value in values:
            _assert_arrow_ok(
                library,
                library.irx_arrow_array_builder_append_int32(builder, value),
            )

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


def test_arrow_symbols_absent_when_unused() -> None:
    """
    title: Arrow runtime declarations should be absent when unused.
    """
    builder = Builder()

    ir_text = builder.translate(_arrow_length_module([]))
    assert "irx_arrow_array_builder_int32_new" in ir_text

    plain_builder = Builder()
    plain_ir = plain_builder.translate(_plain_main_module())
    assert "irx_arrow_" not in plain_ir


def test_arrow_length_codegen_declares_runtime_symbols() -> None:
    """
    title: Arrow lowering should declare runtime symbols and parse as LLVM.
    """
    builder = Builder()
    ir_text = builder.translate(_arrow_length_module([1, 2, 3]))

    llvm.parse_assembly(ir_text)

    active_features = (
        builder.translator.runtime_features.active_feature_names()
    )

    assert "arrow" in active_features
    assert '@"irx_arrow_array_builder_int32_new"' in ir_text
    assert '@"irx_arrow_array_length"' in ir_text
    assert builder.translator.runtime_features.native_artifacts()


def test_arrow_feature_uses_packaged_nanoarrow_sources() -> None:
    """
    title: Arrow runtime should compile against arx-nanoarrow-sources.
    """
    feature = build_arrow_runtime_feature()
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


def test_arrow_length_build_returns_length() -> None:
    """
    title: >-
      Building an Arrow-backed module should link and return the array length.
    """
    builder = Builder()
    module = _arrow_length_module([10, 20, 30])

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
        array_handle = _build_runtime_array(library, [4, 5, 6])
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

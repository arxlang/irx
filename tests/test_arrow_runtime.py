"""
title: Tests for the Arrow runtime feature and lowering path.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap

from pathlib import Path

import astx
import pytest

from irx.builders.llvmliteir import LLVMLiteIR
from irx.runtime.arrow.feature import build_arrow_runtime_feature
from irx.runtime.linking import link_executable
from irx.system import ArrowInt32ArrayLength
from llvmlite import binding as llvm


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


def test_arrow_symbols_absent_when_unused() -> None:
    """
    title: Arrow runtime declarations should be absent when unused.
    """
    builder = LLVMLiteIR()

    ir_text = builder.translate(_arrow_length_module([]))
    assert "irx_arrow_array_builder_int32_new" in ir_text

    plain_builder = LLVMLiteIR()
    plain_ir = plain_builder.translate(_plain_main_module())
    assert "irx_arrow_" not in plain_ir


def test_arrow_length_codegen_declares_runtime_symbols() -> None:
    """
    title: Arrow lowering should declare runtime symbols and parse as LLVM.
    """
    builder = LLVMLiteIR()
    ir_text = builder.translate(_arrow_length_module([1, 2, 3]))

    llvm.parse_assembly(ir_text)

    active_features = (
        builder.translator.runtime_features.active_feature_names()
    )

    assert "arrow" in active_features
    assert '@"irx_arrow_array_builder_int32_new"' in ir_text
    assert '@"irx_arrow_array_length"' in ir_text
    assert builder.translator.runtime_features.native_artifacts()


def test_arrow_length_build_returns_length() -> None:
    """
    title: >-
      Building an Arrow-backed module should link and return the array length.
    """
    builder = LLVMLiteIR()
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

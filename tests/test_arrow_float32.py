"""
title: Tests for ArrowFloat32Array ASTx node lowering.
"""

import shutil
import subprocess

from pathlib import Path

import astx
import pytest

from irx.arrow import ArrowFloat32Array
from irx.builders.llvmliteir import LLVMLiteIR


def _make_float32_array_module(
    values: list[float],
) -> tuple[astx.Module, LLVMLiteIR]:
    """
    title: Build an ASTx module containing an ArrowFloat32Array expression.
    parameters:
      values:
        type: list[float]
    returns:
      type: tuple[astx.Module, LLVMLiteIR]
    """
    builder = LLVMLiteIR()
    module = builder.module()

    proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    block = astx.Block()

    arrow_node = ArrowFloat32Array([astx.LiteralFloat32(v) for v in values])
    block.append(arrow_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(len(values))))

    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    return module, builder


def test_arrow_float32_ir_shape() -> None:
    """
    title: Verify that ArrowFloat32Array emits the correct runtime symbols.
    summary: >-
      Translate-only test (no clang required). Asserts the three float32
      builder symbols appear in the emitted LLVM IR text.
    """
    module, builder = _make_float32_array_module([1.0, 2.0, 3.0])
    ir_text = builder.translate(module)

    assert "irx_arrow_array_builder_float32_new" in ir_text
    assert "irx_arrow_array_builder_append_float32" in ir_text
    assert "irx_arrow_array_builder_finish" in ir_text


@pytest.mark.skipif(
    shutil.which("clang") is None,
    reason="clang not available on PATH",
)
def test_arrow_float32_build_returns_length(
    tmp_path: Path,
) -> None:
    """
    title: Verify that an ArrowFloat32Array binary returns the array length.
    summary: >-
      Builds and executes a native binary containing an ArrowFloat32Array with
      3 elements. Asserts the process exit code equals 3.
    parameters:
      tmp_path:
        type: Path
    """
    module, builder = _make_float32_array_module([1.0, 2.0, 3.0])
    out_path = str(tmp_path / "test_f32_arrow")
    builder.build(module, out_path)
    proc = subprocess.run([out_path], capture_output=True, check=False)
    expected_length = len([1.0, 2.0, 3.0])
    assert proc.returncode == expected_length

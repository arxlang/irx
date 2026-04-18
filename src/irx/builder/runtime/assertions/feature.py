"""
title: Assertion runtime feature declarations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from llvmlite import ir

from irx.builder.runtime.features import (
    ExternalSymbolSpec,
    NativeArtifact,
    RuntimeFeature,
    declare_external_function,
)
from irx.typecheck import typechecked

if TYPE_CHECKING:
    from irx.builder.protocols import VisitorProtocol

ASSERT_RUNTIME_FEATURE_NAME = "assertions"
ASSERT_FAILURE_SYMBOL_NAME = "__arx_assert_fail"


@typechecked
def build_assertions_runtime_feature() -> RuntimeFeature:
    """
    title: Build the assertion runtime feature specification.
    returns:
      type: RuntimeFeature
    """
    runtime_root = Path(__file__).resolve().parent
    native_root = runtime_root / "native"
    return RuntimeFeature(
        name=ASSERT_RUNTIME_FEATURE_NAME,
        symbols={
            ASSERT_FAILURE_SYMBOL_NAME: ExternalSymbolSpec(
                ASSERT_FAILURE_SYMBOL_NAME,
                _declare_assert_failure,
            ),
        },
        artifacts=(
            NativeArtifact(
                kind="c_source",
                path=native_root / "irx_assert_runtime.c",
                include_dirs=(native_root,),
                compile_flags=("-std=c99",),
            ),
        ),
    )


@typechecked
def _declare_assert_failure(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare the assertion failure runtime helper.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.VOID_TYPE,
        [
            visitor._llvm.INT8_TYPE.as_pointer(),
            visitor._llvm.INT32_TYPE,
            visitor._llvm.INT32_TYPE,
            visitor._llvm.INT8_TYPE.as_pointer(),
        ],
    )
    return declare_external_function(
        visitor._llvm.module,
        ASSERT_FAILURE_SYMBOL_NAME,
        fn_type,
    )

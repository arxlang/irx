"""
title: Dynamic-list runtime feature declarations.
summary: >-
  Declares the narrow append/index runtime surface for IRX lists. The current
  ABI intentionally does not expose a destroy/release helper yet, so produced
  list storage remains process-lifetime for now.
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
from irx.builtins.collections.list import (
    LIST_APPEND_SYMBOL,
    LIST_AT_SYMBOL,
)
from irx.typecheck import typechecked

if TYPE_CHECKING:
    from irx.builder.protocols import VisitorProtocol


@typechecked
def build_list_runtime_feature() -> RuntimeFeature:
    """
    title: Build the dynamic-list runtime feature specification.
    returns:
      type: RuntimeFeature
    """
    native_root = Path(__file__).resolve().parent / "native"
    return RuntimeFeature(
        name="list",
        symbols={
            LIST_APPEND_SYMBOL: ExternalSymbolSpec(
                LIST_APPEND_SYMBOL,
                _declare_list_append,
            ),
            LIST_AT_SYMBOL: ExternalSymbolSpec(
                LIST_AT_SYMBOL,
                _declare_list_at,
            ),
        },
        artifacts=(
            NativeArtifact(
                kind="c_source",
                path=native_root / "irx_list_runtime.c",
                include_dirs=(native_root,),
                compile_flags=("-std=c99",),
            ),
        ),
        metadata={
            "canonical_name": "list",
            "symbols": (LIST_APPEND_SYMBOL, LIST_AT_SYMBOL),
            "limitations": (
                "append/index only",
                "no destroy or release helper yet",
            ),
        },
    )


@typechecked
def _list_llvm_type(visitor: VisitorProtocol) -> ir.LiteralStructType:
    """
    title: Return the canonical lowered list ABI type.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.LiteralStructType
    """
    return ir.LiteralStructType(
        [
            visitor._llvm.INT8_TYPE.as_pointer(),
            visitor._llvm.INT64_TYPE,
            visitor._llvm.INT64_TYPE,
            visitor._llvm.INT64_TYPE,
        ]
    )


@typechecked
def _declare_list_append(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare the dynamic-list append helper.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.INT32_TYPE,
        [
            _list_llvm_type(visitor).as_pointer(),
            visitor._llvm.INT8_TYPE.as_pointer(),
        ],
    )
    return declare_external_function(
        visitor._llvm.module,
        LIST_APPEND_SYMBOL,
        fn_type,
    )


@typechecked
def _declare_list_at(visitor: VisitorProtocol) -> ir.Function:
    """
    title: Declare the dynamic-list indexed-access helper.
    parameters:
      visitor:
        type: VisitorProtocol
    returns:
      type: ir.Function
    """
    fn_type = ir.FunctionType(
        visitor._llvm.INT8_TYPE.as_pointer(),
        [
            _list_llvm_type(visitor).as_pointer(),
            visitor._llvm.INT64_TYPE,
        ],
    )
    return declare_external_function(
        visitor._llvm.module,
        LIST_AT_SYMBOL,
        fn_type,
    )

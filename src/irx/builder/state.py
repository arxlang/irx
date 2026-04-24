"""
title: Shared state typing for llvmlite-based codegen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from llvmlite import ir

from irx.typecheck import typechecked

ResultStackValue = ir.Value | ir.Function | None
NamedValueMap = dict[str, Any]
CleanupEmitter = Callable[[], None]


@typechecked
@dataclass(frozen=True)
class LoopTargets:
    """
    title: Canonical break/continue targets for one active loop.
    attributes:
      break_target:
        type: ir.Block
      continue_target:
        type: ir.Block
      cleanup_depth:
        type: int
    """

    break_target: ir.Block
    continue_target: ir.Block
    cleanup_depth: int = 0


__all__ = [
    "CleanupEmitter",
    "LoopTargets",
    "NamedValueMap",
    "ResultStackValue",
]

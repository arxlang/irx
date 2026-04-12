"""
title: Shared state typing for llvmlite-based codegen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llvmlite import ir

from irx.typecheck import typechecked

ResultStackValue = ir.Value | ir.Function | None
NamedValueMap = dict[str, Any]


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
    """

    break_target: ir.Block
    continue_target: ir.Block


__all__ = ["LoopTargets", "NamedValueMap", "ResultStackValue"]

"""
title: Shared concrete core for LLVMLiteIR visitors.
"""

from __future__ import annotations

from typing import Any

import astx

from irx.analysis import analyze
from irx.builders._llvmliteir_legacy import (
    LLVMLiteIRVisitor as _LegacyLLVMLiteIRVisitor,
)
from irx.runtime.registry import RuntimeFeatureState

_LEGACY_VISIT = _LegacyLLVMLiteIRVisitor.visit


def _dispatch_legacy_visit(self: Any, node: astx.AST) -> None:
    """
    title: Delegate one visit overload to the legacy implementation.
    parameters:
      self:
        type: Any
      node:
        type: astx.AST
    """
    _LEGACY_VISIT(self, node)


class _LLVMLiteIRVisitorCore(_LegacyLLVMLiteIRVisitor):
    """
    title: Shared concrete core for LLVMLiteIR visitor state and helpers.
    attributes:
      runtime_features:
        type: RuntimeFeatureState
    """

    runtime_features: RuntimeFeatureState

    def translate(self, node: astx.AST) -> str:
        """
        title: Analyze and lower an AST to LLVM IR text.
        parameters:
          node:
            type: astx.AST
        returns:
          type: str
        """
        analyzed = analyze(node)
        return super().translate(analyzed)


__all__ = ["_LLVMLiteIRVisitorCore", "_dispatch_legacy_visit"]

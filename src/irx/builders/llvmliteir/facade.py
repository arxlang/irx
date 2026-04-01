"""
title: Public LLVMLiteIR facade and composed visitor.
"""

from __future__ import annotations

from public import public

from irx.builders._llvmliteir_legacy import (
    LLVMLiteIR as _LegacyLLVMLiteIR,
)
from irx.builders.llvmliteir.core import _LLVMLiteIRVisitorCore
from irx.builders.llvmliteir.visitors import (
    ArrowVisitorMixin,
    BinaryOpVisitorMixin,
    ControlFlowVisitorMixin,
    FunctionVisitorMixin,
    LiteralVisitorMixin,
    ModuleVisitorMixin,
    SystemVisitorMixin,
    TemporalVisitorMixin,
    UnaryOpVisitorMixin,
    VariableVisitorMixin,
)


@public
class LLVMLiteIRVisitor(
    _LLVMLiteIRVisitorCore,
    LiteralVisitorMixin,
    VariableVisitorMixin,
    UnaryOpVisitorMixin,
    BinaryOpVisitorMixin,
    ControlFlowVisitorMixin,
    FunctionVisitorMixin,
    TemporalVisitorMixin,
    ArrowVisitorMixin,
    SystemVisitorMixin,
    ModuleVisitorMixin,
):
    """
    title: Package-composed LLVMLiteIR visitor.
    """


@public
class LLVMLiteIR(_LegacyLLVMLiteIR):
    """
    title: Public LLVMLiteIR builder facade.
    attributes:
      translator:
        type: LLVMLiteIRVisitor
    """

    translator: LLVMLiteIRVisitor

    def __init__(self) -> None:
        super().__init__()
        self.translator = self._new_translator()

    def _new_translator(self) -> LLVMLiteIRVisitor:
        return LLVMLiteIRVisitor(
            active_runtime_features=set(self.runtime_feature_names)
        )

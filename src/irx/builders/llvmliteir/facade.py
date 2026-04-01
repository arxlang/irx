"""
title: Public llvmliteir facade and composed backend classes.
"""

from __future__ import annotations

from public import public

from irx.builders._llvmliteir_legacy import (
    LLVMLiteIR as _LegacyBuilder,
)
from irx.builders.llvmliteir.core import _VisitorCore
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
class Visitor(
    _VisitorCore,
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
    title: Package-composed backend visitor.
    """


@public
class Builder(_LegacyBuilder):
    """
    title: Public llvmliteir backend facade.
    attributes:
      translator:
        type: Visitor
    """

    translator: Visitor

    def __init__(self) -> None:
        super().__init__()
        self.translator = self._new_translator()

    def _new_translator(self) -> Visitor:
        return Visitor(active_runtime_features=set(self.runtime_feature_names))

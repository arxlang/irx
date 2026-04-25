# mypy: disable-error-code=no-redef
# mypy: disable-error-code=untyped-decorator
# mypy: disable-error-code=attr-defined

"""
title: Expression array visitors.
summary: >-
  Handle array-specific semantic helpers that are separate from tensor
  expression semantics.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.types import is_integer_type
from irx.typecheck import typechecked


@typechecked
class ExpressionArrayVisitorMixin(SemanticVisitorMixinBase):
    """
    title: Expression array visitors.
    """

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ArrayInt32ArrayLength) -> None:
        """
        title: Visit ArrayInt32ArrayLength nodes.
        parameters:
          node:
            type: astx.ArrayInt32ArrayLength
        """
        for item in node.values:
            self.visit(item)
            if not is_integer_type(self._expr_type(item)):
                self.context.diagnostics.add(
                    "Array helper supports only integer expressions",
                    node=item,
                )
        self._set_type(node, astx.Int32())

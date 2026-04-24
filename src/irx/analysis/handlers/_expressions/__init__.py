"""
title: Expression-oriented semantic visitors.
summary: >-
  Resolve lexical identifiers, visible function names, and expression typing
  rules while delegating the implementation to smaller concern-focused mixins.
"""

from __future__ import annotations

from irx.analysis.handlers._expressions.arrays_buffers import (
    ExpressionArrayBufferVisitorMixin,
)
from irx.analysis.handlers._expressions.classes import (
    ExpressionClassVisitorMixin,
)
from irx.analysis.handlers._expressions.literals import (
    ExpressionLiteralVisitorMixin,
)
from irx.analysis.handlers._expressions.modules import (
    ExpressionModuleVisitorMixin,
)
from irx.analysis.handlers._expressions.mutation import (
    ExpressionMutationVisitorMixin,
)
from irx.analysis.handlers._expressions.operators import (
    ExpressionOperatorVisitorMixin,
)
from irx.typecheck import typechecked

__all__ = ["ExpressionVisitorMixin"]


@typechecked
class ExpressionVisitorMixin(
    ExpressionModuleVisitorMixin,
    ExpressionMutationVisitorMixin,
    ExpressionClassVisitorMixin,
    ExpressionOperatorVisitorMixin,
    ExpressionArrayBufferVisitorMixin,
    ExpressionLiteralVisitorMixin,
):
    """
    title: Expression-oriented semantic visitors.
    summary: >-
      Compose the expression-focused semantic mixins so the analyzer keeps the
      same visitor surface while the implementation stays split by concern.
    """

"""
title: Expression-oriented ndarray and buffer-view visitors.
summary: >-
  Compose the ndarray and buffer-view visitor overloads into the public array-
  and-buffer expression mixin.
"""

from __future__ import annotations

from irx.analysis.handlers._expressions.buffer_views import (
    ExpressionBufferViewVisitorMixin,
)
from irx.analysis.handlers._expressions.ndarrays import (
    ExpressionNDArrayVisitorMixin,
)
from irx.typecheck import typechecked


@typechecked
class ExpressionArrayBufferVisitorMixin(
    ExpressionNDArrayVisitorMixin,
    ExpressionBufferViewVisitorMixin,
):
    """
    title: Expression-oriented ndarray and buffer-view visitors.
    """

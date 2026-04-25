"""
title: Expression-oriented tensor and buffer-view visitors.
summary: >-
  Compose the tensor and buffer-view visitor overloads into the public tensor-
  and-buffer expression mixin.
"""

from __future__ import annotations

from irx.analysis.handlers._expressions.buffer_views import (
    ExpressionBufferViewVisitorMixin,
)
from irx.analysis.handlers._expressions.tensors import (
    ExpressionTensorVisitorMixin,
)
from irx.typecheck import typechecked


@typechecked
class ExpressionTensorBufferVisitorMixin(
    ExpressionTensorVisitorMixin,
    ExpressionBufferViewVisitorMixin,
):
    """
    title: Expression-oriented tensor and buffer-view visitors.
    """

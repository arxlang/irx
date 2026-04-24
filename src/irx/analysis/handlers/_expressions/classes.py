"""
title: Expression-oriented class visitors.
summary: >-
  Compose the class-lookup helpers and class-access visitor overloads into the
  public class-focused expression mixin.
"""

from __future__ import annotations

from irx.analysis.handlers._expressions.class_access import (
    ExpressionClassAccessVisitorMixin,
)
from irx.typecheck import typechecked


@typechecked
class ExpressionClassVisitorMixin(ExpressionClassAccessVisitorMixin):
    """
    title: Expression-oriented class visitors.
    """

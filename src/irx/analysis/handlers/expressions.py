"""
title: Expression-oriented semantic visitors.
summary: >-
  Preserve the public expression-handler import path while the split expression
  mixins live in the private `_expressions` package.
"""

from __future__ import annotations

from irx.analysis.handlers._expressions import ExpressionVisitorMixin

__all__ = ["ExpressionVisitorMixin"]

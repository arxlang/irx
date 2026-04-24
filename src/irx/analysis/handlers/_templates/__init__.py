"""
title: Template-specialization semantic helpers.
summary: >-
  Compose the split template-specialization helpers into the internal template
  visitor mixin used by the semantic analyzer.
"""

from __future__ import annotations

from irx.analysis.handlers._templates.analysis import (
    TemplateAnalysisVisitorMixin,
)
from irx.typecheck import typechecked

__all__ = ["TemplateVisitorMixin"]


@typechecked
class TemplateVisitorMixin(TemplateAnalysisVisitorMixin):
    """
    title: Template-specialization semantic helpers.
    """

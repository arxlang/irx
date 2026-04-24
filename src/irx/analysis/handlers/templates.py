"""
title: Template-specialization semantic helpers.
summary: >-
  Preserve the public template-handler import path while the split template
  mixins live in the private `_templates` package.
"""

from __future__ import annotations

from irx.analysis.handlers._templates import TemplateVisitorMixin

__all__ = ["TemplateVisitorMixin"]

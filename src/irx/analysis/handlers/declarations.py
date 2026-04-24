"""
title: Declaration-oriented semantic visitors.
summary: >-
  Preserve the public declaration-handler import path while the split
  declaration mixins live in the private `_declarations` package.
"""

from __future__ import annotations

from irx.analysis.handlers._declarations import DeclarationVisitorMixin

__all__ = ["DeclarationVisitorMixin"]

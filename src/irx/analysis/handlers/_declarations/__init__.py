"""
title: Declaration-oriented semantic visitors.
summary: >-
  Handle modules, functions, structs, and lexical declarations while delegating
  semantic entity creation and registration to smaller concern-focused mixins.
"""

from __future__ import annotations

from irx.analysis.handlers._declarations.blocks import (
    DeclarationBlockVisitorMixin,
)
from irx.analysis.handlers._declarations.class_layout import (
    DeclarationClassLayoutVisitorMixin,
)
from irx.analysis.handlers._declarations.class_members import (
    DeclarationClassMemberVisitorMixin,
)
from irx.analysis.handlers._declarations.class_support import (
    DeclarationClassSupportVisitorMixin,
)
from irx.analysis.handlers._declarations.functions import (
    DeclarationFunctionVisitorMixin,
)
from irx.analysis.handlers._declarations.structs import (
    DeclarationStructVisitorMixin,
)
from irx.typecheck import typechecked

__all__ = ["DeclarationVisitorMixin"]


@typechecked
class DeclarationVisitorMixin(
    DeclarationFunctionVisitorMixin,
    DeclarationStructVisitorMixin,
    DeclarationClassSupportVisitorMixin,
    DeclarationClassLayoutVisitorMixin,
    DeclarationClassMemberVisitorMixin,
    DeclarationBlockVisitorMixin,
):
    """
    title: Declaration-oriented semantic visitors.
    summary: >-
      Compose the declaration-focused semantic mixins so the analyzer keeps the
      same visitor surface while the implementation stays split by concern.
    """

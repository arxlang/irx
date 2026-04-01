# mypy: disable-error-code=no-redef

"""
title: System/runtime visitor mixins for LLVMLiteIR.
"""

from plum import dispatch

from irx import system
from irx.builders.llvmliteir.core import _dispatch_legacy_visit


class SystemVisitorMixin:
    @dispatch
    def visit(self, node: system.Cast) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: system.PrintExpr) -> None:
        _dispatch_legacy_visit(self, node)

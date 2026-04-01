# mypy: disable-error-code=no-redef

"""
title: Variable visitor mixins for LLVMLiteIR.
"""

import astx

from plum import dispatch

from irx.builders.llvmliteir.core import _dispatch_legacy_visit


class VariableVisitorMixin:
    @dispatch
    def visit(self, expr: astx.VariableAssignment) -> None:
        _dispatch_legacy_visit(self, expr)

    @dispatch
    def visit(self, node: astx.Identifier) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.VariableDeclaration) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.InlineVariableDeclaration) -> None:
        _dispatch_legacy_visit(self, node)

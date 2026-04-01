# mypy: disable-error-code=no-redef

"""
title: Function visitor mixins for LLVMLiteIR.
"""

import astx

from plum import dispatch

from irx.builders.llvmliteir.core import _dispatch_legacy_visit


class FunctionVisitorMixin:
    @dispatch
    def visit(self, node: astx.FunctionCall) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.FunctionDef) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.FunctionPrototype) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.FunctionReturn) -> None:
        _dispatch_legacy_visit(self, node)

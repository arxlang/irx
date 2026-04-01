"""
title: Unary-operator visitor mixins for LLVMLiteIR.
"""

import astx

from plum import dispatch

from irx.builders.llvmliteir.core import _dispatch_legacy_visit


class UnaryOpVisitorMixin:
    @dispatch
    def visit(self, node: astx.UnaryOp) -> None:
        _dispatch_legacy_visit(self, node)

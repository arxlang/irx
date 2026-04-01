"""
title: Binary-operator visitor mixins for LLVMLiteIR.
"""

import astx

from plum import dispatch

from irx.builders.llvmliteir.core import _dispatch_legacy_visit


class BinaryOpVisitorMixin:
    @dispatch
    def visit(self, node: astx.BinaryOp) -> None:
        _dispatch_legacy_visit(self, node)

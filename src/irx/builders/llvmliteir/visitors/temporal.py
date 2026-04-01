# mypy: disable-error-code=no-redef

"""
title: Temporal literal visitor mixins for LLVMLiteIR.
"""

import astx

from plum import dispatch

from irx.builders.llvmliteir.core import _dispatch_legacy_visit


class TemporalVisitorMixin:
    @dispatch
    def visit(self, node: astx.LiteralTime) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralTimestamp) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralDateTime) -> None:
        _dispatch_legacy_visit(self, node)

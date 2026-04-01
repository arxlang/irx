"""
title: Arrow visitor mixins for LLVMLiteIR.
"""

from plum import dispatch

from irx import arrow as irx_arrow
from irx.builders.llvmliteir.core import _dispatch_legacy_visit


class ArrowVisitorMixin:
    @dispatch
    def visit(self, node: irx_arrow.ArrowInt32ArrayLength) -> None:
        _dispatch_legacy_visit(self, node)

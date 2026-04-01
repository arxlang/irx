# mypy: disable-error-code=no-redef

"""
title: Module-level visitor mixins for LLVMLiteIR.
"""

import astx

from plum import dispatch

from irx.builders.llvmliteir.core import _dispatch_legacy_visit


class ModuleVisitorMixin:
    @dispatch
    def visit(self, node: astx.Module) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.StructDefStmt) -> None:
        _dispatch_legacy_visit(self, node)

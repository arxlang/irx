# mypy: disable-error-code=no-redef

"""
title: Control-flow visitor mixins for LLVMLiteIR.
"""

import astx

from plum import dispatch

from irx.builders.llvmliteir.core import _dispatch_legacy_visit


class ControlFlowVisitorMixin:
    @dispatch
    def visit(self, block: astx.Block) -> None:
        _dispatch_legacy_visit(self, block)

    @dispatch
    def visit(self, node: astx.IfStmt) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, expr: astx.WhileStmt) -> None:
        _dispatch_legacy_visit(self, expr)

    @dispatch
    def visit(self, node: astx.ForCountLoopStmt) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.ForRangeLoopStmt) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.BreakStmt) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.ContinueStmt) -> None:
        _dispatch_legacy_visit(self, node)

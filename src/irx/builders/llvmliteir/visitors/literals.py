# mypy: disable-error-code=no-redef

"""
title: Literal visitor mixins for LLVMLiteIR.
"""

import astx

from plum import dispatch

from irx.builders.llvmliteir.core import _dispatch_legacy_visit


class LiteralVisitorMixin:
    @dispatch
    def visit(self, node: astx.LiteralInt32) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, expr: astx.LiteralFloat32) -> None:
        _dispatch_legacy_visit(self, expr)

    @dispatch
    def visit(self, expr: astx.LiteralFloat64) -> None:
        _dispatch_legacy_visit(self, expr)

    @dispatch
    def visit(self, node: astx.LiteralFloat16) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, expr: astx.LiteralNone) -> None:
        _dispatch_legacy_visit(self, expr)

    @dispatch
    def visit(self, node: astx.LiteralBoolean) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralInt64) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralInt8) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralUInt8) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralUInt16) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralUInt32) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralUInt64) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralUInt128) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, expr: astx.LiteralUTF8Char) -> None:
        _dispatch_legacy_visit(self, expr)

    @dispatch
    def visit(self, expr: astx.LiteralUTF8String) -> None:
        _dispatch_legacy_visit(self, expr)

    @dispatch
    def visit(self, expr: astx.LiteralString) -> None:
        _dispatch_legacy_visit(self, expr)

    @dispatch
    def visit(self, node: astx.LiteralList) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralSet) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralTuple) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralDict) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.SubscriptExpr) -> None:
        _dispatch_legacy_visit(self, node)

    @dispatch
    def visit(self, node: astx.LiteralInt16) -> None:
        _dispatch_legacy_visit(self, node)

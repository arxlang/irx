# mypy: disable-error-code=no-redef

"""
title: Tests for the shared visitor base.
"""

import astx
import pytest

from irx.analysis.facade import SemanticAnalyzer
from irx.builders.llvmliteir import Visitor
from irx.visitors import BaseVisitor
from plum import dispatch


class _SpecializedVisitor(BaseVisitor):
    """
    title: Minimal visitor used to validate shared dispatch behavior.
    attributes:
      calls:
        type: list[str]
    """

    calls: list[str]

    def __init__(self) -> None:
        self.calls = []

    @dispatch
    def visit(self, node: astx.LiteralInt32) -> None:
        self.calls.append(type(node).__name__)


def test_base_visitor_raises_not_implemented() -> None:
    """
    title: BaseVisitor should raise a consistent error for ASTx nodes.
    """
    visitor = BaseVisitor()
    with pytest.raises(
        NotImplementedError,
        match=r"BaseVisitor\.visit\(LiteralInt32\) is not implemented",
    ):
        visitor.visit(astx.LiteralInt32(1))


def test_specialized_visitor_dispatch_overrides_base_default() -> None:
    """
    title: Specialized visit overloads should win over base defaults.
    """
    visitor = _SpecializedVisitor()

    visitor.visit(astx.LiteralInt32(1))

    assert visitor.calls == ["LiteralInt32"]


def test_semantic_analyzer_inherits_shared_visit_contract() -> None:
    """
    title: SemanticAnalyzer should override the shared visitor defaults.
    """
    node = astx.LiteralInt32(1)

    analyzed = SemanticAnalyzer().analyze(node)

    assert analyzed is node
    assert getattr(node, "semantic", None) is not None


def test_llvmlite_visitor_inherits_shared_visit_contract() -> None:
    """
    title: LLVM visitor mixins should override the shared visitor defaults.
    """
    visitor = Visitor()

    visitor.visit(astx.LiteralInt32(1))

    assert visitor.result_stack

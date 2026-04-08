# mypy: disable-error-code=no-redef

"""
title: Control-flow semantic visitors.
summary: >-
  Handle returns, loops, and branch-level validation while using shared
  registry helpers for lexical declarations introduced by loops.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.types import is_assignable
from irx.typecheck import typechecked


@typechecked
class ControlFlowVisitorMixin(SemanticVisitorMixinBase):
    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionReturn) -> None:
        """
        title: Visit FunctionReturn nodes.
        parameters:
          node:
            type: astx.FunctionReturn
        """
        if self.context.current_function is None:
            self.context.diagnostics.add(
                "Return statement outside function.",
                node=node,
            )
            return
        if node.value is not None:
            self.visit(node.value)
        return_type = self.context.current_function.return_type
        value_type = self._expr_type(node.value)
        if not is_assignable(return_type, value_type):
            self.context.diagnostics.add(
                "Return type mismatch.",
                node=node,
            )
        self._set_type(node, return_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.IfStmt) -> None:
        """
        title: Visit IfStmt nodes.
        parameters:
          node:
            type: astx.IfStmt
        """
        self.visit(node.condition)
        self.visit(node.then)
        if node.else_ is not None:
            self.visit(node.else_)
        self._set_type(node, None)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.WhileStmt) -> None:
        """
        title: Visit WhileStmt nodes.
        parameters:
          node:
            type: astx.WhileStmt
        """
        self.visit(node.condition)
        with self.context.in_loop():
            self.visit(node.body)
        self._set_type(node, None)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ForCountLoopStmt) -> None:
        """
        title: Visit ForCountLoopStmt nodes.
        parameters:
          node:
            type: astx.ForCountLoopStmt
        """
        with self.context.scope("for-count"):
            if node.initializer.value is not None:
                self.visit(node.initializer.value)
            symbol = self.registry.declare_local(
                node.initializer.name,
                node.initializer.type_,
                is_mutable=(
                    node.initializer.mutability != astx.MutabilityKind.constant
                ),
                declaration=node.initializer,
            )
            self._set_symbol(node.initializer, symbol)
            self.visit(node.condition)
            self.visit(node.update)
            with self.context.in_loop():
                self.visit(node.body)
        self._set_type(node, None)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ForRangeLoopStmt) -> None:
        """
        title: Visit ForRangeLoopStmt nodes.
        parameters:
          node:
            type: astx.ForRangeLoopStmt
        """
        with self.context.scope("for-range"):
            self.visit(node.start)
            self.visit(node.end)
            if not isinstance(node.step, astx.LiteralNone):
                self.visit(node.step)
            symbol = self.registry.declare_local(
                node.variable.name,
                node.variable.type_,
                is_mutable=(
                    node.variable.mutability != astx.MutabilityKind.constant
                ),
                declaration=node.variable,
            )
            self._set_symbol(node.variable, symbol)
            with self.context.in_loop():
                self.visit(node.body)
        self._set_type(node, None)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BreakStmt) -> None:
        """
        title: Visit BreakStmt nodes.
        parameters:
          node:
            type: astx.BreakStmt
        """
        if self.context.loop_depth <= 0:
            self.context.diagnostics.add(
                "Break statement outside loop.",
                node=node,
            )
        self._set_type(node, None)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ContinueStmt) -> None:
        """
        title: Visit ContinueStmt nodes.
        parameters:
          node:
            type: astx.ContinueStmt
        """
        if self.context.loop_depth <= 0:
            self.context.diagnostics.add(
                "Continue statement outside loop.",
                node=node,
            )
        self._set_type(node, None)

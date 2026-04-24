# mypy: disable-error-code=no-redef

"""
title: Control-flow semantic visitors.
summary: >-
  Handle returns, loops, and branch-level validation while using shared
  registry helpers for lexical declarations introduced by loops.
"""

from __future__ import annotations

from dataclasses import replace

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.iterables import resolve_iteration_capability
from irx.analysis.types import (
    display_type_name,
    is_boolean_type,
    is_float_type,
    is_integer_type,
    is_string_type,
)
from irx.analysis.validation import resolve_return
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ControlFlowVisitorMixin(SemanticVisitorMixinBase):
    def _validate_boolean_condition(
        self,
        condition: astx.AST,
        *,
        label: str,
    ) -> None:
        """
        title: Validate a control-flow condition is Boolean.
        parameters:
          condition:
            type: astx.AST
          label:
            type: str
        """
        if not self._require_value_expression(condition, context=label):
            return
        condition_type = self._expr_type(condition)
        if condition_type is None or is_boolean_type(condition_type):
            return
        self.context.diagnostics.add(
            f"{label} condition must be Boolean, got "
            f"{display_type_name(condition_type)}",
            node=condition,
            code=DiagnosticCodes.SEMANTIC_INVALID_CONDITION,
        )

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.AssertStmt) -> None:
        """
        title: Visit AssertStmt nodes.
        parameters:
          node:
            type: astx.AssertStmt
        """
        self.visit(node.condition)
        self._validate_boolean_condition(node.condition, label="assert")

        if node.message is not None:
            self.visit(node.message)
            if self._require_value_expression(
                node.message,
                context="AssertStmt message",
            ):
                message_type = self._expr_type(node.message)
                if not (
                    is_string_type(message_type)
                    or is_integer_type(message_type)
                    or is_float_type(message_type)
                    or is_boolean_type(message_type)
                ):
                    self.context.diagnostics.add(
                        "unsupported AssertStmt message type "
                        f"{display_type_name(message_type)}",
                        node=node.message,
                        code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                    )

        self._set_type(node, None)

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
                code=DiagnosticCodes.SEMANTIC_INVALID_RETURN,
            )
            return
        if node.value is not None:
            self.visit(node.value)
        return_resolution = resolve_return(
            self.context.diagnostics,
            function=self.context.current_function,
            value=node.value,
            value_type=self._expr_type(node.value),
            node=node,
        )
        self._set_return(node, return_resolution)
        self._set_type(node, return_resolution.expected_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.IfStmt) -> None:
        """
        title: Visit IfStmt nodes.
        parameters:
          node:
            type: astx.IfStmt
        """
        self.visit(node.condition)
        self._validate_boolean_condition(node.condition, label="if")
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
        self._validate_boolean_condition(node.condition, label="while")
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
            self._validate_boolean_condition(
                node.condition,
                label="for-count loop",
            )
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
    def visit(self, node: astx.ForInLoopStmt) -> None:
        """
        title: Visit ForInLoopStmt nodes.
        parameters:
          node:
            type: astx.ForInLoopStmt
        """
        with self.context.scope("for-in"):
            self.visit(node.iterable)
            iterable_type = self._expr_type(node.iterable)
            iteration = resolve_iteration_capability(
                node.iterable,
                iterable_type,
            )
            if iteration is None:
                self.context.diagnostics.add(
                    "for-in requires an iterable value, got "
                    f"{display_type_name(iterable_type)}",
                    node=node.iterable,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
                self._set_iteration(node, None)
                self._set_type(node, None)
                return

            symbol = self._declare_iteration_target(
                node.target,
                iteration.element_type,
                kind="for-in",
            )
            resolved_iteration = replace(iteration, target_symbol=symbol)
            self._set_iteration(node.iterable, resolved_iteration)
            self._set_iteration(node, resolved_iteration)
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
                code=DiagnosticCodes.SEMANTIC_INVALID_CONTROL_FLOW,
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
                code=DiagnosticCodes.SEMANTIC_INVALID_CONTROL_FLOW,
            )
        self._set_type(node, None)

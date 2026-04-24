# mypy: disable-error-code=no-redef

"""
title: Control-flow semantic visitors.
summary: >-
  Handle returns, loops, and branch-level validation while using shared
  registry helpers for lexical declarations introduced by loops.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from astx.types import AnyType as AstxAnyType

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.iterables import resolve_iteration_capability
from irx.analysis.resolved_nodes import (
    MethodDispatchKind,
    ResolvedContextManager,
    ResolvedMethodCall,
    SemanticClass,
    SemanticSymbol,
)
from irx.analysis.types import (
    display_type_name,
    is_assignable,
    is_boolean_type,
    is_float_type,
    is_integer_type,
    is_none_type,
    is_string_type,
)
from irx.analysis.validation import resolve_return, validate_call
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ControlFlowVisitorMixin(SemanticVisitorMixinBase):
    def _resolve_context_method(
        self,
        node: astx.WithStmt,
        class_: SemanticClass,
        method_name: str,
        manager_type: astx.DataType,
    ) -> ResolvedMethodCall | None:
        """
        title: Resolve one context-manager protocol method.
        parameters:
          node:
            type: astx.WithStmt
          class_:
            type: SemanticClass
          method_name:
            type: str
          manager_type:
            type: astx.DataType
        returns:
          type: ResolvedMethodCall | None
        """
        resolved_overload = cast(Any, self)._resolve_method_overload(
            class_,
            method_name,
            [],
            is_static=False,
            node=node,
        )
        if resolved_overload is None:
            return None

        member, candidates = resolved_overload
        function = member.lowered_function
        if function is None:
            raise TypeError(
                f"context manager method '{method_name}' must have a "
                "lowered function"
            )

        visible_function = cast(Any, self)._visible_method_function(
            class_,
            member,
        )
        if function.template_params:
            specialized_function = self._resolve_template_method_call_target(
                function,
                visible_function,
                [],
                node,
            )
            if specialized_function is None:
                return None
            visible_function = self._specialize_signature(
                visible_function,
                self._specialization_bindings_map(specialized_function),
            )
            function = specialized_function

        call_resolution = validate_call(
            self.context.diagnostics,
            function=visible_function,
            arg_types=[],
            node=node,
        )
        dispatch_kind = MethodDispatchKind.DIRECT
        if member.dispatch_slot is not None:
            dispatch_kind = MethodDispatchKind.INDIRECT

        return ResolvedMethodCall(
            class_=class_,
            member=member,
            function=function,
            overload_key=(member.signature_key or member.qualified_name),
            dispatch_kind=dispatch_kind,
            call=call_resolution,
            candidates=candidates,
            receiver_type=manager_type,
            receiver_class=class_,
            slot_index=member.dispatch_slot,
        )

    def _declare_context_target(
        self,
        target: astx.AST,
        enter_type: astx.DataType,
    ) -> SemanticSymbol | None:
        """
        title: Declare one with-statement target binding.
        parameters:
          target:
            type: astx.AST
          enter_type:
            type: astx.DataType
        returns:
          type: SemanticSymbol | None
        """
        if is_none_type(enter_type):
            self.context.diagnostics.add(
                "with target requires __enter__ to return a value",
                node=target,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
            return None

        if isinstance(target, astx.Identifier):
            symbol = self.registry.declare_local(
                target.name,
                enter_type,
                is_mutable=False,
                declaration=target,
                kind="with",
            )
            self._set_symbol(target, symbol)
            return symbol

        if not isinstance(
            target,
            (astx.InlineVariableDeclaration, astx.VariableDeclaration),
        ):
            self.context.diagnostics.add(
                "with target must be an identifier or inline variable "
                "declaration",
                node=target,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
            return None

        if target.value is not None and not isinstance(
            target.value,
            astx.Undefined,
        ):
            self.context.diagnostics.add(
                "with target declaration must not define an initializer",
                node=target,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )

        if isinstance(target.type_, AstxAnyType):
            target.type_ = enter_type
        else:
            self._resolve_declared_type(target.type_, node=target)
            if not is_assignable(target.type_, enter_type):
                self.context.diagnostics.add(
                    "with target "
                    f"'{target.name}' expects "
                    f"{display_type_name(target.type_)} but __enter__ "
                    f"returns {display_type_name(enter_type)}",
                    node=target,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )

        symbol = self.registry.declare_local(
            target.name,
            target.type_,
            is_mutable=target.mutability != astx.MutabilityKind.constant,
            declaration=target,
            kind="with",
        )
        self._set_symbol(target, symbol)
        return symbol

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
    def visit(self, node: astx.WithStmt) -> None:
        """
        title: Visit WithStmt nodes.
        parameters:
          node:
            type: astx.WithStmt
        """
        self.visit(node.manager)
        if not self._require_value_expression(
            node.manager,
            context="with manager",
        ):
            self._set_type(node, None)
            return

        manager_type = self._expr_type(node.manager)
        class_ = self._resolve_class_from_type(
            manager_type,
            node=node.manager,
            unknown_message="with manager uses unknown class '{name}'",
        )
        if class_ is None:
            if not isinstance(manager_type, astx.ClassType):
                self.context.diagnostics.add(
                    "with manager requires a class value, got "
                    f"{display_type_name(manager_type)}",
                    node=node.manager,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            with self.context.scope("with"):
                self.visit(node.body)
            self._set_type(node, None)
            return

        resolved_manager_type = cast(astx.DataType, manager_type)
        enter = self._resolve_context_method(
            node,
            class_,
            "__enter__",
            resolved_manager_type,
        )
        exit_ = self._resolve_context_method(
            node,
            class_,
            "__exit__",
            resolved_manager_type,
        )

        target_symbol = None
        with self.context.scope("with"):
            if node.target is not None and enter is not None:
                target_symbol = self._declare_context_target(
                    node.target,
                    enter.call.result_type,
                )
            self.visit(node.body)

        if enter is not None and exit_ is not None:
            self._set_context_manager(
                node,
                ResolvedContextManager(
                    class_=class_,
                    manager_type=resolved_manager_type,
                    enter=enter,
                    exit=exit_,
                    target_symbol=target_symbol,
                ),
            )
        else:
            self._set_context_manager(node, None)
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

"""
title: Template-specialization state helpers.
summary: >-
  Reset template-analysis state and manage AST-side metadata used across
  specialization generation.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.handlers._templates.support import (
    TemplateSupportVisitorMixin,
)
from irx.analysis.resolved_nodes import SemanticFunction
from irx.typecheck import typechecked

_SPECIALIZATION_ANALYZED_ATTR = "irx_template_specialization_analyzed"

_OWNER_MODULE_ATTR = "irx_owner_module"

_TEMPLATE_PREPARED_ATTR = "irx_template_specializations_prepared"


@typechecked
class TemplateStateVisitorMixin(TemplateSupportVisitorMixin):
    """
    title: Template-specialization state and AST-reset helpers
    """

    def _clear_semantic_sidecars(self, node: astx.AST) -> None:
        """
        title: Remove semantic sidecars from one cloned AST subtree.
        parameters:
          node:
            type: astx.AST
        """
        seen: set[int] = set()

        def clear(current: astx.AST) -> None:
            """
            title: Clear semantic sidecars from one reachable AST node.
            parameters:
              current:
                type: astx.AST
            """
            current_id = id(current)
            if current_id in seen:
                return
            seen.add(current_id)
            if hasattr(current, "semantic"):
                delattr(current, "semantic")
            for value in vars(current).values():
                if isinstance(value, astx.AST):
                    clear(value)
                    continue
                if isinstance(value, list | tuple):
                    for item in value:
                        if isinstance(item, astx.AST):
                            clear(item)

        clear(node)

    def _reset_template_analysis_state(self, module: astx.Module) -> None:
        """
        title: Reset per-run template state attached to one module AST.
        parameters:
          module:
            type: astx.Module
        """
        astx.clear_generated_template_nodes(module)
        seen: set[int] = set()

        def clear(current: astx.AST) -> None:
            """
            title: Clear template-analysis markers from one reachable AST node.
            parameters:
              current:
                type: astx.AST
            """
            current_id = id(current)
            if current_id in seen:
                return
            seen.add(current_id)
            for attr_name in (
                _SPECIALIZATION_ANALYZED_ATTR,
                _TEMPLATE_PREPARED_ATTR,
            ):
                if hasattr(current, attr_name):
                    delattr(current, attr_name)
            for value in vars(current).values():
                if isinstance(value, astx.AST):
                    clear(value)
                    continue
                if isinstance(value, list | tuple):
                    for item in value:
                        if isinstance(item, astx.AST):
                            clear(item)

        clear(module)

    def _substitute_declared_types(
        self,
        node: astx.AST,
        bindings: dict[str, astx.DataType],
    ) -> None:
        """
        title: >-
          Substitute template types throughout one cloned function subtree.
        parameters:
          node:
            type: astx.AST
          bindings:
            type: dict[str, astx.DataType]
        """
        if isinstance(node, astx.FunctionDef):
            self._substitute_declared_types(node.prototype, bindings)
            self._substitute_declared_types(node.body, bindings)
            explicit_args = astx.get_template_args(node)
            if explicit_args is not None:
                astx.set_template_args(
                    node,
                    tuple(
                        self._substitute_type(arg, bindings)
                        for arg in explicit_args
                    ),
                )
            return
        if isinstance(node, astx.FunctionPrototype):
            for argument in node.args.nodes:
                self._substitute_declared_types(argument, bindings)
            node.return_type = self._substitute_type(
                node.return_type,
                bindings,
            )
            explicit_args = astx.get_template_args(node)
            if explicit_args is not None:
                astx.set_template_args(
                    node,
                    tuple(
                        self._substitute_type(arg, bindings)
                        for arg in explicit_args
                    ),
                )
            return
        if isinstance(node, astx.Argument):
            node.type_ = self._substitute_type(node.type_, bindings)
            return
        if isinstance(node, astx.VariableDeclaration):
            node.type_ = self._substitute_type(node.type_, bindings)
            if node.value is not None:
                self._substitute_declared_types(node.value, bindings)
            return
        if isinstance(node, astx.Cast):
            node.target_type = self._substitute_type(
                node.target_type,
                bindings,
            )
            self._substitute_declared_types(node.value, bindings)
            return

        explicit_args = astx.get_template_args(node)
        if explicit_args is not None:
            astx.set_template_args(
                node,
                tuple(
                    self._substitute_type(arg, bindings)
                    for arg in explicit_args
                ),
            )

        for name, value in vars(node).items():
            if name in {
                "semantic",
                "parent",
                "type_",
                "return_type",
                "target_type",
            }:
                continue
            if isinstance(value, astx.AST):
                self._substitute_declared_types(value, bindings)
                continue
            if isinstance(value, list | tuple):
                for item in value:
                    if isinstance(item, astx.AST):
                        self._substitute_declared_types(item, bindings)

    def _function_owner_module(
        self,
        function: SemanticFunction,
    ) -> astx.Module | None:
        """
        title: Return the owning AST module for one semantic function.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: astx.Module | None
        """
        if (
            self.session is not None
            and function.module_key in self.session.modules
        ):
            return self.session.module(function.module_key).ast
        current_module = getattr(self, "_current_ast_module", None)
        if isinstance(current_module, astx.Module):
            return current_module
        owner = getattr(function.prototype, _OWNER_MODULE_ATTR, None)
        if isinstance(owner, astx.Module):
            return owner
        owner = getattr(function.definition, _OWNER_MODULE_ATTR, None)
        if isinstance(owner, astx.Module):
            return owner
        return None

    def _template_specializations_prepared(
        self,
        function: SemanticFunction,
    ) -> bool:
        """
        title: >-
          Return whether one template function already prepared its clones.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: bool
        """
        carrier = function.definition or function.prototype
        return bool(getattr(carrier, _TEMPLATE_PREPARED_ATTR, False))

    def _mark_template_specializations_prepared(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Mark one template function as having prepared its clones.
        parameters:
          function:
            type: SemanticFunction
        """
        carrier = function.definition or function.prototype
        setattr(carrier, _TEMPLATE_PREPARED_ATTR, True)

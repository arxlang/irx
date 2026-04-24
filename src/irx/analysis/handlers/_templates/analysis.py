"""
title: Template-specialization analysis helpers.
summary: >-
  Prepare generated specializations and analyze their concrete function bodies
  during template processing.
"""

from __future__ import annotations

from itertools import product

from irx import astx
from irx.analysis.handlers._templates.resolution import (
    TemplateResolutionVisitorMixin,
)
from irx.analysis.handlers._templates.state import (
    _OWNER_MODULE_ATTR,
    _SPECIALIZATION_ANALYZED_ATTR,
)
from irx.analysis.resolved_nodes import SemanticFunction
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class TemplateAnalysisVisitorMixin(TemplateResolutionVisitorMixin):
    """
    title: Template-specialization preparation and semantic-analysis helpers
    """

    def _prepare_function_template_specializations(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Materialize all concrete specializations for one template func.
        parameters:
          function:
            type: SemanticFunction
        """
        if (
            not function.template_params
            or function.template_definition is not None
        ):
            return
        if function.definition is None:
            self.context.diagnostics.add(
                f"Template function '{function.name}' must have a definition",
                node=function.prototype,
                code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
            )
            return
        if self._template_specializations_prepared(function):
            return
        domains = tuple(
            self._template_param_domain(param)
            for param in function.template_params
        )
        for concrete_args in product(*domains):
            self._build_template_specialization(function, concrete_args)
        self._mark_template_specializations_prepared(function)

    def _prepare_template_specialization_skeletons(
        self,
        module: astx.Module,
    ) -> None:
        """
        title: Materialize specialization skeletons for module templates.
        parameters:
          module:
            type: astx.Module
        """
        for node in module.nodes:
            if isinstance(node, astx.FunctionPrototype):
                setattr(node, _OWNER_MODULE_ATTR, module)
                function = self.context.get_function(
                    self._current_module_key(),
                    node.name,
                )
                if (
                    function is not None
                    and function.template_params
                    and function.definition is None
                ):
                    self.context.diagnostics.add(
                        f"Template function '{function.name}' must have a "
                        "definition",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                continue
            if not isinstance(node, astx.FunctionDef):
                continue
            setattr(node, _OWNER_MODULE_ATTR, module)
            setattr(node.prototype, _OWNER_MODULE_ATTR, module)
            function = self.context.get_function(
                self._current_module_key(),
                node.name,
            )
            if function is None:
                continue
            self._prepare_function_template_specializations(function)

    def _analyze_specialized_function_body(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Analyze one generated concrete specialization body.
        parameters:
          function:
            type: SemanticFunction
        """
        definition = function.definition
        if definition is None:
            raise TypeError("template specialization requires a definition")
        for argument in definition.prototype.args.nodes:
            self._resolve_declared_type(argument.type_, node=argument)
        self._resolve_declared_type(
            definition.prototype.return_type,
            node=definition,
        )
        self._set_function(definition.prototype, function)
        self._set_function(definition, function)
        self._set_type(definition.prototype, None)
        self._set_type(definition, None)
        hidden_parameter_count = len(function.args) - len(
            definition.prototype.args.nodes
        )
        with self.context.in_function(function):
            with self.context.scope("function"):
                for index, arg_symbol in enumerate(function.args):
                    self.context.scopes.declare(arg_symbol)
                    if index < hidden_parameter_count:
                        continue
                    arg_node = definition.prototype.args.nodes[
                        index - hidden_parameter_count
                    ]
                    self._set_symbol(arg_node, arg_symbol)
                    self._set_type(arg_node, arg_symbol.type_)
                self.visit(definition.body)
        if not isinstance(
            function.return_type, astx.NoneType
        ) and not self._guarantees_return(definition.body):
            self.context.diagnostics.add(
                f"Function '{function.name}' with return type "
                f"'{function.return_type}' is missing a return statement",
                node=definition,
            )

    def _analyze_prepared_template_specialization(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Analyze one prepared concrete specialization once.
        parameters:
          function:
            type: SemanticFunction
        """
        definition = function.definition
        if definition is None:
            raise TypeError(
                "prepared template specialization lacks definition"
            )
        if getattr(definition, _SPECIALIZATION_ANALYZED_ATTR, False):
            return
        diagnostic_count_before = len(self.context.diagnostics.diagnostics)
        self._analyze_specialized_function_body(function)
        setattr(definition, _SPECIALIZATION_ANALYZED_ATTR, True)
        if (
            len(self.context.diagnostics.diagnostics)
            == diagnostic_count_before
        ):
            return
        template_definition = function.template_definition
        if template_definition is None:
            return
        bindings_text = self._format_template_bindings(
            function.template_bindings
        )
        self.context.diagnostics.add(
            f"Template function '{template_definition.name}' is invalid for "
            f"{bindings_text}",
            node=(
                template_definition.definition or template_definition.prototype
            ),
            code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
        )

    def _analyze_function_template_specializations(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Analyze all prepared specializations for one template function.
        parameters:
          function:
            type: SemanticFunction
        """
        for specialization in tuple(function.specializations.values()):
            self._analyze_prepared_template_specialization(specialization)

    def _analyze_prepared_template_specializations(
        self,
        module: astx.Module,
    ) -> None:
        """
        title: Analyze the generated specializations attached to one module.
        parameters:
          module:
            type: astx.Module
        """
        for node in astx.generated_template_nodes(module):
            semantic = getattr(node, "semantic", None)
            function = getattr(semantic, "resolved_function", None)
            if not isinstance(function, SemanticFunction):
                continue
            self._analyze_prepared_template_specialization(function)

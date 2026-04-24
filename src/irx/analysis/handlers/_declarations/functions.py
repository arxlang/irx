# mypy: disable-error-code=no-redef
# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=untyped-decorator

"""
title: Declaration function visitors.
summary: >-
  Resolve function declarations, synchronize normalized signatures, and analyze
  non-template function bodies.
"""

from __future__ import annotations

from dataclasses import replace

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.module_symbols import qualified_local_name
from irx.analysis.resolved_nodes import SemanticFunction
from irx.analysis.types import clone_type
from irx.typecheck import typechecked


@typechecked
class DeclarationFunctionVisitorMixin(SemanticVisitorMixinBase):
    """
    title: Declaration visitors for function signatures and bodies
    """

    def _synchronize_function_signature(
        self,
        function: SemanticFunction,
        prototype: astx.FunctionPrototype,
        *,
        definition: astx.FunctionDef | None = None,
    ) -> SemanticFunction:
        """
        title: Synchronize one semantic function with resolved AST types.
        parameters:
          function:
            type: SemanticFunction
          prototype:
            type: astx.FunctionPrototype
          definition:
            type: astx.FunctionDef | None
        returns:
          type: SemanticFunction
        """
        signature = self.registry.normalize_function_signature(
            prototype,
            definition=definition,
        )
        if (
            function.prototype is not prototype
            and not self.registry.signatures_match(
                function.signature, signature
            )
        ):
            return function
        if (
            definition is not None
            and function.definition is not None
            and function.definition is not definition
            and not self.registry.signatures_match(
                function.signature, signature
            )
        ):
            return function

        updated = replace(
            function,
            return_type=clone_type(signature.return_type),
            args=tuple(
                replace(
                    arg_symbol,
                    name=arg_node.name,
                    type_=clone_type(arg_node.type_),
                    qualified_name=qualified_local_name(
                        function.module_key,
                        arg_symbol.kind,
                        arg_node.name,
                        arg_symbol.symbol_id,
                    ),
                )
                for arg_node, arg_symbol in zip(
                    prototype.args.nodes,
                    function.args,
                )
            ),
            signature=signature,
            prototype=prototype,
            definition=(
                definition if definition is not None else function.definition
            ),
            template_params=tuple(
                astx.TemplateParam(
                    param.name,
                    clone_type(param.bound),
                    param.loc,
                )
                for param in astx.get_template_params(prototype)
            ),
        )
        self.context.register_function(updated)
        return updated

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionPrototype) -> None:
        """
        title: Visit FunctionPrototype nodes.
        parameters:
          node:
            type: astx.FunctionPrototype
        """
        for template_param in astx.get_template_params(node):
            self._resolve_declared_type(template_param.bound, node=node)
        for arg in node.args.nodes:
            self._resolve_declared_type(arg.type_, node=arg)
        self._resolve_declared_type(node.return_type, node=node)
        function = self.registry.resolve_function(node.name)
        if function is None:
            function = self.registry.register_function(node)
        function = self._synchronize_function_signature(function, node)
        self.bindings.bind_function(node.name, function, node=node)
        self._set_function(node, function)
        if function.template_params:
            self._set_type(node, None)
            return
        with self.context.in_function(function):
            self._analyze_parameter_defaults(function)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionDef) -> None:
        """
        title: Visit FunctionDef nodes.
        parameters:
          node:
            type: astx.FunctionDef
        """
        for template_param in astx.get_template_params(node.prototype):
            self._resolve_declared_type(template_param.bound, node=node)
        for arg in node.prototype.args.nodes:
            self._resolve_declared_type(arg.type_, node=arg)
        self._resolve_declared_type(node.prototype.return_type, node=node)
        function = self.registry.resolve_function(node.name)
        if function is None:
            function = self.registry.register_function(
                node.prototype,
                definition=node,
            )
        function = self._synchronize_function_signature(
            function,
            node.prototype,
            definition=node,
        )
        self.bindings.bind_function(node.name, function, node=node)
        self._set_function(node.prototype, function)
        self._set_function(node, function)
        if function.template_params:
            self._set_type(node.prototype, None)
            self._set_type(node, None)
            return
        with self.context.in_function(function):
            self._analyze_parameter_defaults(function)
            with self.context.scope("function"):
                for arg_node, arg_symbol in zip(
                    node.prototype.args.nodes,
                    function.args,
                ):
                    self.context.scopes.declare(arg_symbol)
                    self._set_symbol(arg_node, arg_symbol)
                    self._set_type(arg_node, arg_symbol.type_)
                self.visit(node.body)
        if not isinstance(
            function.return_type, astx.NoneType
        ) and not self._guarantees_return(node.body):
            self.context.diagnostics.add(
                f"Function '{node.name}' with return type "
                f"'{function.return_type}' is missing a return statement",
                node=node,
            )

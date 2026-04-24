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
from irx.analysis.resolved_nodes import (
    ResolvedGeneratorFunction,
    SemanticFunction,
)
from irx.analysis.types import clone_type
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
def _is_yield_node(node: astx.AST) -> bool:
    """
    title: Return whether one node is a yield form.
    parameters:
      node:
        type: astx.AST
    returns:
      type: bool
    """
    return isinstance(
        node,
        (astx.YieldExpr, astx.YieldFromExpr, astx.YieldStmt),
    )


@typechecked
def _nested_yield_nodes(node: astx.AST) -> tuple[astx.AST, ...]:
    """
    title: Return yield nodes nested below one non-function statement.
    parameters:
      node:
        type: astx.AST
    returns:
      type: tuple[astx.AST, Ellipsis]
    """
    if _is_yield_node(node):
        return (node,)
    if isinstance(node, astx.FunctionDef):
        return ()
    if isinstance(node, astx.Block):
        return tuple(
            yield_node
            for child in node.nodes
            for yield_node in _nested_yield_nodes(child)
        )

    child_blocks: list[astx.Block] = []
    for attribute_name in ("body", "then", "else_"):
        child = getattr(node, attribute_name, None)
        if isinstance(child, astx.Block):
            child_blocks.append(child)

    return tuple(
        yield_node
        for child in child_blocks
        for yield_node in _nested_yield_nodes(child)
    )


@typechecked
def _top_level_yield_nodes(block: astx.Block) -> tuple[astx.AST, ...]:
    """
    title: Return yield nodes that are direct statements in a block.
    parameters:
      block:
        type: astx.Block
    returns:
      type: tuple[astx.AST, Ellipsis]
    """
    return tuple(node for node in block.nodes if _is_yield_node(node))


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

    def _resolve_generator_function(
        self,
        function: SemanticFunction,
        node: astx.FunctionDef,
    ) -> ResolvedGeneratorFunction | None:
        """
        title: Resolve generator metadata for one function definition.
        parameters:
          function:
            type: SemanticFunction
          node:
            type: astx.FunctionDef
        returns:
          type: ResolvedGeneratorFunction | None
        """
        yield_nodes = _top_level_yield_nodes(node.body)
        nested_yields = tuple(
            yield_node
            for child in node.body.nodes
            if not _is_yield_node(child)
            for yield_node in _nested_yield_nodes(child)
        )
        if not yield_nodes and not nested_yields:
            return None

        if not isinstance(function.signature.return_type, astx.GeneratorType):
            self.context.diagnostics.add(
                f"Function '{node.name}' contains yield but must declare "
                "GeneratorType[...] as its return type",
                node=node.prototype,
                code=DiagnosticCodes.SEMANTIC_INVALID_RETURN,
            )
            return None

        for yield_node in nested_yields:
            self.context.diagnostics.add(
                "yield inside nested control flow is not supported yet",
                node=yield_node,
                code=DiagnosticCodes.SEMANTIC_INVALID_CONTROL_FLOW,
            )

        generator = ResolvedGeneratorFunction(
            function=function,
            yield_type=clone_type(function.signature.return_type.yield_type),
            yield_nodes=yield_nodes,
        )
        function.signature.metadata["generator"] = generator
        self._set_generator_function(node.prototype, generator)
        self._set_generator_function(node, generator)
        return generator

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
        generator = self._resolve_generator_function(function, node)
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
        if (
            not isinstance(function.return_type, astx.NoneType)
            and generator is None
            and not self._guarantees_return(node.body)
        ):
            self.context.diagnostics.add(
                f"Function '{node.name}' with return type "
                f"'{function.return_type}' is missing a return statement",
                node=node,
            )

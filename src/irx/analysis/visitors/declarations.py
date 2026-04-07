# mypy: disable-error-code=no-redef

"""
title: Declaration-oriented semantic visitors.
summary: >-
  Handle modules, functions, structs, and lexical declarations while delegating
  semantic entity creation and registration to shared infrastructure.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.validation import validate_assignment
from irx.analysis.visitors.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)


class DeclarationVisitorMixin(SemanticVisitorMixinBase):
    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, module: astx.Module) -> None:
        """
        title: Visit Module nodes.
        parameters:
          module:
            type: astx.Module
        """
        with self.context.in_module(module.name):
            self._visit_module(module, predeclared=False)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, block: astx.Block) -> None:
        """
        title: Visit Block nodes.
        parameters:
          block:
            type: astx.Block
        """
        self._set_type(block, None)
        for node in block.nodes:
            self.visit(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionPrototype) -> None:
        """
        title: Visit FunctionPrototype nodes.
        parameters:
          node:
            type: astx.FunctionPrototype
        """
        function = self.registry.resolve_function(node.name)
        if function is None:
            function = self.registry.register_function(node)
        self.bindings.bind_function(node.name, function, node=node)
        self._set_function(node, function)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionDef) -> None:
        """
        title: Visit FunctionDef nodes.
        parameters:
          node:
            type: astx.FunctionDef
        """
        function = self.registry.resolve_function(node.name)
        if function is None:
            function = self.registry.register_function(
                node.prototype,
                definition=node,
            )
        self.bindings.bind_function(node.name, function, node=node)
        self._set_function(node.prototype, function)
        self._set_function(node, function)
        with self.context.in_function(function):
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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.VariableDeclaration) -> None:
        """
        title: Visit VariableDeclaration nodes.
        parameters:
          node:
            type: astx.VariableDeclaration
        """
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self.visit(node.value)
            validate_assignment(
                self.context.diagnostics,
                target_name=node.name,
                target_type=node.type_,
                value_type=self._expr_type(node.value),
                node=node,
            )
        symbol = self.registry.declare_local(
            node.name,
            node.type_,
            is_mutable=node.mutability != astx.MutabilityKind.constant,
            declaration=node,
        )
        self._set_symbol(node, symbol)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.InlineVariableDeclaration) -> None:
        """
        title: Visit InlineVariableDeclaration nodes.
        parameters:
          node:
            type: astx.InlineVariableDeclaration
        """
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self.visit(node.value)
            validate_assignment(
                self.context.diagnostics,
                target_name=node.name,
                target_type=node.type_,
                value_type=self._expr_type(node.value),
                node=node,
            )
        symbol = self.registry.declare_local(
            node.name,
            node.type_,
            is_mutable=node.mutability != astx.MutabilityKind.constant,
            declaration=node,
        )
        self._set_symbol(node, symbol)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.StructDefStmt) -> None:
        """
        title: Visit StructDefStmt nodes.
        parameters:
          node:
            type: astx.StructDefStmt
        """
        struct = self.registry.register_struct(node)
        self.bindings.bind_struct(node.name, struct, node=node)
        self._set_struct(node, struct)
        seen: set[str] = set()
        for attr in node.attributes:
            if attr.name in seen:
                self.context.diagnostics.add(
                    f"Struct field '{attr.name}' already defined.",
                    node=attr,
                )
            seen.add(attr.name)
        self._set_type(node, None)

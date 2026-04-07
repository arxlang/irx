"""
title: Shared semantic-analyzer core.
summary: >-
  Hold analyzer state, semantic sidecar helpers, and shared traversal utilities
  that specialized visitor mixins build on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from plum import dispatch

from irx import astx
from irx.analysis.bindings import VisibleBindings
from irx.analysis.context import SemanticContext
from irx.analysis.factories import SemanticEntityFactory
from irx.analysis.module_interfaces import ModuleKey, ParsedModule
from irx.analysis.registry import SemanticRegistry
from irx.analysis.resolved_nodes import (
    ResolvedAssignment,
    ResolvedImportBinding,
    ResolvedOperator,
    SemanticFlags,
    SemanticFunction,
    SemanticInfo,
    SemanticModule,
    SemanticStruct,
    SemanticSymbol,
)
from irx.analysis.session import CompilationSession
from irx.analysis.types import clone_type
from irx.base.visitors.base import BaseVisitor

if TYPE_CHECKING:

    class SemanticVisitorMixinBase:
        """
        title: Type-checking-only base for semantic visitor mixins.
        attributes:
          context:
            type: SemanticContext
          session:
            type: CompilationSession | None
          factory:
            type: SemanticEntityFactory
          registry:
            type: SemanticRegistry
          bindings:
            type: VisibleBindings
        """

        context: SemanticContext
        session: CompilationSession | None
        factory: SemanticEntityFactory
        registry: SemanticRegistry
        bindings: VisibleBindings

        def visit(self, node: astx.AST) -> None:
            """
            title: Visit AST nodes.
            parameters:
              node:
                type: astx.AST
            """
            raise NotImplementedError

        def _semantic(self, node: astx.AST) -> SemanticInfo:
            """
            title: Return one node's semantic sidecar.
            parameters:
              node:
                type: astx.AST
            returns:
              type: SemanticInfo
            """
            raise NotImplementedError

        def _set_type(
            self,
            node: astx.AST,
            type_: astx.DataType | None,
        ) -> astx.DataType | None:
            """
            title: Attach one resolved type to a node.
            parameters:
              node:
                type: astx.AST
              type_:
                type: astx.DataType | None
            returns:
              type: astx.DataType | None
            """
            raise NotImplementedError

        def _set_symbol(
            self,
            node: astx.AST,
            symbol: SemanticSymbol | None,
        ) -> SemanticSymbol | None:
            """
            title: Attach one resolved symbol to a node.
            parameters:
              node:
                type: astx.AST
              symbol:
                type: SemanticSymbol | None
            returns:
              type: SemanticSymbol | None
            """
            raise NotImplementedError

        def _set_function(
            self,
            node: astx.AST,
            function: SemanticFunction | None,
        ) -> SemanticFunction | None:
            """
            title: Attach one resolved function to a node.
            parameters:
              node:
                type: astx.AST
              function:
                type: SemanticFunction | None
            returns:
              type: SemanticFunction | None
            """
            raise NotImplementedError

        def _set_struct(
            self,
            node: astx.AST,
            struct: SemanticStruct | None,
        ) -> SemanticStruct | None:
            """
            title: Attach one resolved struct to a node.
            parameters:
              node:
                type: astx.AST
              struct:
                type: SemanticStruct | None
            returns:
              type: SemanticStruct | None
            """
            raise NotImplementedError

        def _set_module(
            self,
            node: astx.AST,
            module: SemanticModule | None,
        ) -> SemanticModule | None:
            """
            title: Attach one resolved module to a node.
            parameters:
              node:
                type: astx.AST
              module:
                type: SemanticModule | None
            returns:
              type: SemanticModule | None
            """
            raise NotImplementedError

        def _set_imports(
            self,
            node: astx.AST,
            imports: tuple[ResolvedImportBinding, ...],
        ) -> None:
            """
            title: Attach resolved imports to a node.
            parameters:
              node:
                type: astx.AST
              imports:
                type: tuple[ResolvedImportBinding, Ellipsis]
            """
            raise NotImplementedError

        def _set_flags(self, node: astx.AST, flags: SemanticFlags) -> None:
            """
            title: Attach semantic flags to a node.
            parameters:
              node:
                type: astx.AST
              flags:
                type: SemanticFlags
            """
            raise NotImplementedError

        def _set_operator(
            self,
            node: astx.AST,
            operator: ResolvedOperator | None,
        ) -> None:
            """
            title: Attach one resolved operator to a node.
            parameters:
              node:
                type: astx.AST
              operator:
                type: ResolvedOperator | None
            """
            raise NotImplementedError

        def _set_assignment(
            self,
            node: astx.AST,
            symbol: SemanticSymbol | None,
        ) -> None:
            """
            title: Attach one resolved assignment target to a node.
            parameters:
              node:
                type: astx.AST
              symbol:
                type: SemanticSymbol | None
            """
            raise NotImplementedError

        def _current_module_key(self) -> ModuleKey:
            """
            title: Return the current module key.
            returns:
              type: ModuleKey
            """
            raise NotImplementedError

        def _imports_supported_here(self, node: astx.AST) -> bool:
            """
            title: Return True when imports are allowed here.
            parameters:
              node:
                type: astx.AST
            returns:
              type: bool
            """
            raise NotImplementedError

        def _expr_type(self, node: astx.AST | None) -> astx.DataType | None:
            """
            title: Return one node's resolved expression type.
            parameters:
              node:
                type: astx.AST | None
            returns:
              type: astx.DataType | None
            """
            raise NotImplementedError

        def _visit_module(
            self,
            module: astx.Module,
            *,
            predeclared: bool,
        ) -> None:
            """
            title: Visit one module with optional predeclaration.
            parameters:
              module:
                type: astx.Module
              predeclared:
                type: bool
            """
            raise NotImplementedError

        def _guarantees_return(self, node: astx.AST) -> bool:
            """
            title: Return whether a statement subtree guarantees return.
            parameters:
              node:
                type: astx.AST
            returns:
              type: bool
            """
            raise NotImplementedError

else:

    class SemanticVisitorMixinBase:
        """
        title: Runtime-empty base for semantic visitor mixins.
        """


class SemanticAnalyzerCore(BaseVisitor):
    """
    title: Shared semantic analyzer core.
    summary: >-
      Provide shared semantic-analysis state and helper behavior for the
      specialized visitor mixins that implement node-specific rules.
    attributes:
      context:
        type: SemanticContext
      session:
        type: CompilationSession | None
      factory:
        type: SemanticEntityFactory
      registry:
        type: SemanticRegistry
      bindings:
        type: VisibleBindings
    """

    context: SemanticContext
    session: CompilationSession | None
    factory: SemanticEntityFactory
    registry: SemanticRegistry
    bindings: VisibleBindings

    def __init__(
        self,
        *,
        context: SemanticContext | None = None,
        session: CompilationSession | None = None,
    ) -> None:
        """
        title: Initialize SemanticAnalyzerCore.
        parameters:
          context:
            type: SemanticContext | None
          session:
            type: CompilationSession | None
        """
        self.session = session
        self.context = context or SemanticContext()
        if session is not None:
            self.context.diagnostics = session.diagnostics
        self.factory = SemanticEntityFactory(self.context)
        self.registry = SemanticRegistry(self.context, self.factory)
        self.bindings = VisibleBindings(
            context=self.context,
            factory=self.factory,
            bindings=(
                session.visible_bindings if session is not None else None
            ),
        )

    def analyze(self, node: astx.AST) -> astx.AST:
        """
        title: Analyze one AST root.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.AST
        """
        if isinstance(node, astx.Module):
            parsed_module = ParsedModule(node.name, node)
            self.analyze_parsed_module(parsed_module, predeclared=False)
        else:
            with self.context.scope("module"):
                self.visit(node)
        self.context.diagnostics.raise_if_errors()
        return node

    def analyze_parsed_module(
        self,
        parsed_module: ParsedModule,
        *,
        predeclared: bool,
    ) -> astx.Module:
        """
        title: Analyze one parsed module.
        parameters:
          parsed_module:
            type: ParsedModule
          predeclared:
            type: bool
        returns:
          type: astx.Module
        """
        with self.context.in_module(parsed_module.key):
            self._visit_module(parsed_module.ast, predeclared=predeclared)
        return parsed_module.ast

    def _semantic(self, node: astx.AST) -> SemanticInfo:
        """
        title: Return one node's semantic sidecar.
        parameters:
          node:
            type: astx.AST
        returns:
          type: SemanticInfo
        """
        info = cast(SemanticInfo | None, getattr(node, "semantic", None))
        if info is None or not isinstance(info, SemanticInfo):
            info = SemanticInfo()
            setattr(node, "semantic", info)
        return info

    def _set_type(
        self, node: astx.AST, type_: astx.DataType | None
    ) -> astx.DataType | None:
        """
        title: Attach one resolved type to a node.
        parameters:
          node:
            type: astx.AST
          type_:
            type: astx.DataType | None
        returns:
          type: astx.DataType | None
        """
        info = self._semantic(node)
        info.resolved_type = type_
        if type_ is not None and hasattr(node, "type_"):
            try:
                setattr(node, "type_", clone_type(type_))
            except Exception:
                pass
        return type_

    def _set_symbol(
        self, node: astx.AST, symbol: SemanticSymbol | None
    ) -> SemanticSymbol | None:
        """
        title: Attach one resolved symbol to a node.
        parameters:
          node:
            type: astx.AST
          symbol:
            type: SemanticSymbol | None
        returns:
          type: SemanticSymbol | None
        """
        info = self._semantic(node)
        info.resolved_symbol = symbol
        if symbol is not None:
            self._set_type(node, symbol.type_)
        return symbol

    def _set_function(
        self, node: astx.AST, function: SemanticFunction | None
    ) -> SemanticFunction | None:
        """
        title: Attach one resolved function to a node.
        parameters:
          node:
            type: astx.AST
          function:
            type: SemanticFunction | None
        returns:
          type: SemanticFunction | None
        """
        info = self._semantic(node)
        info.resolved_function = function
        if function is not None:
            self._set_type(node, function.return_type)
        return function

    def _set_struct(
        self,
        node: astx.AST,
        struct: SemanticStruct | None,
    ) -> SemanticStruct | None:
        """
        title: Attach one resolved struct to a node.
        parameters:
          node:
            type: astx.AST
          struct:
            type: SemanticStruct | None
        returns:
          type: SemanticStruct | None
        """
        info = self._semantic(node)
        info.resolved_struct = struct
        return struct

    def _set_module(
        self,
        node: astx.AST,
        module: SemanticModule | None,
    ) -> SemanticModule | None:
        """
        title: Attach one resolved module to a node.
        parameters:
          node:
            type: astx.AST
          module:
            type: SemanticModule | None
        returns:
          type: SemanticModule | None
        """
        info = self._semantic(node)
        info.resolved_module = module
        return module

    def _set_imports(
        self,
        node: astx.AST,
        imports: tuple[ResolvedImportBinding, ...],
    ) -> None:
        """
        title: Attach resolved imports to a node.
        parameters:
          node:
            type: astx.AST
          imports:
            type: tuple[ResolvedImportBinding, Ellipsis]
        """
        self._semantic(node).resolved_imports = imports

    def _set_flags(self, node: astx.AST, flags: SemanticFlags) -> None:
        """
        title: Attach semantic flags to a node.
        parameters:
          node:
            type: astx.AST
          flags:
            type: SemanticFlags
        """
        self._semantic(node).semantic_flags = flags

    def _set_operator(
        self,
        node: astx.AST,
        operator: ResolvedOperator | None,
    ) -> None:
        """
        title: Attach one resolved operator to a node.
        parameters:
          node:
            type: astx.AST
          operator:
            type: ResolvedOperator | None
        """
        self._semantic(node).resolved_operator = operator

    def _set_assignment(
        self, node: astx.AST, symbol: SemanticSymbol | None
    ) -> None:
        """
        title: Attach one resolved assignment target to a node.
        parameters:
          node:
            type: astx.AST
          symbol:
            type: SemanticSymbol | None
        """
        info = self._semantic(node)
        if symbol is None:
            info.resolved_assignment = None
            return
        info.resolved_assignment = ResolvedAssignment(symbol)

    def _current_module_key(self) -> ModuleKey:
        """
        title: Return the current module key.
        returns:
          type: ModuleKey
        """
        return self.context.current_module_key or "<root>"

    def _imports_supported_here(self, node: astx.AST) -> bool:
        """
        title: Return True when imports are allowed here.
        parameters:
          node:
            type: astx.AST
        returns:
          type: bool
        """
        if self.session is None:
            self.context.diagnostics.add(
                "Import statements require analyze_modules(...) with a "
                "host-provided resolver.",
                node=node,
            )
            return False
        current_scope = self.context.scopes.current
        if current_scope is None or current_scope.kind != "module":
            self.context.diagnostics.add(
                "Import statements are supported only at module top level "
                "in this MVP.",
                node=node,
            )
            return False
        return True

    def _expr_type(self, node: astx.AST | None) -> astx.DataType | None:
        """
        title: Return one node's resolved expression type.
        parameters:
          node:
            type: astx.AST | None
        returns:
          type: astx.DataType | None
        """
        if node is None:
            return None
        info = cast(SemanticInfo | None, getattr(node, "semantic", None))
        if info is not None and info.resolved_type is not None:
            return info.resolved_type
        return getattr(node, "type_", None)

    def _predeclare_module_members(self, module: astx.Module) -> None:
        """
        title: Predeclare one module's top-level members.
        parameters:
          module:
            type: astx.Module
        """
        for node in module.nodes:
            if isinstance(node, astx.FunctionPrototype):
                function = self.registry.register_function(node)
                self.bindings.bind_function(node.name, function, node=node)
                self._set_function(node, function)
            elif isinstance(node, astx.FunctionDef):
                function = self.registry.register_function(
                    node.prototype,
                    definition=node,
                )
                self.bindings.bind_function(
                    node.prototype.name,
                    function,
                    node=node,
                )
                self._set_function(node.prototype, function)
                self._set_function(node, function)
            elif isinstance(node, astx.StructDefStmt):
                struct = self.registry.register_struct(node)
                self.bindings.bind_struct(node.name, struct, node=node)
                self._set_struct(node, struct)

    def _visit_module(
        self,
        module: astx.Module,
        *,
        predeclared: bool,
    ) -> None:
        """
        title: Visit one module with optional predeclaration.
        parameters:
          module:
            type: astx.Module
          predeclared:
            type: bool
        """
        self._set_type(module, None)
        with self.context.scope("module"):
            if not predeclared:
                self._predeclare_module_members(module)
            for node in module.nodes:
                self.visit(node)

    def _visit_plain_typed_node(self, node: astx.AST) -> None:
        """
        title: Visit one plain typed node.
        parameters:
          node:
            type: astx.AST
        """
        self._set_type(node, getattr(node, "type_", None))

    def _guarantees_return(self, node: astx.AST) -> bool:
        """
        title: Return whether a statement subtree guarantees return.
        parameters:
          node:
            type: astx.AST
        returns:
          type: bool
        """
        if isinstance(node, astx.FunctionReturn):
            return True
        if isinstance(node, astx.Block):
            for child in node.nodes:
                if self._guarantees_return(child):
                    return True
            return False
        if isinstance(node, astx.IfStmt):
            if node.else_ is None:
                return False
            return self._guarantees_return(
                node.then
            ) and self._guarantees_return(node.else_)
        return False

    @dispatch
    def visit(self, node: astx.AST) -> None:
        """
        title: Visit generic AST nodes.
        parameters:
          node:
            type: astx.AST
        """
        self._visit_plain_typed_node(node)

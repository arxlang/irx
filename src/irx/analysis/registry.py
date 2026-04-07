"""
title: Semantic registration helpers.
summary: >-
  Centralize semantic declaration and top-level registration policy so the
  analyzer delegates identity and duplicate handling to a smaller subsystem.
"""

from __future__ import annotations

from dataclasses import replace

from public import public

from irx import astx
from irx.analysis.context import SemanticContext
from irx.analysis.factories import SemanticEntityFactory
from irx.analysis.module_interfaces import ModuleKey
from irx.analysis.resolved_nodes import (
    SemanticFunction,
    SemanticStruct,
    SemanticSymbol,
)


@public
class SemanticRegistry:
    """
    title: Semantic entity registration policy.
    summary: >-
      Register locals, functions, and structs while enforcing the duplicate
      declaration rules that semantic analysis currently exposes.
    attributes:
      context:
        type: SemanticContext
      factory:
        type: SemanticEntityFactory
    """

    context: SemanticContext
    factory: SemanticEntityFactory

    def __init__(
        self,
        context: SemanticContext,
        factory: SemanticEntityFactory,
    ) -> None:
        """
        title: Initialize SemanticRegistry.
        parameters:
          context:
            type: SemanticContext
          factory:
            type: SemanticEntityFactory
        """
        self.context = context
        self.factory = factory

    def _current_module_key(self) -> ModuleKey:
        """
        title: Return the active module key.
        returns:
          type: ModuleKey
        """
        return self.context.current_module_key or "<root>"

    def declare_local(
        self,
        name: str,
        type_: astx.DataType,
        *,
        is_mutable: bool,
        declaration: astx.AST,
        kind: str = "variable",
    ) -> SemanticSymbol:
        """
        title: Declare one lexical symbol.
        parameters:
          name:
            type: str
          type_:
            type: astx.DataType
          is_mutable:
            type: bool
          declaration:
            type: astx.AST
          kind:
            type: str
        returns:
          type: SemanticSymbol
        """
        symbol = self.factory.make_variable_symbol(
            self._current_module_key(),
            name,
            type_,
            is_mutable=is_mutable,
            declaration=declaration,
            kind=kind,
        )
        if not self.context.scopes.declare(symbol):
            self.context.diagnostics.add(
                f"Identifier already declared: {name}",
                node=declaration,
            )
        return symbol

    def register_function(
        self,
        prototype: astx.FunctionPrototype,
        *,
        definition: astx.FunctionDef | None = None,
    ) -> SemanticFunction:
        """
        title: Register one top-level function.
        parameters:
          prototype:
            type: astx.FunctionPrototype
          definition:
            type: astx.FunctionDef | None
        returns:
          type: SemanticFunction
        """
        module_key = self._current_module_key()
        existing = self.context.get_function(module_key, prototype.name)
        if existing is not None:
            if definition is not None and existing.definition is not None:
                self.context.diagnostics.add(
                    f"Function '{prototype.name}' already defined",
                    node=definition,
                )
            if definition is not None and existing.definition is None:
                updated = replace(existing, definition=definition)
                self.context.register_function(updated)
                return updated
            return existing

        function = self.factory.make_function(
            module_key,
            prototype,
            definition=definition,
        )
        self.context.register_function(function)
        return function

    def register_struct(
        self,
        node: astx.StructDefStmt,
    ) -> SemanticStruct:
        """
        title: Register one top-level struct.
        parameters:
          node:
            type: astx.StructDefStmt
        returns:
          type: SemanticStruct
        """
        module_key = self._current_module_key()
        existing = self.context.get_struct(module_key, node.name)
        if existing is not None:
            if existing.declaration is not node:
                self.context.diagnostics.add(
                    f"Struct '{node.name}' already defined.",
                    node=node,
                )
            return existing

        struct = self.factory.make_struct(module_key, node)
        self.context.register_struct(struct)
        return struct

    def resolve_function(
        self,
        name: str,
        *,
        module_key: ModuleKey | None = None,
    ) -> SemanticFunction | None:
        """
        title: Resolve one registered function.
        parameters:
          name:
            type: str
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticFunction | None
        """
        lookup_module_key = module_key or self._current_module_key()
        return self.context.get_function(lookup_module_key, name)

    def resolve_struct(
        self,
        name: str,
        *,
        module_key: ModuleKey | None = None,
    ) -> SemanticStruct | None:
        """
        title: Resolve one registered struct.
        parameters:
          name:
            type: str
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticStruct | None
        """
        lookup_module_key = module_key or self._current_module_key()
        return self.context.get_struct(lookup_module_key, name)

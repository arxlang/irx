"""
title: Module-visible binding tables.
summary: >-
  Own the per-module namespace that imports and top-level declarations expose,
  keeping visible-name lookup separate from lexical scope resolution.
"""

from __future__ import annotations

from public import public

from irx import astx
from irx.analysis.context import SemanticContext
from irx.analysis.factories import SemanticEntityFactory
from irx.analysis.module_interfaces import ModuleKey
from irx.analysis.resolved_nodes import (
    SemanticBinding,
    SemanticClass,
    SemanticFunction,
    SemanticModule,
    SemanticStruct,
)
from irx.typecheck import typechecked


@public
@typechecked
class VisibleBindings:
    """
    title: Module-visible binding manager.
    summary: >-
      Track top-level and imported names that are visible within each module
      namespace and enforce conflicts independently from lexical scopes.
    attributes:
      context:
        type: SemanticContext
      factory:
        type: SemanticEntityFactory
      _bindings:
        type: dict[ModuleKey, dict[str, SemanticBinding]]
    """

    context: SemanticContext
    factory: SemanticEntityFactory
    _bindings: dict[ModuleKey, dict[str, SemanticBinding]]

    def __init__(
        self,
        *,
        context: SemanticContext,
        factory: SemanticEntityFactory,
        bindings: dict[ModuleKey, dict[str, SemanticBinding]] | None = None,
    ) -> None:
        """
        title: Initialize VisibleBindings.
        parameters:
          context:
            type: SemanticContext
          factory:
            type: SemanticEntityFactory
          bindings:
            type: dict[ModuleKey, dict[str, SemanticBinding]] | None
        """
        self.context = context
        self.factory = factory
        self._bindings = bindings or {}

    @property
    def tables(self) -> dict[ModuleKey, dict[str, SemanticBinding]]:
        """
        title: Return the backing binding tables.
        returns:
          type: dict[ModuleKey, dict[str, SemanticBinding]]
        """
        return self._bindings

    def _current_module_key(
        self,
        module_key: ModuleKey | None = None,
    ) -> ModuleKey:
        """
        title: Return the active or explicit module key.
        parameters:
          module_key:
            type: ModuleKey | None
        returns:
          type: ModuleKey
        """
        return module_key or self.context.current_module_key or "<root>"

    def bindings_for(
        self,
        module_key: ModuleKey | None = None,
    ) -> dict[str, SemanticBinding]:
        """
        title: Return one module's visible binding table.
        parameters:
          module_key:
            type: ModuleKey | None
        returns:
          type: dict[str, SemanticBinding]
        """
        current_module_key = self._current_module_key(module_key)
        return self._bindings.setdefault(current_module_key, {})

    def binding_conflicts(
        self,
        existing: SemanticBinding,
        new_binding: SemanticBinding,
    ) -> bool:
        """
        title: Return True when two bindings disagree.
        parameters:
          existing:
            type: SemanticBinding
          new_binding:
            type: SemanticBinding
        returns:
          type: bool
        """
        return (
            existing.kind != new_binding.kind
            or existing.qualified_name != new_binding.qualified_name
        )

    def bind(
        self,
        local_name: str,
        binding: SemanticBinding,
        *,
        node: astx.AST,
        module_key: ModuleKey | None = None,
    ) -> SemanticBinding:
        """
        title: Bind one visible name in a module namespace.
        parameters:
          local_name:
            type: str
          binding:
            type: SemanticBinding
          node:
            type: astx.AST
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticBinding
        """
        bindings = self.bindings_for(module_key)
        existing = bindings.get(local_name)
        if existing is not None and self.binding_conflicts(existing, binding):
            self.context.diagnostics.add(
                f"Conflicting binding for '{local_name}'",
                node=node,
            )
            return binding
        bindings[local_name] = binding
        return binding

    def bind_function(
        self,
        local_name: str,
        function: SemanticFunction,
        *,
        node: astx.AST,
        module_key: ModuleKey | None = None,
    ) -> SemanticBinding:
        """
        title: Bind a function in a module namespace.
        parameters:
          local_name:
            type: str
          function:
            type: SemanticFunction
          node:
            type: astx.AST
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticBinding
        """
        return self.bind(
            local_name,
            self.factory.make_function_binding(function),
            node=node,
            module_key=module_key,
        )

    def bind_struct(
        self,
        local_name: str,
        struct: SemanticStruct,
        *,
        node: astx.AST,
        module_key: ModuleKey | None = None,
    ) -> SemanticBinding:
        """
        title: Bind a struct in a module namespace.
        parameters:
          local_name:
            type: str
          struct:
            type: SemanticStruct
          node:
            type: astx.AST
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticBinding
        """
        return self.bind(
            local_name,
            self.factory.make_struct_binding(struct),
            node=node,
            module_key=module_key,
        )

    def bind_class(
        self,
        local_name: str,
        class_: SemanticClass,
        *,
        node: astx.AST,
        module_key: ModuleKey | None = None,
    ) -> SemanticBinding:
        """
        title: Bind a class in a module namespace.
        parameters:
          local_name:
            type: str
          class_:
            type: SemanticClass
          node:
            type: astx.AST
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticBinding
        """
        return self.bind(
            local_name,
            self.factory.make_class_binding(class_),
            node=node,
            module_key=module_key,
        )

    def bind_module(
        self,
        local_name: str,
        module: SemanticModule,
        *,
        node: astx.AST,
        module_key: ModuleKey | None = None,
    ) -> SemanticBinding:
        """
        title: Bind an imported module in a module namespace.
        parameters:
          local_name:
            type: str
          module:
            type: SemanticModule
          node:
            type: astx.AST
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticBinding
        """
        return self.bind(
            local_name,
            self.factory.make_module_binding(module),
            node=node,
            module_key=module_key,
        )

    def resolve(
        self,
        name: str,
        *,
        module_key: ModuleKey | None = None,
    ) -> SemanticBinding | None:
        """
        title: Resolve one visible binding by name.
        parameters:
          name:
            type: str
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticBinding | None
        """
        current_module_key = self._current_module_key(module_key)
        bindings = self._bindings.get(current_module_key, {})
        return bindings.get(name)

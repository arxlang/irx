# mypy: disable-error-code=no-redef

"""
title: Import and module-binding semantic visitors.
summary: >-
  Resolve host-provided imports into semantic module-visible bindings while
  keeping import coordination separate from declaration and expression rules.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.visitors.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.typecheck import typechecked


@typechecked
class ImportVisitorMixin(SemanticVisitorMixinBase):
    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ImportStmt) -> None:
        """
        title: Visit ImportStmt nodes.
        parameters:
          node:
            type: astx.ImportStmt
        """
        self._set_type(node, None)
        if not self._imports_supported_here(node):
            return

        session = self.session
        assert session is not None

        resolved_imports = []
        for alias in node.names:
            resolved = session.resolve_import_specifier(
                self._current_module_key(),
                node,
                alias.name,
            )
            if resolved is None:
                continue
            semantic_module = self.factory.make_module(
                resolved.key,
                display_name=resolved.display_name,
            )
            local_name = alias.asname or alias.name
            binding = self.bindings.bind_module(
                local_name,
                semantic_module,
                node=alias,
            )
            resolved_binding = self.factory.make_import_binding(
                local_name=local_name,
                requested_name=alias.name,
                source_module_key=resolved.key,
                binding=binding,
            )
            resolved_imports.append(resolved_binding)
            self._set_module(alias, semantic_module)
            self._set_imports(alias, (resolved_binding,))
        self._set_imports(node, tuple(resolved_imports))

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ImportFromStmt) -> None:
        """
        title: Visit ImportFromStmt nodes.
        parameters:
          node:
            type: astx.ImportFromStmt
        """
        self._set_type(node, None)
        if not self._imports_supported_here(node):
            return

        session = self.session
        assert session is not None

        requested_specifier = f"{'.' * node.level}{node.module or ''}"
        resolved_module = session.resolve_import_specifier(
            self._current_module_key(),
            node,
            requested_specifier,
        )
        if resolved_module is None:
            return

        target_module = self.factory.make_module(
            resolved_module.key,
            display_name=resolved_module.display_name,
        )
        self._set_module(node, target_module)
        resolved_imports = []

        for alias in node.names:
            if alias.name == "*":
                self.context.diagnostics.add(
                    "Wildcard imports are not supported in this MVP.",
                    node=alias,
                )
                continue
            target_binding = self.bindings.resolve(
                alias.name,
                module_key=resolved_module.key,
            )
            if target_binding is None or target_binding.kind not in {
                "function",
                "struct",
            }:
                self.context.diagnostics.add(
                    f"Imported symbol '{alias.name}' was not found in "
                    f"module '{requested_specifier}'",
                    node=alias,
                )
                continue
            local_name = alias.asname or alias.name
            binding = self.bindings.bind(
                local_name,
                target_binding,
                node=alias,
            )
            resolved_binding = self.factory.make_import_binding(
                local_name=local_name,
                requested_name=alias.name,
                source_module_key=resolved_module.key,
                binding=binding,
            )
            resolved_imports.append(resolved_binding)
            self._set_module(alias, target_module)
            self._set_imports(alias, (resolved_binding,))
            if target_binding.function is not None:
                self._set_function(alias, target_binding.function)
            if target_binding.struct is not None:
                self._set_struct(alias, target_binding.struct)

        self._set_imports(node, tuple(resolved_imports))

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ImportExpr) -> None:
        """
        title: Visit ImportExpr nodes.
        parameters:
          node:
            type: astx.ImportExpr
        """
        self.context.diagnostics.add(
            "Import expressions are not supported in this MVP.",
            node=node,
        )
        self._set_type(node, None)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ImportFromExpr) -> None:
        """
        title: Visit ImportFromExpr nodes.
        parameters:
          node:
            type: astx.ImportFromExpr
        """
        self.context.diagnostics.add(
            "Import expressions are not supported in this MVP.",
            node=node,
        )
        self._set_type(node, None)

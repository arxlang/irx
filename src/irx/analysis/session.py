"""
title: Multi-module compilation session management.
summary: >-
  Track the reachable parsed-module graph, import edges, diagnostics, and
  visible bindings for one multi-module analysis run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from public import public

from irx import astx
from irx.analysis.module_interfaces import (
    ImportResolver,
    ModuleKey,
    ParsedModule,
)
from irx.analysis.resolved_nodes import SemanticBinding
from irx.diagnostics import DiagnosticBag
from irx.typecheck import typechecked

_IMPORTABLE_BINDING_KINDS = frozenset({"function", "struct", "class"})


@typechecked
def _module_import_specifier(node: astx.ImportFromStmt) -> str:
    """
    title: Return the resolver-facing module specifier for import-from nodes.
    summary: >-
      Reconstruct the raw module specifier string that should be handed back to
      the host resolver for a from-import statement.
    parameters:
      node:
        type: astx.ImportFromStmt
    returns:
      type: str
    """
    return f"{'.' * node.level}{node.module or ''}"


@typechecked
def _declared_importable_names(module: astx.Module) -> set[str]:
    """
    title: Return the directly declared importable names for one module.
    summary: >-
      Collect the top-level names that import-from statements may bind without
      consulting transitive imports.
    parameters:
      module:
        type: astx.Module
    returns:
      type: set[str]
    """
    names: set[str] = set()
    for node in module.nodes:
        if isinstance(node, astx.FunctionPrototype):
            names.add(node.name)
            continue
        if isinstance(node, astx.FunctionDef):
            names.add(node.prototype.name)
            continue
        if isinstance(node, (astx.StructDefStmt, astx.ClassDefStmt)):
            names.add(node.name)
    return names


@public
@typechecked
@dataclass
class CompilationSession:
    """
    title: Shared state for multi-module analysis and lowering.
    summary: >-
      Own the loaded module graph and cross-module binding state that analysis
      and lowering share for one compilation.
    attributes:
      root:
        type: ParsedModule
      resolver:
        type: ImportResolver
      modules:
        type: dict[ModuleKey, ParsedModule]
      graph:
        type: dict[ModuleKey, set[ModuleKey]]
      load_order:
        type: list[ModuleKey]
      diagnostics:
        type: DiagnosticBag
      visible_bindings:
        type: dict[ModuleKey, dict[str, SemanticBinding]]
      _resolution_cache:
        type: dict[tuple[ModuleKey, str], ParsedModule | None]
      _probe_cache:
        type: dict[tuple[ModuleKey, str], ParsedModule | None]
    """

    root: ParsedModule
    resolver: ImportResolver
    modules: dict[ModuleKey, ParsedModule] = field(default_factory=dict)
    graph: dict[ModuleKey, set[ModuleKey]] = field(default_factory=dict)
    load_order: list[ModuleKey] = field(default_factory=list)
    diagnostics: DiagnosticBag = field(default_factory=DiagnosticBag)
    visible_bindings: dict[ModuleKey, dict[str, SemanticBinding]] = field(
        default_factory=dict
    )
    _resolution_cache: dict[tuple[ModuleKey, str], ParsedModule | None] = (
        field(default_factory=dict)
    )
    _probe_cache: dict[tuple[ModuleKey, str], ParsedModule | None] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """
        title: Initialize session caches with the root module.
        summary: >-
          Seed the session with the root parsed module so graph expansion has
          an initial node.
        """
        self.register_module(self.root)

    def register_module(self, parsed_module: ParsedModule) -> ParsedModule:
        """
        title: Register one parsed module in the session cache.
        summary: >-
          Cache a parsed module once and initialize its graph and visible
          binding slots.
        parameters:
          parsed_module:
            type: ParsedModule
        returns:
          type: ParsedModule
        """
        existing = self.modules.get(parsed_module.key)
        if existing is not None:
            return existing
        self.modules[parsed_module.key] = parsed_module
        self.graph.setdefault(parsed_module.key, set())
        self.visible_bindings.setdefault(parsed_module.key, {})
        return parsed_module

    def module(self, module_key: ModuleKey) -> ParsedModule:
        """
        title: Return a parsed module by key.
        summary: >-
          Look up a previously-registered parsed module by its canonical host
          key.
        parameters:
          module_key:
            type: ModuleKey
        returns:
          type: ParsedModule
        """
        return self.modules[module_key]

    def ordered_modules(self) -> list[ParsedModule]:
        """
        title: Return parsed modules in stable dependency order.
        summary: >-
          Materialize the dependency-ordered module list used by later semantic
          and lowering passes.
        returns:
          type: list[ParsedModule]
        """
        return [self.modules[module_key] for module_key in self.load_order]

    def resolve_import_specifier(
        self,
        requesting_module_key: ModuleKey,
        import_node: astx.ImportStmt | astx.ImportFromStmt,
        requested_specifier: str,
    ) -> ParsedModule | None:
        """
        title: Resolve one import request through the host resolver.
        summary: >-
          Call the host resolver once per import request, memoizing both
          successes and failures.
        parameters:
          requesting_module_key:
            type: ModuleKey
          import_node:
            type: astx.ImportStmt | astx.ImportFromStmt
          requested_specifier:
            type: str
        returns:
          type: ParsedModule | None
        """
        cache_key = (requesting_module_key, requested_specifier)
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]
        if (
            cache_key in self._probe_cache
            and self._probe_cache[cache_key] is not None
        ):
            resolved = self._probe_cache[cache_key]
            self._resolution_cache[cache_key] = resolved
            return resolved

        try:
            resolved = self.resolver(
                requesting_module_key,
                import_node,
                requested_specifier,
            )
        except Exception as exc:
            self.diagnostics.add(
                f"Unable to resolve module '{requested_specifier}': {exc}",
                node=import_node,
                module_key=requesting_module_key,
            )
            self._resolution_cache[cache_key] = None
            return None

        self.register_module(resolved)
        self._probe_cache[cache_key] = resolved
        self._resolution_cache[cache_key] = resolved
        return resolved

    def probe_import_specifier(
        self,
        requesting_module_key: ModuleKey,
        import_node: astx.ImportStmt | astx.ImportFromStmt,
        requested_specifier: str,
    ) -> ParsedModule | None:
        """
        title: Probe one import request without emitting diagnostics.
        summary: >-
          Try the host resolver for speculative import edges such as child-
          module fallbacks while keeping expected missing-module probes silent
          but still surfacing unexpected resolver failures.
        parameters:
          requesting_module_key:
            type: ModuleKey
          import_node:
            type: astx.ImportStmt | astx.ImportFromStmt
          requested_specifier:
            type: str
        returns:
          type: ParsedModule | None
        """
        cache_key = (requesting_module_key, requested_specifier)
        if cache_key in self._probe_cache:
            return self._probe_cache[cache_key]

        try:
            resolved = self.resolver(
                requesting_module_key,
                import_node,
                requested_specifier,
            )
        except LookupError:
            self._probe_cache[cache_key] = None
            return None

        self.register_module(resolved)
        self._probe_cache[cache_key] = resolved
        return resolved

    def resolve_import_from_name(
        self,
        requesting_module_key: ModuleKey,
        import_node: astx.ImportFromStmt,
        parent_module_key: ModuleKey,
        imported_name: str,
    ) -> tuple[SemanticBinding | None, ParsedModule | None]:
        """
        title: >-
          Resolve one import-from name to a direct binding or child module.
        summary: >-
          Apply symbol-first, child-module-second import-from semantics using
          the parent module's visible bindings before attempting module
          namespace fallback.
        parameters:
          requesting_module_key:
            type: ModuleKey
          import_node:
            type: astx.ImportFromStmt
          parent_module_key:
            type: ModuleKey
          imported_name:
            type: str
        returns:
          type: tuple[SemanticBinding | None, ParsedModule | None]
        """
        target_binding = self.visible_bindings.get(parent_module_key, {}).get(
            imported_name
        )
        if (
            target_binding is not None
            and target_binding.kind in _IMPORTABLE_BINDING_KINDS
        ):
            return target_binding, None

        requested_specifier = f"{parent_module_key}.{imported_name}"
        resolved_child = self.probe_import_specifier(
            requesting_module_key,
            import_node,
            requested_specifier,
        )
        return None, resolved_child

    def expand_graph(self) -> None:
        """
        title: Expand the reachable import graph from the root module.
        summary: >-
          Walk top-level imports from the root module, load every reachable
          dependency, and reject cycles.
        """
        self.load_order.clear()
        temporary: list[ModuleKey] = []
        temporary_lookup: set[ModuleKey] = set()
        permanent: set[ModuleKey] = set()
        importable_names: dict[ModuleKey, set[str]] = {}

        def dfs(module_key: ModuleKey) -> set[str]:
            """
            title: Visit one reachable module during graph expansion.
            summary: >-
              Depth-first walk one module, record its outgoing edges, append it
              to the stable load order after its dependencies, and track which
              names later from-import statements may bind directly from that
              module.
            parameters:
              module_key:
                type: ModuleKey
            returns:
              type: set[str]
            """
            if module_key in permanent:
                return importable_names[module_key]
            if module_key in temporary_lookup:
                cycle_start = temporary.index(module_key)
                cycle_path = [*temporary[cycle_start:], module_key]
                cycle_str = " -> ".join(str(item) for item in cycle_path)
                self.diagnostics.add(
                    f"Cyclic import detected: {cycle_str}",
                    node=self.modules[module_key].ast,
                    module_key=module_key,
                )
                return importable_names.get(module_key, set())

            temporary.append(module_key)
            temporary_lookup.add(module_key)

            parsed_module = self.modules[module_key]
            dependencies: list[ModuleKey] = []
            module_importable_names = importable_names.setdefault(
                module_key,
                _declared_importable_names(parsed_module.ast),
            )
            for node in parsed_module.ast.nodes:
                if isinstance(node, astx.ImportStmt):
                    for alias in node.names:
                        resolved = self.resolve_import_specifier(
                            module_key,
                            node,
                            alias.name,
                        )
                        if resolved is None:
                            continue
                        self.graph.setdefault(module_key, set()).add(
                            resolved.key
                        )
                        dependencies.append(resolved.key)
                elif isinstance(node, astx.ImportFromStmt):
                    resolved_parent = self.resolve_import_specifier(
                        module_key,
                        node,
                        _module_import_specifier(node),
                    )
                    if resolved_parent is None:
                        continue
                    self.graph.setdefault(module_key, set()).add(
                        resolved_parent.key
                    )
                    dependencies.append(resolved_parent.key)
                    parent_importable_names = dfs(resolved_parent.key)

                    for alias in node.names:
                        if alias.name == "*":
                            continue

                        local_name = alias.asname or alias.name
                        if alias.name in parent_importable_names:
                            module_importable_names.add(local_name)
                            continue

                        resolved_child = self.probe_import_specifier(
                            module_key,
                            node,
                            f"{resolved_parent.key}.{alias.name}",
                        )
                        if resolved_child is None:
                            continue
                        self.graph.setdefault(module_key, set()).add(
                            resolved_child.key
                        )
                        dependencies.append(resolved_child.key)

            for dependency_key in dependencies:
                dfs(dependency_key)

            temporary.pop()
            temporary_lookup.remove(module_key)
            permanent.add(module_key)
            self.load_order.append(module_key)
            return module_importable_names

        dfs(self.root.key)

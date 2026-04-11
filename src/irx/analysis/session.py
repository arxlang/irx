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
        self._resolution_cache[cache_key] = resolved
        return resolved

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

        def dfs(module_key: ModuleKey) -> None:
            """
            title: Visit one reachable module during graph expansion.
            summary: >-
              Depth-first walk one module, record its outgoing edges, and
              append it to the stable load order after its dependencies.
            parameters:
              module_key:
                type: ModuleKey
            """
            if module_key in permanent:
                return
            if module_key in temporary_lookup:
                cycle_start = temporary.index(module_key)
                cycle_path = [*temporary[cycle_start:], module_key]
                cycle_str = " -> ".join(str(item) for item in cycle_path)
                self.diagnostics.add(
                    f"Cyclic import detected: {cycle_str}",
                    node=self.modules[module_key].ast,
                    module_key=module_key,
                )
                return

            temporary.append(module_key)
            temporary_lookup.add(module_key)

            parsed_module = self.modules[module_key]
            dependencies: list[ModuleKey] = []
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
                    resolved = self.resolve_import_specifier(
                        module_key,
                        node,
                        _module_import_specifier(node),
                    )
                    if resolved is None:
                        continue
                    self.graph.setdefault(module_key, set()).add(resolved.key)
                    dependencies.append(resolved.key)

            for dependency_key in dependencies:
                dfs(dependency_key)

            temporary.pop()
            temporary_lookup.remove(module_key)
            permanent.add(module_key)
            self.load_order.append(module_key)

        dfs(self.root.key)

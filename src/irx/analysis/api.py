"""
title: Public semantic-analysis API entry points.
summary: >-
  Expose the stable semantic-analysis entry points that hosts and backend
  lowering call before code generation begins, while keeping the concrete
  analyzer implementation in smaller internal modules.
"""

from __future__ import annotations

from typing import cast

from public import public

from irx import astx
from irx.analysis.analyzer import SemanticAnalyzer
from irx.analysis.module_interfaces import (
    ImportResolver,
    ParsedModule,
)
from irx.analysis.session import CompilationSession
from irx.typecheck import typechecked

__all__ = [
    "SemanticAnalyzer",
    "analyze",
    "analyze_module",
    "analyze_modules",
]


@public
@typechecked
def analyze(node: astx.AST) -> astx.AST:
    """
    title: Analyze one AST root and attach semantic sidecars.
    summary: >-
      Run the stable single-root semantic-validation path, attach node.semantic
      sidecars to analyzed nodes, and raise SemanticError before lowering when
      diagnostics exist.
    parameters:
      node:
        type: astx.AST
    returns:
      type: astx.AST
    """
    return SemanticAnalyzer().analyze(node)


@public
@typechecked
def analyze_module(module: astx.Module) -> astx.Module:
    """
    title: Analyze an AST module.
    summary: >-
      Convenience wrapper around analyze(...) for module roots with the same
      semantic-error boundary and sidecar guarantees.
    parameters:
      module:
        type: astx.Module
    returns:
      type: astx.Module
    """
    return cast(astx.Module, analyze(module))


@public
@typechecked
def analyze_modules(
    root: ParsedModule,
    resolver: ImportResolver,
) -> CompilationSession:
    """
    title: Analyze a reachable graph of host-provided parsed modules.
    summary: >-
      Run the stable multi-module semantic pipeline: expand the reachable
      module graph, predeclare top-level members, resolve top-level imports,
      attach semantic sidecars, and raise SemanticError before lowering when
      diagnostics exist.
    parameters:
      root:
        type: ParsedModule
      resolver:
        type: ImportResolver
    returns:
      type: CompilationSession
    """
    session = CompilationSession(root=root, resolver=resolver)
    session.expand_graph()

    analyzer = SemanticAnalyzer(session=session)

    for parsed_module in session.ordered_modules():
        with analyzer.context.in_module(parsed_module.key):
            analyzer._predeclare_module_members(parsed_module.ast)

    for parsed_module in session.ordered_modules():
        with analyzer.context.in_module(parsed_module.key):
            with analyzer.context.scope("module"):
                for node in parsed_module.ast.nodes:
                    if isinstance(
                        node,
                        (astx.ImportStmt, astx.ImportFromStmt),
                    ):
                        analyzer.visit(node)

    for parsed_module in session.ordered_modules():
        analyzer.analyze_parsed_module(parsed_module, predeclared=True)

    session.diagnostics.raise_if_errors()
    return session

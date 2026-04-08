"""
title: Public semantic-analysis entry points.
summary: >-
  Expose the public single- and multi-module semantic-analysis functions while
  keeping the concrete analyzer implementation in smaller internal modules.
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

__all__ = [
    "SemanticAnalyzer",
    "analyze",
    "analyze_module",
    "analyze_modules",
]


@public
def analyze(node: astx.AST) -> astx.AST:
    """
    title: Analyze one AST root and attach node.semantic sidecars.
    parameters:
      node:
        type: astx.AST
    returns:
      type: astx.AST
    """
    return SemanticAnalyzer().analyze(node)


@public
def analyze_module(module: astx.Module) -> astx.Module:
    """
    title: Analyze an AST module.
    parameters:
      module:
        type: astx.Module
    returns:
      type: astx.Module
    """
    return cast(astx.Module, analyze(module))


@public
def analyze_modules(
    root: ParsedModule,
    resolver: ImportResolver,
) -> CompilationSession:
    """
    title: Analyze a reachable graph of host-provided parsed modules.
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

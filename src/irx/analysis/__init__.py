"""
title: Semantic analysis package.
summary: >-
  Re-export the semantic-analysis APIs and sidecar types used by hosts, tests,
  and backend codegen.
"""

from irx.analysis.api import analyze, analyze_module, analyze_modules
from irx.analysis.diagnostics import Diagnostic, DiagnosticBag, SemanticError
from irx.analysis.module_interfaces import (
    ImportResolver,
    ModuleKey,
    ParsedModule,
)
from irx.analysis.resolved_nodes import (
    ResolvedAssignment,
    ResolvedImportBinding,
    ResolvedOperator,
    SemanticBinding,
    SemanticFlags,
    SemanticFunction,
    SemanticInfo,
    SemanticModule,
    SemanticStruct,
    SemanticSymbol,
)
from irx.analysis.session import CompilationSession

__all__ = [
    "CompilationSession",
    "Diagnostic",
    "DiagnosticBag",
    "ImportResolver",
    "ModuleKey",
    "ParsedModule",
    "ResolvedAssignment",
    "ResolvedImportBinding",
    "ResolvedOperator",
    "SemanticBinding",
    "SemanticError",
    "SemanticFlags",
    "SemanticFunction",
    "SemanticInfo",
    "SemanticModule",
    "SemanticStruct",
    "SemanticSymbol",
    "analyze",
    "analyze_module",
    "analyze_modules",
]

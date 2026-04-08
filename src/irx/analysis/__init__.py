"""
title: Semantic analysis package.
summary: >-
  Re-export the semantic-analysis APIs, semantic-contract types, and sidecar
  metadata used by hosts, tests, and backend lowering.
"""

from irx.analysis.api import (
    SemanticAnalyzer,
    analyze,
    analyze_module,
    analyze_modules,
)
from irx.analysis.contract import (
    PhaseErrorBoundary,
    SemanticContract,
    SemanticPhase,
    get_semantic_contract,
)
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
    "PhaseErrorBoundary",
    "ResolvedAssignment",
    "ResolvedImportBinding",
    "ResolvedOperator",
    "SemanticAnalyzer",
    "SemanticBinding",
    "SemanticContract",
    "SemanticError",
    "SemanticFlags",
    "SemanticFunction",
    "SemanticInfo",
    "SemanticModule",
    "SemanticPhase",
    "SemanticStruct",
    "SemanticSymbol",
    "analyze",
    "analyze_module",
    "analyze_modules",
    "get_semantic_contract",
]

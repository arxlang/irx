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
from irx.analysis.module_interfaces import (
    ImportResolver,
    ModuleKey,
    ParsedModule,
)
from irx.analysis.resolved_nodes import (
    ClassMemberKind,
    FFIAdmissibility,
    FFICallableInfo,
    FFILinkStrategy,
    FFITypeClass,
    FFITypeInfo,
    ResolvedAssignment,
    ResolvedImportBinding,
    ResolvedOperator,
    SemanticBinding,
    SemanticClass,
    SemanticClassMember,
    SemanticFlags,
    SemanticFunction,
    SemanticInfo,
    SemanticModule,
    SemanticStruct,
    SemanticSymbol,
)
from irx.analysis.session import CompilationSession
from irx.diagnostics import Diagnostic, DiagnosticBag, SemanticError

__all__ = [
    "ClassMemberKind",
    "CompilationSession",
    "Diagnostic",
    "DiagnosticBag",
    "FFIAdmissibility",
    "FFICallableInfo",
    "FFILinkStrategy",
    "FFITypeClass",
    "FFITypeInfo",
    "ImportResolver",
    "ModuleKey",
    "ParsedModule",
    "PhaseErrorBoundary",
    "ResolvedAssignment",
    "ResolvedImportBinding",
    "ResolvedOperator",
    "SemanticAnalyzer",
    "SemanticBinding",
    "SemanticClass",
    "SemanticClassMember",
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

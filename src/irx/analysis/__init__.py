"""
title: Semantic analysis package.
"""

from irx.analysis.diagnostics import Diagnostic, DiagnosticBag, SemanticError
from irx.analysis.facade import analyze, analyze_module
from irx.analysis.resolved_nodes import (
    ResolvedAssignment,
    ResolvedOperator,
    SemanticFlags,
    SemanticFunction,
    SemanticInfo,
    SemanticSymbol,
)

__all__ = [
    "Diagnostic",
    "DiagnosticBag",
    "ResolvedAssignment",
    "ResolvedOperator",
    "SemanticError",
    "SemanticFlags",
    "SemanticFunction",
    "SemanticInfo",
    "SemanticSymbol",
    "analyze",
    "analyze_module",
]

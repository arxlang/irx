"""
title: Semantic-analysis diagnostics re-exports.
summary: >-
  Keep the public semantic diagnostics import path stable while delegating the
  concrete implementation to the shared diagnostics module.
"""

from irx.diagnostics import (
    Diagnostic,
    DiagnosticBag,
    DiagnosticRelatedInformation,
    SemanticError,
    SourceLocation,
    format_source_location,
    get_node_source_location,
)

__all__ = [
    "Diagnostic",
    "DiagnosticBag",
    "DiagnosticRelatedInformation",
    "SemanticError",
    "SourceLocation",
    "format_source_location",
    "get_node_source_location",
]

"""
title: Public semantic contract for hosts and lowering.
summary: >-
  Encode the stable semantic-analysis phases, required metadata, host input
  constraints, and phase error boundaries that IRx guarantees before backend
  lowering begins.
"""

from __future__ import annotations

from dataclasses import dataclass

from public import public

from irx.typecheck import typechecked

__all__ = [
    "PhaseErrorBoundary",
    "SemanticContract",
    "SemanticPhase",
    "get_semantic_contract",
]


@public
@typechecked
@dataclass(frozen=True)
class SemanticPhase:
    """
    title: One stable semantic phase.
    summary: >-
      Describe one externally-visible stage in the semantic pipeline and the
      guarantees it establishes for later lowering.
    attributes:
      name:
        type: str
      entrypoints:
        type: tuple[str, Ellipsis]
      guarantees:
        type: tuple[str, Ellipsis]
    """

    name: str
    entrypoints: tuple[str, ...]
    guarantees: tuple[str, ...]


@public
@typechecked
@dataclass(frozen=True)
class PhaseErrorBoundary:
    """
    title: One compiler-phase error boundary.
    summary: >-
      Describe which category of failure belongs to which pipeline phase and
      how callers should interpret it.
    attributes:
      phase:
        type: str
      raises:
        type: str
      surfaces:
        type: tuple[str, Ellipsis]
      summary:
        type: str
    """

    phase: str
    raises: str
    surfaces: tuple[str, ...]
    summary: str


@public
@typechecked
@dataclass(frozen=True)
class SemanticContract:
    """
    title: Stable semantic contract exposed by IRx.
    summary: >-
      Collect the semantic phases, metadata requirements, host input
      constraints, and error boundaries that analysis guarantees to lowering
      and host compilers.
    attributes:
      stable_phases:
        type: tuple[SemanticPhase, Ellipsis]
      required_node_semantic_fields:
        type: tuple[str, Ellipsis]
      required_session_fields:
        type: tuple[str, Ellipsis]
      allowed_host_entrypoints:
        type: tuple[str, Ellipsis]
      allowed_host_inputs:
        type: tuple[str, Ellipsis]
      phase_error_boundaries:
        type: tuple[PhaseErrorBoundary, Ellipsis]
    """

    stable_phases: tuple[SemanticPhase, ...]
    required_node_semantic_fields: tuple[str, ...]
    required_session_fields: tuple[str, ...]
    allowed_host_entrypoints: tuple[str, ...]
    allowed_host_inputs: tuple[str, ...]
    phase_error_boundaries: tuple[PhaseErrorBoundary, ...]


_SEMANTIC_CONTRACT = SemanticContract(
    stable_phases=(
        SemanticPhase(
            name="module_graph_expansion",
            entrypoints=("irx.analysis.analyze_modules",),
            guarantees=(
                "Resolve the reachable ParsedModule graph through the host "
                "ImportResolver.",
                "Record a stable dependency order in CompilationSession."
                "load_order.",
                "Report missing-module and cyclic-import diagnostics before "
                "full module analysis starts.",
            ),
        ),
        SemanticPhase(
            name="top_level_predeclaration",
            entrypoints=("irx.analysis.analyze_modules",),
            guarantees=(
                "Register top-level functions and structs for every reachable "
                "module before body validation.",
                "Establish module-aware semantic identities that later "
                "imports and lowering reuse.",
            ),
        ),
        SemanticPhase(
            name="top_level_import_resolution",
            entrypoints=("irx.analysis.analyze_modules",),
            guarantees=(
                "Resolve module-top-level imports into module-visible "
                "bindings.",
                "Reject unsupported import forms before lowering.",
            ),
        ),
        SemanticPhase(
            name="semantic_validation",
            entrypoints=(
                "irx.analysis.analyze",
                "irx.analysis.analyze_module",
                "irx.analysis.analyze_modules",
            ),
            guarantees=(
                "Attach SemanticInfo sidecars to analyzed AST nodes.",
                "Normalize resolved symbols, functions, structs, modules, "
                "operators, assignment targets, and semantic flags.",
                "Raise SemanticError instead of entering lowering when "
                "diagnostics exist.",
            ),
        ),
    ),
    required_node_semantic_fields=(
        "resolved_type",
        "resolved_symbol",
        "resolved_function",
        "resolved_struct",
        "resolved_module",
        "resolved_imports",
        "resolved_operator",
        "resolved_assignment",
        "semantic_flags",
        "extras",
    ),
    required_session_fields=(
        "root",
        "modules",
        "graph",
        "load_order",
        "visible_bindings",
    ),
    allowed_host_entrypoints=(
        "irx.analysis.analyze",
        "irx.analysis.analyze_module",
        "irx.analysis.analyze_modules",
    ),
    allowed_host_inputs=(
        "Hosts pass ASTx nodes and host-owned ParsedModule values into IRx; "
        "IRx does not parse source text or discover packages.",
        "Single-root analysis applies to AST roots that do not require "
        "cross-module import resolution.",
        "Cross-module analysis requires analyze_modules(root, resolver) with "
        "a host-supplied ImportResolver.",
        "Imports are part of the stable contract only at module top level in "
        "the current MVP.",
        "Wildcard imports and import expressions are rejected semantically "
        "and are not part of the lowering contract.",
    ),
    phase_error_boundaries=(
        PhaseErrorBoundary(
            phase="semantic",
            raises="SemanticError",
            surfaces=(
                "irx.analysis.analyze",
                "irx.analysis.analyze_module",
                "irx.analysis.analyze_modules",
                "irx.builder.Visitor.translate",
                "irx.builder.Builder.translate",
            ),
            summary=(
                "Meaning, validity, and supported-input failures stop at the "
                "semantic boundary before lowering begins."
            ),
        ),
        PhaseErrorBoundary(
            phase="lowering",
            raises="implementation-defined backend exception",
            surfaces=(
                "irx.builder.Visitor.translate",
                "irx.builder.Builder.translate",
            ),
            summary=(
                "Failures after successful semantic analysis belong to LLVM "
                "IR emission or backend implementation, not to semantic "
                "validation."
            ),
        ),
        PhaseErrorBoundary(
            phase="linking_runtime",
            raises="RuntimeError or toolchain/runtime exception",
            surfaces=("irx.builder.Builder.build",),
            summary=(
                "Native artifact compilation, linker execution, and runtime "
                "integration failures happen after lowering and outside the "
                "semantic contract."
            ),
        ),
    ),
)


@public
def get_semantic_contract() -> SemanticContract:
    """
    title: Return the stable public semantic contract.
    summary: >-
      Expose the semantic contract that hosts and backend lowering may rely on
      before code generation begins.
    returns:
      type: SemanticContract
    """
    return _SEMANTIC_CONTRACT

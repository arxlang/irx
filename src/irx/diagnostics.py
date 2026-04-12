"""
title: Shared diagnostics and source-location helpers.
summary: >-
  Provide one small diagnostics foundation that semantic analysis, lowering,
  native compilation, linking, and runtime feature resolution can all share.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Iterable

from public import public

from irx import astx
from irx.typecheck import typechecked

DEFAULT_DIAGNOSTIC_CODE_PREFIX = "IRX-"
_RENDERED_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*-[A-Z]\d{3,}$")


@public
@typechecked
@dataclass(frozen=True)
class DiagnosticCodeFormatter:
    """
    title: Render logical diagnostic identifiers with one configurable prefix.
    attributes:
      prefix:
        type: str
    """

    prefix: str = DEFAULT_DIAGNOSTIC_CODE_PREFIX

    def __post_init__(self) -> None:
        """
        title: Normalize the configured diagnostic prefix.
        """
        normalized = self.prefix.strip()
        if normalized and not normalized.endswith("-"):
            normalized = f"{normalized}-"
        object.__setattr__(self, "prefix", normalized)

    def format(self, code: str | None) -> str | None:
        """
        title: Format one logical diagnostic identifier.
        parameters:
          code:
            type: str | None
        returns:
          type: str | None
        """
        if code is None:
            return None
        stripped = code.strip()
        if not stripped:
            return None
        if _RENDERED_CODE_PATTERN.match(stripped):
            return stripped
        return f"{self.prefix}{stripped}"


_DIAGNOSTIC_CODE_FORMATTER = DiagnosticCodeFormatter()


@public
@typechecked
def get_diagnostic_code_formatter() -> DiagnosticCodeFormatter:
    """
    title: Return the active diagnostic-code formatter.
    returns:
      type: DiagnosticCodeFormatter
    """
    return _DIAGNOSTIC_CODE_FORMATTER


@public
@typechecked
def set_diagnostic_code_formatter(
    formatter: DiagnosticCodeFormatter,
) -> DiagnosticCodeFormatter:
    """
    title: Replace the process-wide diagnostic-code formatter.
    parameters:
      formatter:
        type: DiagnosticCodeFormatter
    returns:
      type: DiagnosticCodeFormatter
    """
    globals()["_DIAGNOSTIC_CODE_FORMATTER"] = formatter
    return formatter


@public
@typechecked
def set_diagnostic_code_prefix(prefix: str) -> DiagnosticCodeFormatter:
    """
    title: Configure the process-wide diagnostic-code prefix.
    parameters:
      prefix:
        type: str
    returns:
      type: DiagnosticCodeFormatter
    """
    return set_diagnostic_code_formatter(DiagnosticCodeFormatter(prefix))


@public
@typechecked
def format_diagnostic_code(
    code: str | None,
    *,
    code_formatter: DiagnosticCodeFormatter | None = None,
) -> str | None:
    """
    title: Render one logical diagnostic identifier.
    parameters:
      code:
        type: str | None
      code_formatter:
        type: DiagnosticCodeFormatter | None
    returns:
      type: str | None
    """
    formatter = code_formatter or get_diagnostic_code_formatter()
    return formatter.format(code)


@public
@typechecked
@dataclass(frozen=True)
class SourceLocation:
    """
    title: One best-effort source location.
    attributes:
      line:
        type: int | None
      col:
        type: int | None
      end_line:
        type: int | None
      end_col:
        type: int | None
    """

    line: int | None = None
    col: int | None = None
    end_line: int | None = None
    end_col: int | None = None

    def is_known(self) -> bool:
        """
        title: Return whether any source position is known.
        returns:
          type: bool
        """
        return self.line is not None or self.col is not None

    def format(self) -> str:
        """
        title: Render one location without module identity.
        returns:
          type: str
        """
        if self.line is None:
            return ""
        if self.col is None:
            return str(self.line)
        return f"{self.line}:{self.col}"


@typechecked
def _non_negative_int(value: object) -> int | None:
    """
    title: Return one non-negative integer when present.
    parameters:
      value:
        type: object
    returns:
      type: int | None
    """
    if isinstance(value, int) and value >= 0:
        return value
    return None


@public
@typechecked
def source_location_from_loc(loc: object | None) -> SourceLocation | None:
    """
    title: Convert one arbitrary location-like object to SourceLocation.
    parameters:
      loc:
        type: object | None
    returns:
      type: SourceLocation | None
    """
    if loc is None:
        return None
    line = _non_negative_int(getattr(loc, "line", None))
    col = _non_negative_int(getattr(loc, "col", getattr(loc, "column", None)))
    end_line = _non_negative_int(getattr(loc, "end_line", None))
    end_col = _non_negative_int(
        getattr(loc, "end_col", getattr(loc, "end_column", None))
    )
    source = SourceLocation(
        line=line,
        col=col,
        end_line=end_line,
        end_col=end_col,
    )
    if not source.is_known():
        return None
    return source


@public
@typechecked
def get_node_source_location(node: astx.AST | None) -> SourceLocation | None:
    """
    title: Return one AST node's best-effort source location.
    parameters:
      node:
        type: astx.AST | None
    returns:
      type: SourceLocation | None
    """
    if node is None:
        return None
    return source_location_from_loc(getattr(node, "loc", None))


@typechecked
def _string_or_none(value: object) -> str | None:
    """
    title: Return one non-empty string when present.
    parameters:
      value:
        type: object
    returns:
      type: str | None
    """
    if isinstance(value, str) and value:
        return value
    return None


@public
@typechecked
def get_node_module_key(node: astx.AST | None) -> str | None:
    """
    title: Return one AST node's best-effort module attribution.
    parameters:
      node:
        type: astx.AST | None
    returns:
      type: str | None
    """
    if node is None:
        return None

    direct = _string_or_none(getattr(node, "module_key", None))
    if direct is not None:
        return direct

    semantic = getattr(node, "semantic", None)
    if semantic is None:
        return None

    candidates: list[object] = [
        getattr(semantic, "resolved_function", None),
        getattr(semantic, "resolved_symbol", None),
        getattr(semantic, "resolved_struct", None),
        getattr(semantic, "resolved_module", None),
    ]
    resolved_call = getattr(semantic, "resolved_call", None)
    if resolved_call is not None:
        candidates.append(
            getattr(getattr(resolved_call, "callee", None), "function", None)
        )
    resolved_return = getattr(semantic, "resolved_return", None)
    if resolved_return is not None:
        candidates.append(
            getattr(
                getattr(resolved_return, "callable", None), "function", None
            )
        )
    resolved_assignment = getattr(semantic, "resolved_assignment", None)
    if resolved_assignment is not None:
        candidates.append(getattr(resolved_assignment, "target", None))
    resolved_field_access = getattr(semantic, "resolved_field_access", None)
    if resolved_field_access is not None:
        candidates.append(getattr(resolved_field_access, "struct", None))

    for candidate in candidates:
        module_key = _string_or_none(getattr(candidate, "module_key", None))
        if module_key is not None:
            return module_key
    return None


@public
@typechecked
def format_source_location(
    module_key: str | None = None,
    source: SourceLocation | None = None,
) -> str:
    """
    title: Format one module-aware source location.
    parameters:
      module_key:
        type: str | None
      source:
        type: SourceLocation | None
    returns:
      type: str
    """
    location_text = source.format() if source is not None else ""
    if module_key and location_text:
        return f"{module_key}:{location_text}"
    if module_key:
        return module_key
    return location_text


@public
@typechecked
@dataclass(frozen=True)
class DiagnosticRelatedInformation:
    """
    title: One secondary diagnostic location or note.
    attributes:
      message:
        type: str
      node:
        type: astx.AST | None
      module_key:
        type: str | None
      source:
        type: SourceLocation | None
    """

    message: str
    node: astx.AST | None = None
    module_key: str | None = None
    source: SourceLocation | None = None

    def resolved_source(self) -> SourceLocation | None:
        """
        title: Return the related entry's source location.
        returns:
          type: SourceLocation | None
        """
        return self.source or get_node_source_location(self.node)

    def resolved_module_key(self) -> str | None:
        """
        title: Return the related entry's module attribution.
        returns:
          type: str | None
        """
        return self.module_key or get_node_module_key(self.node)


@public
@typechecked
@dataclass(frozen=True)
class Diagnostic:
    """
    title: One structured diagnostic record.
    attributes:
      message:
        type: str
      node:
        type: astx.AST | None
      code:
        type: str | None
      severity:
        type: str
      module_key:
        type: str | None
      phase:
        type: str
      source:
        type: SourceLocation | None
      notes:
        type: tuple[str, Ellipsis]
      hint:
        type: str | None
      cause:
        type: Exception | None
      related:
        type: tuple[DiagnosticRelatedInformation, Ellipsis]
    """

    message: str
    node: astx.AST | None = None
    code: str | None = None
    severity: str = "error"
    module_key: str | None = None
    phase: str = "semantic"
    source: SourceLocation | None = None
    notes: tuple[str, ...] = ()
    hint: str | None = None
    cause: Exception | None = None
    related: tuple[DiagnosticRelatedInformation, ...] = ()

    def resolved_source(self) -> SourceLocation | None:
        """
        title: Return the diagnostic's best-effort source location.
        returns:
          type: SourceLocation | None
        """
        return self.source or get_node_source_location(self.node)

    def resolved_module_key(self) -> str | None:
        """
        title: Return the diagnostic's best-effort module attribution.
        returns:
          type: str | None
        """
        return self.module_key or get_node_module_key(self.node)

    def rendered_code(
        self,
        *,
        code_formatter: DiagnosticCodeFormatter | None = None,
    ) -> str | None:
        """
        title: Return the final rendered diagnostic code.
        parameters:
          code_formatter:
            type: DiagnosticCodeFormatter | None
        returns:
          type: str | None
        """
        return format_diagnostic_code(
            self.code,
            code_formatter=code_formatter,
        )

    def format(
        self,
        *,
        code_formatter: DiagnosticCodeFormatter | None = None,
    ) -> str:
        """
        title: Format the diagnostic for human display.
        parameters:
          code_formatter:
            type: DiagnosticCodeFormatter | None
        returns:
          type: str
        """
        prefix = format_source_location(
            self.resolved_module_key(),
            self.resolved_source(),
        )
        prefix_text = f"{prefix}: " if prefix else ""
        rendered_code = self.rendered_code(code_formatter=code_formatter)
        label = self.severity
        if rendered_code is not None:
            label = f"{label}[{rendered_code}]"
        if self.phase and self.phase != "semantic":
            label = f"{label} ({self.phase})"

        lines = [f"{prefix_text}{label}: {self.message}"]
        for note in self.notes:
            lines.append(f"  note: {note}")
        if self.hint is not None:
            lines.append(f"  hint: {self.hint}")
        if self.cause is not None:
            cause_message = (
                str(self.cause).strip() or self.cause.__class__.__name__
            )
            lines.append(
                f"  cause: {self.cause.__class__.__name__}: {cause_message}"
            )
        for related in self.related:
            related_prefix = format_source_location(
                related.resolved_module_key(),
                related.resolved_source(),
            )
            related_text = (
                f"{related_prefix}: {related.message}"
                if related_prefix
                else related.message
            )
            lines.append(f"  related: {related_text}")
        return "\n".join(lines)


@public
@typechecked
class DiagnosticBag:
    """
    title: Collect diagnostics across one semantic-analysis attempt.
    attributes:
      diagnostics:
        type: list[Diagnostic]
      default_module_key:
        type: str | None
    """

    diagnostics: list[Diagnostic]
    default_module_key: str | None

    def __init__(self) -> None:
        """
        title: Initialize DiagnosticBag.
        """
        self.diagnostics = []
        self.default_module_key = None

    def add(
        self,
        message: str,
        *,
        node: astx.AST | None = None,
        code: str | None = None,
        severity: str = "error",
        module_key: str | None = None,
        phase: str = "semantic",
        source: SourceLocation | None = None,
        notes: Iterable[str] = (),
        hint: str | None = None,
        cause: Exception | None = None,
        related: Iterable[DiagnosticRelatedInformation] = (),
    ) -> None:
        """
        title: Add one diagnostic to the bag.
        parameters:
          message:
            type: str
          node:
            type: astx.AST | None
          code:
            type: str | None
          severity:
            type: str
          module_key:
            type: str | None
          phase:
            type: str
          source:
            type: SourceLocation | None
          notes:
            type: Iterable[str]
          hint:
            type: str | None
          cause:
            type: Exception | None
          related:
            type: Iterable[DiagnosticRelatedInformation]
        """
        self.diagnostics.append(
            Diagnostic(
                message=message,
                node=node,
                code=code,
                severity=severity,
                module_key=module_key or self.default_module_key,
                phase=phase,
                source=source,
                notes=tuple(notes),
                hint=hint,
                cause=cause,
                related=tuple(related),
            )
        )

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        """
        title: Extend the bag with additional diagnostics.
        parameters:
          diagnostics:
            type: Iterable[Diagnostic]
        """
        self.diagnostics.extend(diagnostics)

    def has_errors(self) -> bool:
        """
        title: Return True when the bag contains diagnostics.
        returns:
          type: bool
        """
        return bool(self.diagnostics)

    def format(
        self,
        *,
        code_formatter: DiagnosticCodeFormatter | None = None,
    ) -> str:
        """
        title: Format the whole bag.
        parameters:
          code_formatter:
            type: DiagnosticCodeFormatter | None
        returns:
          type: str
        """
        return "\n".join(
            diagnostic.format(code_formatter=code_formatter)
            for diagnostic in self.diagnostics
        )

    def raise_if_errors(self) -> None:
        """
        title: Raise SemanticError when diagnostics exist.
        """
        if self.has_errors():
            raise SemanticError(self)


@public
@typechecked
class IRxDiagnosticError(Exception):
    """
    title: Raised when one non-semantic compiler phase emits a diagnostic.
    attributes:
      diagnostic:
        type: Diagnostic
      code_formatter:
        type: DiagnosticCodeFormatter
    """

    diagnostic: Diagnostic
    code_formatter: DiagnosticCodeFormatter

    def __init__(
        self,
        diagnostic: Diagnostic,
        *,
        code_formatter: DiagnosticCodeFormatter | None = None,
    ) -> None:
        """
        title: Initialize IRxDiagnosticError.
        parameters:
          diagnostic:
            type: Diagnostic
          code_formatter:
            type: DiagnosticCodeFormatter | None
        """
        self.diagnostic = diagnostic
        self.code_formatter = code_formatter or get_diagnostic_code_formatter()
        super().__init__(diagnostic.format(code_formatter=self.code_formatter))

    def format(self) -> str:
        """
        title: Return the formatted diagnostic message.
        returns:
          type: str
        """
        return self.diagnostic.format(code_formatter=self.code_formatter)


@public
@typechecked
class LoweringError(IRxDiagnosticError):
    """
    title: Raised when lowering fails with one structured diagnostic.
    attributes:
      diagnostic:
        type: Diagnostic
      code_formatter:
        type: DiagnosticCodeFormatter
    """


@public
@typechecked
class NativeCompileError(IRxDiagnosticError):
    """
    title: Raised when native artifact compilation fails.
    attributes:
      diagnostic:
        type: Diagnostic
      code_formatter:
        type: DiagnosticCodeFormatter
    """


@public
@typechecked
class LinkingError(IRxDiagnosticError):
    """
    title: Raised when final executable linking fails.
    attributes:
      diagnostic:
        type: Diagnostic
      code_formatter:
        type: DiagnosticCodeFormatter
    """


@public
@typechecked
class RuntimeFeatureError(IRxDiagnosticError):
    """
    title: Raised when runtime feature activation or symbol resolution fails.
    attributes:
      diagnostic:
        type: Diagnostic
      code_formatter:
        type: DiagnosticCodeFormatter
    """


@public
@typechecked
class SemanticError(Exception):
    """
    title: Raised when semantic analysis fails.
    attributes:
      diagnostics:
        type: DiagnosticBag
      code_formatter:
        type: DiagnosticCodeFormatter
    """

    diagnostics: DiagnosticBag
    code_formatter: DiagnosticCodeFormatter

    def __init__(
        self,
        diagnostics: DiagnosticBag,
        *,
        code_formatter: DiagnosticCodeFormatter | None = None,
    ) -> None:
        """
        title: Initialize SemanticError.
        parameters:
          diagnostics:
            type: DiagnosticBag
          code_formatter:
            type: DiagnosticCodeFormatter | None
        """
        self.diagnostics = diagnostics
        self.code_formatter = code_formatter or get_diagnostic_code_formatter()
        super().__init__(
            diagnostics.format(code_formatter=self.code_formatter)
        )

    def format(self) -> str:
        """
        title: Return the formatted diagnostic bag.
        returns:
          type: str
        """
        return self.diagnostics.format(code_formatter=self.code_formatter)


@public
@typechecked
class DiagnosticCodes:
    """
    title: Stable logical diagnostic identifiers.
    summary: >-
      Group the most important semantic, lowering, FFI, runtime, compile, and
      link families under short stable identifiers. These identifiers are
      rendered through one shared DiagnosticCodeFormatter.
    """

    SEMANTIC_UNRESOLVED_NAME = "S001"
    SEMANTIC_DUPLICATE_DECLARATION = "S002"
    SEMANTIC_INVALID_CONDITION = "S003"
    SEMANTIC_INVALID_RETURN = "S004"
    SEMANTIC_INVALID_ASSIGNMENT_TARGET = "S005"
    SEMANTIC_INVALID_FIELD_ACCESS = "S006"
    SEMANTIC_INVALID_CONTROL_FLOW = "S007"
    SEMANTIC_CALL_ARITY = "S008"
    SEMANTIC_BUFFER_MISUSE = "S009"
    SEMANTIC_TYPE_MISMATCH = "S010"
    FFI_INVALID_SIGNATURE = "F001"
    LOWERING_MISSING_SEMANTIC_METADATA = "L001"
    LOWERING_TYPE_MISMATCH = "L010"
    LOWERING_INVALID_CONTROL_FLOW = "L011"
    NATIVE_COMPILE_FAILED = "C001"
    LINK_FAILED = "K001"
    RUNTIME_FEATURE_UNKNOWN = "R001"
    RUNTIME_FEATURE_SYMBOL_MISSING = "R002"
    RUNTIME_ARTIFACT_KIND_INVALID = "R003"


__all__ = [
    "DEFAULT_DIAGNOSTIC_CODE_PREFIX",
    "Diagnostic",
    "DiagnosticBag",
    "DiagnosticCodeFormatter",
    "DiagnosticCodes",
    "DiagnosticRelatedInformation",
    "IRxDiagnosticError",
    "LinkingError",
    "LoweringError",
    "NativeCompileError",
    "RuntimeFeatureError",
    "SemanticError",
    "SourceLocation",
    "format_diagnostic_code",
    "format_source_location",
    "get_diagnostic_code_formatter",
    "get_node_module_key",
    "get_node_source_location",
    "set_diagnostic_code_formatter",
    "set_diagnostic_code_prefix",
    "source_location_from_loc",
]

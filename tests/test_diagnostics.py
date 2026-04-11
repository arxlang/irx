"""
title: Tests for shared diagnostics formatting and code rendering.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from irx import astx
from irx.diagnostics import (
    Diagnostic,
    DiagnosticBag,
    DiagnosticCodeFormatter,
    DiagnosticCodes,
    DiagnosticRelatedInformation,
    SemanticError,
    get_diagnostic_code_formatter,
    set_diagnostic_code_formatter,
    set_diagnostic_code_prefix,
)


def _with_loc(node: astx.AST, line: int, col: int) -> astx.AST:
    """
    title: Attach a simple source location to one AST node.
    parameters:
      node:
        type: astx.AST
      line:
        type: int
      col:
        type: int
    returns:
      type: astx.AST
    """
    node.loc = SimpleNamespace(line=line, col=col)
    return node


def test_diagnostic_formats_module_line_and_code_on_one_line() -> None:
    """
    title: Diagnostics should render compact one-line location and code output.
    """
    node = _with_loc(astx.Identifier("missing"), 12, 8)
    diagnostic = Diagnostic(
        message="cannot resolve name 'missing'",
        node=node,
        module_key="module_a",
        code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
    )

    assert (
        diagnostic.format()
        == "module_a:12:8: error[IRX-S001]: cannot resolve name 'missing'"
    )


def test_diagnostic_formats_multiline_notes_hint_and_related() -> None:
    """
    title: Diagnostics should preserve structured follow-up context.
    """
    node = _with_loc(astx.Identifier("value"), 4, 2)
    previous = _with_loc(astx.Identifier("value"), 1, 1)
    diagnostic = Diagnostic(
        message="Identifier already declared: value",
        node=node,
        module_key="module_a",
        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
        notes=("duplicate declarations in one scope are not allowed",),
        hint="rename one declaration or move it to a nested scope",
        related=(
            DiagnosticRelatedInformation(
                "previous declaration is here",
                node=previous,
                module_key="module_a",
            ),
        ),
    )

    formatted = diagnostic.format()

    assert (
        "module_a:4:2: error[IRX-S002]: Identifier already declared: value"
        in formatted
    )
    assert (
        "note: duplicate declarations in one scope are not allowed"
        in formatted
    )
    assert (
        "hint: rename one declaration or move it to a nested scope"
        in formatted
    )
    assert "related: module_a:1:1: previous declaration is here" in formatted


def test_diagnostic_formats_cleanly_without_location() -> None:
    """
    title: Diagnostics without source data should still format compactly.
    """
    diagnostic = Diagnostic(
        message="link failed while producing 'demo'",
        code=DiagnosticCodes.LINK_FAILED,
        phase="link",
    )

    assert (
        diagnostic.format()
        == "error[IRX-K001] (link): link failed while producing 'demo'"
    )


def test_default_and_overridden_prefix_render_through_shared_formatter() -> (
    None
):
    """
    title: Diagnostic code rendering should use one shared configurable prefix.
    """
    diagnostic = Diagnostic(
        message="cannot assign Float64 to 'count' of type Int32",
        code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
    )
    original = get_diagnostic_code_formatter()

    try:
        assert diagnostic.format() == (
            "error[IRX-S010]: cannot assign Float64 to 'count' of type Int32"
        )
        set_diagnostic_code_prefix("ARX-")
        assert diagnostic.format() == (
            "error[ARX-S010]: cannot assign Float64 to 'count' of type Int32"
        )
        set_diagnostic_code_formatter(DiagnosticCodeFormatter("IRX-"))
        assert diagnostic.format() == (
            "error[IRX-S010]: cannot assign Float64 to 'count' of type Int32"
        )
    finally:
        set_diagnostic_code_formatter(original)


def test_semantic_error_preserves_aggregated_diagnostic_bag() -> None:
    """
    title: SemanticError should still expose the original diagnostic bag.
    """
    bag = DiagnosticBag()
    bag.add(
        "cannot resolve name 'missing'",
        node=_with_loc(astx.Identifier("missing"), 2, 3),
        module_key="demo",
        code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
    )

    with pytest.raises(SemanticError) as exc_info:
        bag.raise_if_errors()

    assert exc_info.value.diagnostics is bag
    assert "demo:2:3: error[IRX-S001]: cannot resolve name 'missing'" in str(
        exc_info.value
    )

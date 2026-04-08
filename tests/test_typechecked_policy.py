"""
title: Policy tests for project-wide runtime type checking.
"""

from __future__ import annotations

import ast

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "irx"


def _expr_name(node: ast.expr) -> str:
    """
    title: Return a compact AST expression name.
    parameters:
      node:
        type: ast.expr
    returns:
      type: str
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _expr_name(node.func)
    return ast.unparse(node)


def test_concrete_project_classes_are_typechecked() -> None:
    """
    title: Assert concrete project classes use the typechecked decorator.
    """
    missing: list[str] = []

    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            base_names = {_expr_name(base) for base in node.bases}
            if "Protocol" in base_names:
                continue

            decorator_names = {
                _expr_name(decorator) for decorator in node.decorator_list
            }
            if "typechecked" in decorator_names:
                continue

            missing.append(
                f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.name}"
            )

    assert not missing, (
        "Concrete classes under src/irx must use irx.typecheck.typechecked:\n"
        + "\n".join(missing)
    )

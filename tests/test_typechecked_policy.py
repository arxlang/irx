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


def _decorator_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> set[str]:
    """
    title: Return normalized decorator names for a function or class.
    parameters:
      node:
        type: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    returns:
      type: set[str]
    """
    return {_expr_name(decorator) for decorator in node.decorator_list}


def _is_typechecking_stub(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """
    title: Detect typing-only stub functions kept out of runtime.
    parameters:
      node:
        type: ast.FunctionDef | ast.AsyncFunctionDef
    returns:
      type: bool
    """
    if len(node.body) != 1:
        return False
    stmt = node.body[0]
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    )


def _is_protocol_class(node: ast.ClassDef) -> bool:
    """
    title: Return whether a class is a typing Protocol.
    parameters:
      node:
        type: ast.ClassDef
    returns:
      type: bool
    """
    base_names = {_expr_name(base) for base in node.bases}
    return "Protocol" in base_names


def _rebound_typechecked_names(tree: ast.Module) -> set[str]:
    """
    title: Return top-level function names wrapped by assignment.
    parameters:
      tree:
        type: ast.Module
    returns:
      type: set[str]
    """
    rebound: set[str] = set()

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if _expr_name(node.value.func) != "typechecked":
            continue
        if len(node.value.args) != 1:
            continue
        arg = node.value.args[0]
        if isinstance(arg, ast.Name) and arg.id == target.id:
            rebound.add(target.id)

    return rebound


def test_project_functions_are_typechecked() -> None:
    """
    title: Assert module-level project functions use the typechecked decorator.
    """
    missing: list[str] = []

    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        rebound = _rebound_typechecked_names(tree)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_typechecking_stub(node):
                continue
            if "typechecked" in _decorator_names(node):
                continue
            if node.name in rebound:
                continue

            missing.append(
                f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.name}"
            )

    assert not missing, (
        "Module-level functions under src/irx must use "
        "irx.typecheck.typechecked:\n" + "\n".join(missing)
    )


def test_concrete_project_classes_are_typechecked() -> None:
    """
    title: Assert concrete project classes use the typechecked decorator.
    summary: >-
      Concrete classes are required to use the class decorator so their methods
      are covered at runtime without repeating typechecked on every method.
    """
    missing: list[str] = []

    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            if _is_protocol_class(node):
                continue

            if "typechecked" in _decorator_names(node):
                continue

            missing.append(
                f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.name}"
            )

    assert not missing, (
        "Concrete classes under src/irx must use irx.typecheck.typechecked:\n"
        + "\n".join(missing)
    )

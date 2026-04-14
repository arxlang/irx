"""
title: Tests for class semantic analysis.
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.analysis import (
    ClassMemberResolutionKind,
    SemanticError,
    analyze,
)
from irx.analysis.module_symbols import qualified_class_name
from irx.analysis.resolved_nodes import SemanticInfo

from tests.conftest import make_module


def _semantic(node: astx.AST) -> SemanticInfo:
    """
    title: Return semantic sidecar information for a node.
    parameters:
      node:
        type: astx.AST
    returns:
      type: SemanticInfo
    """
    return cast(SemanticInfo, getattr(node, "semantic"))


def _class_type(name: str) -> astx.ClassType:
    """
    title: Build one named class type reference.
    parameters:
      name:
        type: str
    returns:
      type: astx.ClassType
    """
    return astx.ClassType(name)


def _method(
    name: str,
    *args: astx.Argument,
    return_type: astx.DataType | None = None,
    visibility: astx.VisibilityKind = astx.VisibilityKind.public,
    is_static: bool = False,
) -> astx.FunctionDef:
    """
    title: Build one class method definition.
    parameters:
      name:
        type: str
      return_type:
        type: astx.DataType | None
      visibility:
        type: astx.VisibilityKind
      is_static:
        type: bool
      args:
        type: astx.Argument
        variadic: positional
    returns:
      type: astx.FunctionDef
    """
    prototype = astx.FunctionPrototype(
        name,
        args=astx.Arguments(*args),
        return_type=return_type or astx.Int32(),
        visibility=visibility,
    )
    if is_static:
        prototype.is_static = True
    return astx.FunctionDef(prototype=prototype, body=astx.Block())


def _attribute(
    name: str,
    type_: astx.DataType,
    *,
    mutability: astx.MutabilityKind = astx.MutabilityKind.mutable,
    visibility: astx.VisibilityKind = astx.VisibilityKind.public,
    is_static: bool = False,
    value: astx.AST | None = None,
) -> astx.VariableDeclaration:
    """
    title: Build one class attribute declaration.
    parameters:
      name:
        type: str
      type_:
        type: astx.DataType
      mutability:
        type: astx.MutabilityKind
      visibility:
        type: astx.VisibilityKind
      is_static:
        type: bool
      value:
        type: astx.AST | None
    returns:
      type: astx.VariableDeclaration
    """
    declaration = astx.VariableDeclaration(
        name=name,
        type_=type_,
        mutability=mutability,
        visibility=visibility,
        scope=(astx.ScopeKind.global_ if is_static else astx.ScopeKind.local),
        value=value if value is not None else astx.Undefined(),
    )
    if is_static:
        declaration.is_static = True
    return declaration


def test_analyze_attaches_resolved_class_sidecar_and_qualified_name() -> None:
    """
    title: Class definitions receive stable semantic identities.
    """
    node = astx.ClassDefStmt(
        name="Vector",
        attributes=[_attribute("x", astx.Int32())],
    )

    analyze(make_module("pkg.tools", node))

    resolved = _semantic(node).resolved_class

    assert resolved is not None
    assert resolved.qualified_name == qualified_class_name(
        "pkg.tools", "Vector"
    )
    assert resolved.member_table["x"].type_.__class__ is astx.Int32


def test_analyze_rejects_duplicate_class_definitions() -> None:
    """
    title: Duplicate class names are rejected per module.
    """
    module = make_module(
        "app.main",
        astx.ClassDefStmt(name="Vector"),
        astx.ClassDefStmt(name="Vector"),
    )

    with pytest.raises(SemanticError, match="Class 'Vector' already defined"):
        analyze(module)


def test_analyze_resolves_class_bases_and_c3_mro() -> None:
    """
    title: Classes resolve bases and compute deterministic C3 linearization.
    """
    root = astx.ClassDefStmt(name="Root")
    left = astx.ClassDefStmt(
        name="Left",
        bases=[_class_type("Root")],
        methods=[_method("select")],
    )
    right = astx.ClassDefStmt(
        name="Right",
        bases=[_class_type("Root")],
        methods=[_method("select")],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Left"), _class_type("Right")],
    )

    analyze(make_module("app.main", root, left, right, child))

    resolved = _semantic(child).resolved_class

    assert resolved is not None
    assert [item.name for item in resolved.mro] == [
        "Child",
        "Left",
        "Right",
        "Root",
    ]
    assert resolved.member_table["select"].owner_name == "Left"
    assert resolved.member_resolution["select"].kind is (
        ClassMemberResolutionKind.INHERITED
    )
    assert [
        member.owner_name
        for member in resolved.member_resolution["select"].candidates
    ] == ["Left", "Right"]


def test_analyze_rejects_unknown_base_class() -> None:
    """
    title: Unknown base classes fail semantic resolution.
    """
    module = make_module(
        "app.main",
        astx.ClassDefStmt(name="Child", bases=[_class_type("Missing")]),
    )

    with pytest.raises(SemanticError, match="Unknown base class 'Missing'"):
        analyze(module)


def test_analyze_rejects_duplicate_base_classes() -> None:
    """
    title: Repeated direct bases are rejected.
    """
    base = astx.ClassDefStmt(name="Base")
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base"), _class_type("Base")],
    )

    with pytest.raises(SemanticError, match="repeats base class 'Base'"):
        analyze(make_module("app.main", base, child))


def test_analyze_rejects_self_inheritance() -> None:
    """
    title: Classes cannot inherit from themselves.
    """
    module = make_module(
        "app.main",
        astx.ClassDefStmt(name="SelfRef", bases=[_class_type("SelfRef")]),
    )

    with pytest.raises(SemanticError, match="cannot inherit from itself"):
        analyze(module)


def test_analyze_rejects_inheritance_cycles() -> None:
    """
    title: Cyclic inheritance is rejected.
    """
    first = astx.ClassDefStmt(name="First", bases=[_class_type("Second")])
    second = astx.ClassDefStmt(name="Second", bases=[_class_type("First")])

    with pytest.raises(SemanticError, match="inheritance cycle is invalid"):
        analyze(make_module("app.main", first, second))


def test_analyze_rejects_duplicate_member_names_in_one_class() -> None:
    """
    title: Attributes and methods share one class member namespace.
    """
    node = astx.ClassDefStmt(
        name="Vector",
        attributes=[_attribute("value", astx.Int32())],
        methods=[_method("value")],
    )

    with pytest.raises(
        SemanticError, match="Class member 'value' already defined"
    ):
        analyze(make_module("app.main", node))


def test_analyze_rejects_attribute_redeclaration_from_base() -> None:
    """
    title: Attributes may not be redeclared across inheritance.
    """
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[_attribute("value", astx.Int32())],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        attributes=[_attribute("value", astx.Int32())],
    )

    with pytest.raises(
        SemanticError, match="cannot redeclare inherited member 'value'"
    ):
        analyze(make_module("app.main", base, child))


def test_analyze_tracks_method_override_metadata() -> None:
    """
    title: Method overrides keep explicit override metadata.
    """
    base = astx.ClassDefStmt(
        name="Base",
        methods=[_method("render", astx.Argument("value", astx.Int32()))],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[_method("render", astx.Argument("value", astx.Int32()))],
    )

    analyze(make_module("app.main", base, child))

    resolved = _semantic(child).resolved_class

    assert resolved is not None
    member = resolved.declared_member_table["render"]
    assert member.overrides is not None
    assert member.overrides.endswith("::member::render")
    assert resolved.member_table["render"].owner_name == "Child"
    assert resolved.member_resolution["render"].kind is (
        ClassMemberResolutionKind.OVERRIDE
    )
    assert [
        candidate.owner_name
        for candidate in resolved.member_resolution["render"].candidates
    ] == ["Child", "Base"]


def test_analyze_rejects_method_override_signature_changes() -> None:
    """
    title: Overrides must match inherited signatures exactly.
    """
    base = astx.ClassDefStmt(
        name="Base",
        methods=[_method("render", astx.Argument("value", astx.Int32()))],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[_method("render", astx.Argument("value", astx.Float64()))],
    )

    with pytest.raises(
        SemanticError, match="must match inherited signature exactly"
    ):
        analyze(make_module("app.main", base, child))


def test_analyze_rejects_ambiguous_inherited_attribute() -> None:
    """
    title: Distinct inherited attributes with one name are rejected.
    """
    left = astx.ClassDefStmt(
        name="Left",
        attributes=[_attribute("value", astx.Int32())],
    )
    right = astx.ClassDefStmt(
        name="Right",
        attributes=[_attribute("value", astx.Int32())],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Left"), _class_type("Right")],
    )

    with pytest.raises(
        SemanticError, match="inherits ambiguous attribute 'value'"
    ):
        analyze(make_module("app.main", left, right, child))


def test_analyze_tracks_shared_ancestors_in_diamond_mro() -> None:
    """
    title: Diamond inheritance keeps one shared ancestor in the linearization.
    """
    root = astx.ClassDefStmt(
        name="Root",
        attributes=[_attribute("value", astx.Int32())],
    )
    left = astx.ClassDefStmt(name="Left", bases=[_class_type("Root")])
    right = astx.ClassDefStmt(
        name="Right",
        bases=[_class_type("Root")],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Left"), _class_type("Right")],
    )

    analyze(make_module("app.main", root, left, right, child))

    resolved = _semantic(child).resolved_class

    assert resolved is not None
    assert [item.name for item in resolved.shared_ancestors] == ["Root"]
    assert resolved.member_table["value"].owner_name == "Root"
    assert resolved.member_resolution["value"].kind is (
        ClassMemberResolutionKind.INHERITED
    )
    assert [
        candidate.owner_name
        for candidate in resolved.member_resolution["value"].candidates
    ] == ["Root"]


def test_analyze_rejects_inconsistent_c3_mro() -> None:
    """
    title: Inconsistent multiple-inheritance orderings are rejected.
    """
    x = astx.ClassDefStmt(name="X")
    y = astx.ClassDefStmt(name="Y")
    a = astx.ClassDefStmt(
        name="A",
        bases=[_class_type("X"), _class_type("Y")],
    )
    b = astx.ClassDefStmt(
        name="B",
        bases=[_class_type("Y"), _class_type("X")],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("A"), _class_type("B")],
    )

    with pytest.raises(SemanticError, match="has no consistent MRO"):
        analyze(make_module("app.main", x, y, a, b, child))

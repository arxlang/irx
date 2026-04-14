"""
title: Tests for class semantic analysis.
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.analysis import (
    ClassHeaderFieldKind,
    ClassInitializationSourceKind,
    ClassMemberResolutionKind,
    ClassObjectRepresentationKind,
    SemanticError,
    analyze,
)
from irx.analysis.module_symbols import (
    mangle_class_dispatch_name,
    mangle_class_name,
    mangle_class_static_name,
    qualified_class_name,
)
from irx.analysis.resolved_nodes import SemanticInfo

from tests.conftest import make_module

CLASS_HEADER_SLOT_COUNT = 2
FIRST_INSTANCE_STORAGE_INDEX = CLASS_HEADER_SLOT_COUNT
SECOND_INSTANCE_STORAGE_INDEX = FIRST_INSTANCE_STORAGE_INDEX + 1
THIRD_INSTANCE_STORAGE_INDEX = SECOND_INSTANCE_STORAGE_INDEX + 1
FOURTH_INSTANCE_STORAGE_INDEX = THIRD_INSTANCE_STORAGE_INDEX + 1


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


def _default_return_value(type_: astx.DataType) -> astx.AST | None:
    """
    title: Build one minimal default return value for a method test helper.
    parameters:
      type_:
        type: astx.DataType
    returns:
      type: astx.AST | None
    """
    if isinstance(type_, astx.NoneType):
        return None
    if isinstance(type_, astx.Boolean):
        return astx.LiteralBoolean(False)
    if isinstance(type_, astx.Float64):
        return astx.LiteralFloat64(0.0)
    if isinstance(type_, astx.Float32):
        return astx.LiteralFloat32(0.0)
    if isinstance(type_, astx.Int8):
        return astx.LiteralInt8(0)
    return astx.LiteralInt32(0)


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
    resolved_return_type = return_type or astx.Int32()
    prototype = astx.FunctionPrototype(
        name,
        args=astx.Arguments(*args),
        return_type=resolved_return_type,
        visibility=visibility,
    )
    if is_static:
        prototype.is_static = True
    body = astx.Block()
    default_value = _default_return_value(resolved_return_type)
    body.append(astx.FunctionReturn(default_value))
    return astx.FunctionDef(prototype=prototype, body=body)


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
        methods=[_method("left_only")],
    )
    right = astx.ClassDefStmt(
        name="Right",
        bases=[_class_type("Root")],
        methods=[_method("right_only")],
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
    assert resolved.member_table["left_only"].owner_name == "Left"
    assert resolved.member_table["right_only"].owner_name == "Right"


def test_analyze_rejects_conflicting_inherited_methods_without_override() -> (
    None
):
    """
    title: Sibling methods with one exact signature require an override.
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

    with pytest.raises(
        SemanticError,
        match="inherits conflicting methods named 'select'",
    ):
        analyze(make_module("app.main", root, left, right, child))


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
    assert "::member::render::overload::" in member.overrides
    assert resolved.member_table["render"].owner_name == "Child"
    assert resolved.member_resolution["render"].kind is (
        ClassMemberResolutionKind.OVERRIDE
    )
    assert [
        candidate.owner_name
        for candidate in resolved.member_resolution["render"].candidates
    ] == ["Child", "Base"]


def test_analyze_merges_inherited_method_overloads() -> None:
    """
    title: Same-name methods with distinct signatures remain overloads.
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

    analyze(make_module("app.main", base, child))

    resolved = _semantic(child).resolved_class

    assert resolved is not None
    assert "render" not in resolved.member_table
    assert [
        member.owner_name for member in resolved.method_groups["render"]
    ] == ["Child", "Base"]
    assert [
        resolution.kind for resolution in resolved.method_resolution["render"]
    ] == [
        ClassMemberResolutionKind.DECLARED,
        ClassMemberResolutionKind.INHERITED,
    ]


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


def test_analyze_builds_pointer_layout_and_static_storage_metadata() -> None:
    """
    title: Classes expose stable object-layout and static-storage metadata.
    """
    node = astx.ClassDefStmt(
        name="Vector",
        attributes=[
            _attribute("x", astx.Int32()),
            _attribute(
                "shared",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(7),
            ),
        ],
    )

    analyze(make_module("pkg.tools", node))

    resolved = _semantic(node).resolved_class

    assert resolved is not None
    assert resolved.layout is not None
    assert resolved.layout.object_representation is (
        ClassObjectRepresentationKind.POINTER
    )
    assert resolved.layout.llvm_name == mangle_class_name(
        "pkg.tools",
        "Vector",
    )
    assert resolved.layout.dispatch_global_name == mangle_class_dispatch_name(
        "pkg.tools",
        "Vector",
    )
    assert [header.kind for header in resolved.layout.header_fields] == [
        ClassHeaderFieldKind.TYPE_DESCRIPTOR,
        ClassHeaderFieldKind.DISPATCH_TABLE,
    ]
    assert [
        header.storage_index for header in resolved.layout.header_fields
    ] == [
        0,
        1,
    ]
    assert [
        field.member.name for field in resolved.layout.instance_fields
    ] == [
        "x",
    ]
    assert (
        resolved.layout.visible_field_slots["x"].storage_index
        == FIRST_INSTANCE_STORAGE_INDEX
    )
    assert [
        storage.member.name for storage in resolved.layout.static_fields
    ] == [
        "shared",
    ]
    assert resolved.layout.static_fields[0].global_name == (
        mangle_class_static_name("pkg.tools", "Vector", "shared")
    )


def test_analyze_reuses_dispatch_slots_with_unrelated_hierarchies() -> None:
    """
    title: Unrelated families do not shift dispatch slots in another family.
    """
    base = astx.ClassDefStmt(name="Base", methods=[_method("area")])
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[_method("paint")],
    )

    analyze(make_module("app.one", base, child))
    resolved_base = _semantic(base).resolved_class
    resolved_child = _semantic(child).resolved_class
    assert resolved_base is not None
    assert resolved_child is not None
    baseline_slots = {
        member.name: member.dispatch_slot
        for member in resolved_child.instance_methods
    }

    other = astx.ClassDefStmt(name="Other", methods=[_method("ping")])
    extra = astx.ClassDefStmt(
        name="Extra",
        bases=[_class_type("Other")],
        methods=[_method("pong")],
    )
    base_again = astx.ClassDefStmt(name="Base", methods=[_method("area")])
    child_again = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[_method("paint")],
    )

    analyze(make_module("app.two", other, extra, base_again, child_again))
    resolved_again = _semantic(child_again).resolved_class

    assert resolved_again is not None
    assert {
        member.name: member.dispatch_slot
        for member in resolved_again.instance_methods
    } == baseline_slots


def test_analyze_flattens_inherited_layout_in_canonical_storage_order() -> (
    None
):
    """
    title: Canonical class storage orders ancestors before derived fields.
    """
    root = astx.ClassDefStmt(
        name="Root",
        attributes=[_attribute("root", astx.Int32())],
    )
    left = astx.ClassDefStmt(
        name="Left",
        bases=[_class_type("Root")],
        attributes=[_attribute("left", astx.Boolean())],
    )
    right = astx.ClassDefStmt(
        name="Right",
        bases=[_class_type("Root")],
        attributes=[_attribute("right", astx.Float64())],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Left"), _class_type("Right")],
        attributes=[_attribute("child", astx.Int8())],
    )

    analyze(make_module("app.main", root, left, right, child))

    resolved = _semantic(child).resolved_class

    assert resolved is not None
    assert resolved.layout is not None
    assert [
        (field.owner_name, field.member.name)
        for field in resolved.layout.instance_fields
    ] == [
        ("Root", "root"),
        ("Left", "left"),
        ("Right", "right"),
        ("Child", "child"),
    ]
    assert (
        resolved.layout.visible_field_slots["root"].storage_index
        == FIRST_INSTANCE_STORAGE_INDEX
    )
    assert (
        resolved.layout.visible_field_slots["left"].storage_index
        == SECOND_INSTANCE_STORAGE_INDEX
    )
    assert (
        resolved.layout.visible_field_slots["right"].storage_index
        == THIRD_INSTANCE_STORAGE_INDEX
    )
    assert (
        resolved.layout.visible_field_slots["child"].storage_index
        == FOURTH_INSTANCE_STORAGE_INDEX
    )


def test_analyze_builds_class_initialization_plan_in_storage_order() -> None:
    """
    title: Default construction follows canonical ancestor-first storage order.
    """
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[
            _attribute("root", astx.Int32(), value=astx.LiteralInt32(3))
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        attributes=[
            _attribute("flag", astx.Boolean()),
            _attribute("count", astx.Int32(), value=astx.LiteralInt32(9)),
        ],
    )

    analyze(make_module("app.main", base, child))

    resolved = _semantic(child).resolved_class

    assert resolved is not None
    assert resolved.initialization is not None
    assert [
        initializer.field.member.name
        for initializer in resolved.initialization.instance_initializers
    ] == ["root", "flag", "count"]
    assert [
        initializer.owner_name
        for initializer in resolved.initialization.instance_initializers
    ] == ["Base", "Child", "Child"]
    assert [
        initializer.source_kind
        for initializer in resolved.initialization.instance_initializers
    ] == [
        ClassInitializationSourceKind.DECLARATION,
        ClassInitializationSourceKind.DEFAULT,
        ClassInitializationSourceKind.DECLARATION,
    ]


def test_analyze_tracks_static_initializer_sources() -> None:
    """
    title: Static fields keep one deterministic initialization-source plan.
    """
    node = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "shared",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(7),
            ),
            _attribute("count", astx.Int32(), is_static=True),
        ],
    )

    analyze(make_module("app.main", node))

    resolved = _semantic(node).resolved_class

    assert resolved is not None
    assert resolved.initialization is not None
    assert [
        initializer.storage.member.name
        for initializer in resolved.initialization.static_initializers
    ] == ["shared", "count"]
    assert [
        initializer.source_kind
        for initializer in resolved.initialization.static_initializers
    ] == [
        ClassInitializationSourceKind.DECLARATION,
        ClassInitializationSourceKind.DEFAULT,
    ]


def test_analyze_rejects_nonliteral_static_attribute_initializer() -> None:
    """
    title: Static fields require literal or default initialization for now.
    """
    helper_body = astx.Block()
    helper_body.append(astx.FunctionReturn(astx.LiteralInt32(1)))
    helper = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="helper",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=helper_body,
    )
    node = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "shared",
                astx.Int32(),
                is_static=True,
                value=astx.FunctionCall("helper", []),
            )
        ],
    )

    with pytest.raises(
        SemanticError,
        match="requires a literal initializer or default construction",
    ):
        analyze(make_module("app.main", helper, node))


def test_analyze_attaches_resolved_class_construction_metadata() -> None:
    """
    title: ClassConstruct expressions resolve one default initialization plan.
    """
    node = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute("value", astx.Int32(), value=astx.LiteralInt32(7))
        ],
    )
    construct = astx.ClassConstruct("Counter")
    main_body = astx.Block()
    main_body.append(astx.FunctionReturn(construct))
    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="make_counter",
            args=astx.Arguments(),
            return_type=_class_type("Counter"),
        ),
        body=main_body,
    )

    analyze(make_module("app.main", node, main))

    resolved = _semantic(construct).resolved_class_construction
    resolved_type = _semantic(construct).resolved_type

    assert resolved is not None
    assert resolved.class_.name == "Counter"
    assert [
        initializer.field.member.name
        for initializer in resolved.initialization.instance_initializers
    ] == ["value"]
    assert isinstance(resolved_type, astx.ClassType)
    assert resolved_type.qualified_name == qualified_class_name(
        "app.main",
        "Counter",
    )

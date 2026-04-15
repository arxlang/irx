"""
title: Tests for class method semantics and dispatch metadata.
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.analysis import MethodDispatchKind, SemanticError, analyze
from irx.analysis.resolved_nodes import SemanticInfo

from tests.conftest import make_module

CLASS_HEADER_SLOT_COUNT = 2
FIRST_INSTANCE_STORAGE_INDEX = CLASS_HEADER_SLOT_COUNT
RENDER_OVERLOAD_COUNT = 2
STATIC_LITERAL_VALUE = 7
INHERITED_STATIC_VALUE = 11
PROTECTED_STATIC_VALUE = 5
BASE_FIELD_VALUE = 13
BASE_METHOD_RESULT = 21
CHILD_METHOD_RESULT = 34
STATIC_ASSIGNED_VALUE = 19
INSTANCE_ASSIGNED_VALUE = 23
BASE_ASSIGNED_VALUE = 29
INCREMENTED_STATIC_VALUE = 8


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


def _attribute(
    name: str,
    type_: astx.DataType,
    *,
    visibility: astx.VisibilityKind = astx.VisibilityKind.public,
    mutability: astx.MutabilityKind = astx.MutabilityKind.mutable,
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
      visibility:
        type: astx.VisibilityKind
      mutability:
        type: astx.MutabilityKind
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


def _returning_method(
    name: str,
    return_value: astx.AST,
    *args: astx.Argument,
    return_type: astx.DataType | None = None,
    visibility: astx.VisibilityKind = astx.VisibilityKind.public,
    is_static: bool = False,
) -> astx.FunctionDef:
    """
    title: Build one class method definition with a single return.
    parameters:
      name:
        type: str
      return_value:
        type: astx.AST
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
    body = astx.Block()
    body.append(astx.FunctionReturn(return_value))
    return astx.FunctionDef(prototype=prototype, body=body)


def _method_body(value: astx.AST) -> astx.Block:
    """
    title: Build one single-return method body block.
    parameters:
      value:
        type: astx.AST
    returns:
      type: astx.Block
    """
    body = astx.Block()
    body.append(astx.FunctionReturn(value))
    return body


def _main_returning(value: astx.AST) -> astx.FunctionDef:
    """
    title: Build a simple int32-returning main function.
    parameters:
      value:
        type: astx.AST
    returns:
      type: astx.FunctionDef
    """
    body = astx.Block()
    body.append(astx.FunctionReturn(value))
    return astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=body,
    )


def test_analyze_instance_method_call_uses_hidden_receiver_metadata() -> None:
    """
    title: Instance method calls resolve hidden receiver lowering metadata.
    """
    area_method = _returning_method("area", astx.LiteralInt32(1))
    shape = astx.ClassDefStmt(name="Shape", methods=[area_method])
    call = astx.MethodCall(astx.Identifier("shape"), "area", [])
    measure = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="measure",
            args=astx.Arguments(astx.Argument("shape", _class_type("Shape"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(call),
    )

    analyze(make_module("app.main", shape, measure))

    resolved_class = _semantic(shape).resolved_class
    assert resolved_class is not None
    member = resolved_class.declared_member_table["area"]
    assert member.lowered_function is not None
    assert member.dispatch_slot is not None
    assert [
        parameter.name
        for parameter in member.lowered_function.signature.parameters
    ] == ["self"]
    assert isinstance(
        member.lowered_function.signature.parameters[0].type_, astx.ClassType
    )

    resolved_call = _semantic(call).resolved_method_call
    assert resolved_call is not None
    assert resolved_call.dispatch_kind is MethodDispatchKind.INDIRECT
    assert resolved_call.slot_index == member.dispatch_slot
    assert (
        resolved_call.function.symbol_id == member.lowered_function.symbol_id
    )


def test_analyze_static_method_call_resolves_direct_lowering() -> None:
    """
    title: Static method calls resolve without an implicit receiver.
    """
    identity = _returning_method(
        "identity",
        astx.Identifier("value"),
        astx.Argument("value", astx.Int32()),
        is_static=True,
    )
    math = astx.ClassDefStmt(name="Math", methods=[identity])
    call = astx.StaticMethodCall("Math", "identity", [astx.LiteralInt32(7)])

    analyze(make_module("app.main", math, _main_returning(call)))

    resolved_class = _semantic(math).resolved_class
    assert resolved_class is not None
    member = resolved_class.declared_member_table["identity"]
    assert member.lowered_function is not None
    assert member.dispatch_slot is None
    assert [
        parameter.name
        for parameter in member.lowered_function.signature.parameters
    ] == ["value"]

    resolved_call = _semantic(call).resolved_method_call
    assert resolved_call is not None
    assert resolved_call.dispatch_kind is MethodDispatchKind.DIRECT
    assert resolved_call.receiver_class is None


def test_analyze_self_field_access_uses_class_layout_slot() -> None:
    """
    title: Field access through self resolves against the class layout.
    """
    field_access = astx.FieldAccess(astx.Identifier("self"), "value")
    read = _returning_method("read", field_access)
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[_attribute("value", astx.Int32())],
        methods=[read],
    )

    analyze(make_module("app.main", counter))

    resolved_field = _semantic(field_access).resolved_class_field_access
    assert resolved_field is not None
    assert resolved_field.field.storage_index == FIRST_INSTANCE_STORAGE_INDEX
    assert resolved_field.member.name == "value"


def test_analyze_base_field_access_uses_selected_base_storage() -> None:
    """
    title: Base-qualified field access resolves the selected ancestor slot.
    """
    access = astx.BaseFieldAccess(
        astx.Identifier("value"),
        "Base",
        "count",
    )
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[
            _attribute(
                "count",
                astx.Int32(),
                value=astx.LiteralInt32(BASE_FIELD_VALUE),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
    )
    probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(astx.Argument("value", _class_type("Child"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(access),
    )

    analyze(make_module("app.main", base, child, probe))

    resolved_access = _semantic(access).resolved_base_class_field_access
    assert resolved_access is not None
    assert resolved_access.receiver_class.name == "Child"
    assert resolved_access.base_class.name == "Base"
    assert resolved_access.member.owner_name == "Base"
    assert resolved_access.field.owner_name == "Base"


def test_analyze_base_method_call_bypasses_override_dispatch() -> None:
    """
    title: Base-qualified method calls resolve directly to the base member.
    """
    base = astx.ClassDefStmt(
        name="Base",
        methods=[
            _returning_method(
                "area",
                astx.LiteralInt32(BASE_METHOD_RESULT),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[
            _returning_method(
                "area",
                astx.LiteralInt32(CHILD_METHOD_RESULT),
            )
        ],
    )
    call = astx.BaseMethodCall(
        astx.Identifier("shape"),
        "Base",
        "area",
        [],
    )
    probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(astx.Argument("shape", _class_type("Child"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(call),
    )

    analyze(make_module("app.main", base, child, probe))

    resolved_call = _semantic(call).resolved_method_call
    assert resolved_call is not None
    assert resolved_call.class_.name == "Base"
    assert resolved_call.receiver_class is not None
    assert resolved_call.receiver_class.name == "Child"
    assert resolved_call.member.owner_name == "Base"
    assert resolved_call.dispatch_kind is MethodDispatchKind.DIRECT
    assert resolved_call.slot_index is None


def test_analyze_rejects_base_field_access_for_unrelated_class() -> None:
    """
    title: Base-qualified access requires the named base in receiver MRO.
    """
    access = astx.BaseFieldAccess(
        astx.Identifier("value"),
        "Other",
        "count",
    )
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[_attribute("count", astx.Int32())],
    )
    other = astx.ClassDefStmt(
        name="Other",
        attributes=[_attribute("count", astx.Int32())],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
    )
    probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(astx.Argument("value", _class_type("Child"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(access),
    )

    with pytest.raises(SemanticError, match="does not inherit from"):
        analyze(make_module("app.main", base, other, child, probe))


def test_analyze_rejects_private_base_field_access_via_explicit_base() -> None:
    """
    title: Explicit base-qualified field access still obeys visibility.
    """
    access = astx.BaseFieldAccess(
        astx.Identifier("self"),
        "Base",
        "secret",
    )
    probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=_method_body(access),
    )
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[
            _attribute(
                "secret",
                astx.Int32(),
                visibility=astx.VisibilityKind.private,
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[probe],
    )

    with pytest.raises(SemanticError, match="is not accessible"):
        analyze(make_module("app.main", base, child))


def test_analyze_static_field_access_uses_visible_static_storage() -> None:
    """
    title: Static field access resolves to analyzed static storage metadata.
    """
    access = astx.StaticFieldAccess("Counter", "instances")
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "instances",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(STATIC_LITERAL_VALUE),
            )
        ],
    )

    analyze(make_module("app.main", counter, _main_returning(access)))

    resolved_class = _semantic(counter).resolved_class
    resolved_access = _semantic(access).resolved_static_class_field_access
    assert resolved_class is not None
    assert resolved_class.layout is not None
    assert resolved_access is not None
    assert resolved_access.class_.name == "Counter"
    assert (
        resolved_access.storage.global_name
        == resolved_class.layout.visible_static_storage[
            "instances"
        ].global_name
    )
    assert isinstance(_semantic(access).resolved_type, astx.Int32)


def test_analyze_inherited_static_field_access_uses_selected_storage() -> None:
    """
    title: Inherited static field access reuses resolved ancestor storage.
    """
    access = astx.StaticFieldAccess("Child", "instances")
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[
            _attribute(
                "instances",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(INHERITED_STATIC_VALUE),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
    )

    analyze(make_module("app.main", base, child, _main_returning(access)))

    resolved_access = _semantic(access).resolved_static_class_field_access
    assert resolved_access is not None
    assert resolved_access.class_.name == "Child"
    assert resolved_access.storage.owner_name == "Base"
    assert resolved_access.member.owner_name == "Base"


def test_analyze_allows_protected_static_field_access_from_subclass() -> None:
    """
    title: Protected static fields stay visible within subclass methods.
    """
    probe_access = astx.StaticFieldAccess("Child", "shared")
    probe = _returning_method(
        "probe",
        probe_access,
        is_static=True,
    )
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[
            _attribute(
                "shared",
                astx.Int32(),
                visibility=astx.VisibilityKind.protected,
                is_static=True,
                value=astx.LiteralInt32(PROTECTED_STATIC_VALUE),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[probe],
    )

    analyze(make_module("app.main", base, child))

    resolved_access = cast(
        SemanticInfo,
        getattr(probe_access, "semantic"),
    ).resolved_static_class_field_access
    assert resolved_access is not None
    assert resolved_access.member.owner_name == "Base"


def test_analyze_rejects_private_static_base_field_access_from_subclass() -> (
    None
):
    """
    title: Subclasses cannot read inherited private static base fields.
    """
    probe_access = astx.StaticFieldAccess("Child", "secret")
    probe = _returning_method(
        "probe",
        probe_access,
        is_static=True,
    )
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[
            _attribute(
                "secret",
                astx.Int32(),
                visibility=astx.VisibilityKind.private,
                is_static=True,
                value=astx.LiteralInt32(PROTECTED_STATIC_VALUE),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[probe],
    )

    with pytest.raises(SemanticError, match="is not accessible"):
        analyze(make_module("app.main", base, child))


def test_analyze_rejects_instance_field_access_through_class_name() -> None:
    """
    title: Class-qualified instance attribute reads require a receiver.
    """
    access = astx.StaticFieldAccess("Counter", "value")
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[_attribute("value", astx.Int32())],
    )

    with pytest.raises(SemanticError, match="requires a receiver"):
        analyze(make_module("app.main", counter, _main_returning(access)))


def test_analyze_rejects_static_field_access_on_method_name() -> None:
    """
    title: Static field access rejects method members by attribute contract.
    """
    identity = _returning_method(
        "identity",
        astx.LiteralInt32(1),
        is_static=True,
    )
    math = astx.ClassDefStmt(name="Math", methods=[identity])
    access = astx.StaticFieldAccess("Math", "identity")

    with pytest.raises(SemanticError, match="is not an attribute"):
        analyze(make_module("app.main", math, _main_returning(access)))


def test_analyze_rejects_static_field_access_through_instance_receiver() -> (
    None
):
    """
    title: Instance field syntax still rejects static attributes on receivers.
    """
    read_instances = astx.FieldAccess(astx.Identifier("self"), "instances")
    read = _returning_method("read", read_instances)
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "instances",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(STATIC_LITERAL_VALUE),
            )
        ],
        methods=[read],
    )

    with pytest.raises(
        SemanticError,
        match="must be accessed through the class",
    ):
        analyze(make_module("app.main", counter))


def test_analyze_static_field_assignment_uses_resolved_storage() -> None:
    """
    title: Static field writes reuse analyzed storage metadata.
    """
    assign = astx.BinaryOp(
        "=",
        astx.StaticFieldAccess("Counter", "instances"),
        astx.LiteralInt32(STATIC_ASSIGNED_VALUE),
    )
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "instances",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(STATIC_LITERAL_VALUE),
            )
        ],
    )

    analyze(make_module("app.main", counter, _main_returning(assign)))

    resolved_class = _semantic(counter).resolved_class
    resolved_access = _semantic(assign.lhs).resolved_static_class_field_access
    resolved_assignment = _semantic(assign).resolved_assignment
    assert resolved_class is not None
    assert resolved_class.layout is not None
    assert resolved_access is not None
    assert resolved_assignment is not None
    assert resolved_access.storage.global_name == (
        resolved_class.layout.visible_static_storage["instances"].global_name
    )
    assert isinstance(_semantic(assign).resolved_type, astx.Int32)


def test_analyze_inherited_static_field_assignment_reuses_base_storage() -> (
    None
):
    """
    title: Inherited static writes resolve to the selected base storage.
    """
    assign = astx.BinaryOp(
        "=",
        astx.StaticFieldAccess("Child", "instances"),
        astx.LiteralInt32(STATIC_ASSIGNED_VALUE),
    )
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[
            _attribute(
                "instances",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(INHERITED_STATIC_VALUE),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
    )

    analyze(make_module("app.main", base, child, _main_returning(assign)))

    resolved_access = _semantic(assign.lhs).resolved_static_class_field_access
    assert resolved_access is not None
    assert resolved_access.storage.owner_name == "Base"
    assert resolved_access.member.owner_name == "Base"


def test_analyze_rejects_constant_static_field_assignment() -> None:
    """
    title: Constant static fields reject assignment after initialization.
    """
    assign = astx.BinaryOp(
        "=",
        astx.StaticFieldAccess("Counter", "limit"),
        astx.LiteralInt32(STATIC_ASSIGNED_VALUE),
    )
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "limit",
                astx.Int32(),
                mutability=astx.MutabilityKind.constant,
                is_static=True,
                value=astx.LiteralInt32(STATIC_LITERAL_VALUE),
            )
        ],
    )

    with pytest.raises(SemanticError, match=r"Counter\.limit"):
        analyze(make_module("app.main", counter, _main_returning(assign)))


def test_analyze_rejects_constant_instance_field_assignment() -> None:
    """
    title: Constant instance fields reject assignment through receivers.
    """
    assign = astx.BinaryOp(
        "=",
        astx.FieldAccess(astx.Identifier("self"), "value"),
        astx.LiteralInt32(INSTANCE_ASSIGNED_VALUE),
    )
    write = _returning_method("write", assign)
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "value",
                astx.Int32(),
                mutability=astx.MutabilityKind.constant,
                value=astx.LiteralInt32(STATIC_LITERAL_VALUE),
            )
        ],
        methods=[write],
    )

    with pytest.raises(SemanticError, match=r"Counter\.value"):
        analyze(make_module("app.main", counter))


def test_rejects_constant_base_field_assignment_via_explicit_base() -> None:
    """
    title: Constant base-qualified fields reject explicit writes.
    """
    assign = astx.BinaryOp(
        "=",
        astx.BaseFieldAccess(astx.Identifier("self"), "Base", "value"),
        astx.LiteralInt32(BASE_ASSIGNED_VALUE),
    )
    write = _returning_method("write", assign)
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[
            _attribute(
                "value",
                astx.Int32(),
                mutability=astx.MutabilityKind.constant,
                value=astx.LiteralInt32(BASE_FIELD_VALUE),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[write],
    )

    with pytest.raises(SemanticError, match=r"Base\.value"):
        analyze(make_module("app.main", base, child))


def test_analyze_rejects_constant_static_field_increment() -> None:
    """
    title: Constant static fields reject unary mutation.
    """
    increment = astx.UnaryOp(
        op_code="++",
        operand=astx.StaticFieldAccess("Counter", "limit"),
    )
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "limit",
                astx.Int32(),
                mutability=astx.MutabilityKind.constant,
                is_static=True,
                value=astx.LiteralInt32(STATIC_LITERAL_VALUE),
            )
        ],
    )

    with pytest.raises(SemanticError, match=r"Counter\.limit"):
        analyze(make_module("app.main", counter, _main_returning(increment)))


def test_analyze_allows_static_method_mutating_explicit_receiver_field() -> (
    None
):
    """
    title: Static methods may mutate instance fields via explicit receivers.
    """
    assign = astx.BinaryOp(
        "=",
        astx.FieldAccess(astx.Identifier("value"), "count"),
        astx.LiteralInt32(INSTANCE_ASSIGNED_VALUE),
    )
    write = _returning_method(
        "write",
        assign,
        astx.Argument("value", _class_type("Counter")),
        is_static=True,
    )
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[_attribute("count", astx.Int32())],
        methods=[write],
    )

    analyze(make_module("app.main", counter))

    resolved_access = _semantic(assign.lhs).resolved_class_field_access
    resolved_assignment = _semantic(assign).resolved_assignment
    assert resolved_access is not None
    assert resolved_assignment is not None
    assert resolved_access.member.owner_name == "Counter"


def test_analyze_resolves_exact_method_overloads() -> None:
    """
    title: Method calls choose one exact overload from inherited groups.
    """
    base = astx.ClassDefStmt(
        name="Base",
        methods=[
            _returning_method(
                "render",
                astx.LiteralInt32(1),
                astx.Argument("value", astx.Int32()),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[
            _returning_method(
                "render",
                astx.LiteralInt32(2),
                astx.Argument("value", astx.Float64()),
            )
        ],
    )
    render_int = astx.MethodCall(
        astx.Identifier("value"),
        "render",
        [astx.LiteralInt32(4)],
    )
    render_float = astx.MethodCall(
        astx.Identifier("value"),
        "render",
        [astx.LiteralFloat64(4.0)],
    )
    probe_int = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe_int",
            args=astx.Arguments(astx.Argument("value", _class_type("Child"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(render_int),
    )
    probe_float = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe_float",
            args=astx.Arguments(astx.Argument("value", _class_type("Child"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(render_float),
    )

    analyze(make_module("app.main", base, child, probe_int, probe_float))

    resolved_class = _semantic(child).resolved_class
    assert resolved_class is not None
    assert len(resolved_class.method_groups["render"]) == RENDER_OVERLOAD_COUNT

    int_call = _semantic(render_int).resolved_method_call
    float_call = _semantic(render_float).resolved_method_call
    assert int_call is not None
    assert float_call is not None
    assert int_call.member.owner_name == "Base"
    assert float_call.member.owner_name == "Child"
    assert int_call.overload_key != float_call.overload_key


def test_analyze_rejects_conversion_ranked_method_overloads() -> None:
    """
    title: Overload selection requires one exact explicit argument match.
    """
    number = astx.ClassDefStmt(
        name="Number",
        methods=[
            _returning_method(
                "render",
                astx.LiteralInt32(1),
                astx.Argument("value", astx.Int32()),
            ),
            _returning_method(
                "render",
                astx.LiteralInt32(2),
                astx.Argument("value", astx.Float64()),
            ),
        ],
    )
    call = astx.MethodCall(
        astx.Identifier("value"),
        "render",
        [astx.LiteralInt8(1)],
    )
    probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(astx.Argument("value", _class_type("Number"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(call),
    )

    with pytest.raises(SemanticError, match="no exact overload"):
        analyze(make_module("app.main", number, probe))


def test_analyze_supports_base_typed_method_polymorphism() -> None:
    """
    title: Base-typed receivers keep one shared override dispatch slot.
    """
    base_method = _returning_method("area", astx.LiteralInt32(1))
    child_method = _returning_method("area", astx.LiteralInt32(2))
    base = astx.ClassDefStmt(name="Base", methods=[base_method])
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[child_method],
    )
    area_call = astx.MethodCall(astx.Identifier("shape"), "area", [])
    measure = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="measure",
            args=astx.Arguments(astx.Argument("shape", _class_type("Base"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(area_call),
    )
    wrap = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="wrap",
            args=astx.Arguments(astx.Argument("shape", _class_type("Child"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(
            astx.FunctionCall("measure", [astx.Identifier("shape")])
        ),
    )

    analyze(make_module("app.main", base, child, measure, wrap))

    resolved_base = _semantic(base).resolved_class
    resolved_child = _semantic(child).resolved_class
    resolved_call = _semantic(area_call).resolved_method_call

    assert resolved_base is not None
    assert resolved_child is not None
    assert resolved_call is not None
    assert resolved_call.member.owner_name == "Base"
    assert resolved_call.dispatch_kind is MethodDispatchKind.INDIRECT
    assert (
        resolved_call.slot_index
        == resolved_base.member_table["area"].dispatch_slot
    )
    assert (
        resolved_call.slot_index
        == resolved_child.member_table["area"].dispatch_slot
    )


def test_analyze_accepts_derived_returns_for_base_result_types() -> None:
    """
    title: Derived class values are valid returns where a base is expected.
    """
    base = astx.ClassDefStmt(name="Base")
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
    )
    upcast = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="upcast",
            args=astx.Arguments(astx.Argument("value", _class_type("Child"))),
            return_type=_class_type("Base"),
        ),
        body=_method_body(astx.Identifier("value")),
    )

    analyze(make_module("app.main", base, child, upcast))


def test_analyze_rejects_private_method_access_outside_declaring_class() -> (
    None
):
    """
    title: Private methods are inaccessible from non-member contexts.
    """
    hidden = _returning_method(
        "hidden",
        astx.LiteralInt32(1),
        visibility=astx.VisibilityKind.private,
    )
    secret = astx.ClassDefStmt(name="Secret", methods=[hidden])
    call = astx.MethodCall(astx.Identifier("value"), "hidden", [])
    probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(astx.Argument("value", _class_type("Secret"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(call),
    )

    with pytest.raises(SemanticError, match="is not accessible"):
        analyze(make_module("app.main", secret, probe))


def test_analyze_allows_protected_access_from_sibling_subclass() -> None:
    """
    title: Protected members are accessible from sibling subclasses.
    """
    reveal = _returning_method(
        "reveal",
        astx.LiteralInt32(1),
        visibility=astx.VisibilityKind.protected,
    )
    base = astx.ClassDefStmt(name="Base", methods=[reveal])
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
    )
    sibling_call = astx.MethodCall(astx.Identifier("value"), "reveal", [])
    sibling = astx.ClassDefStmt(
        name="Sibling",
        bases=[_class_type("Base")],
        methods=[
            astx.FunctionDef(
                prototype=astx.FunctionPrototype(
                    name="probe",
                    args=astx.Arguments(
                        astx.Argument("value", _class_type("Child"))
                    ),
                    return_type=astx.Int32(),
                ),
                body=_method_body(sibling_call),
            )
        ],
    )

    analyze(make_module("app.main", base, child, sibling))

    resolved_call = _semantic(sibling_call).resolved_method_call
    assert resolved_call is not None
    assert resolved_call.member.owner_name == "Base"


def test_analyze_rejects_protected_access_outside_subclass_context() -> None:
    """
    title: Protected members stay inaccessible from non-subclass scopes.
    """
    reveal = _returning_method(
        "reveal",
        astx.LiteralInt32(1),
        visibility=astx.VisibilityKind.protected,
    )
    base = astx.ClassDefStmt(name="Base", methods=[reveal])
    call = astx.MethodCall(astx.Identifier("value"), "reveal", [])
    probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(astx.Argument("value", _class_type("Base"))),
            return_type=astx.Int32(),
        ),
        body=_method_body(call),
    )

    with pytest.raises(SemanticError, match="is not accessible"):
        analyze(make_module("app.main", base, probe))


def test_analyze_allows_private_static_base_method_on_derived_class() -> None:
    """
    title: Declaring classes may call private static members on child names.
    """
    hidden = _returning_method(
        "hidden",
        astx.LiteralInt32(1),
        visibility=astx.VisibilityKind.private,
        is_static=True,
    )
    probe_call = astx.StaticMethodCall("Child", "hidden", [])
    probe = _returning_method(
        "probe",
        probe_call,
        is_static=True,
    )
    base = astx.ClassDefStmt(name="Base", methods=[hidden, probe])
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
    )

    analyze(make_module("app.main", base, child))

    resolved_call = _semantic(probe_call).resolved_method_call
    assert resolved_call is not None
    assert resolved_call.member.owner_name == "Base"
    assert resolved_call.dispatch_kind is MethodDispatchKind.DIRECT


def test_analyze_rejects_private_base_method_access_from_subclass() -> None:
    """
    title: Subclasses cannot call inherited private base methods.
    """
    hidden = _returning_method(
        "hidden",
        astx.LiteralInt32(1),
        visibility=astx.VisibilityKind.private,
    )
    child_call = astx.MethodCall(astx.Identifier("self"), "hidden", [])
    child_probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=_method_body(child_call),
    )
    base = astx.ClassDefStmt(name="Base", methods=[hidden])
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[child_probe],
    )

    with pytest.raises(SemanticError, match="is not accessible"):
        analyze(make_module("app.main", base, child))


def test_analyze_rejects_private_base_field_access_from_subclass() -> None:
    """
    title: Subclasses cannot read inherited private base fields.
    """
    secret = astx.VariableDeclaration(
        name="secret",
        type_=astx.Int32(),
        mutability=astx.MutabilityKind.mutable,
        visibility=astx.VisibilityKind.private,
    )
    read_secret = astx.FieldAccess(astx.Identifier("self"), "secret")
    child_probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=_method_body(read_secret),
    )
    base = astx.ClassDefStmt(name="Base", attributes=[secret])
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[child_probe],
    )

    with pytest.raises(SemanticError, match="is not accessible"):
        analyze(make_module("app.main", base, child))

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


def _attribute(name: str, type_: astx.DataType) -> astx.VariableDeclaration:
    """
    title: Build one instance attribute declaration.
    parameters:
      name:
        type: str
      type_:
        type: astx.DataType
    returns:
      type: astx.VariableDeclaration
    """
    return astx.VariableDeclaration(
        name=name,
        type_=type_,
        mutability=astx.MutabilityKind.mutable,
    )


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

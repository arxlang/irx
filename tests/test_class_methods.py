"""
title: Stable class method lowering tests.
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.analysis import analyze
from irx.analysis.module_symbols import (
    mangle_class_dispatch_name,
    mangle_class_method_name,
)
from irx.analysis.resolved_nodes import SemanticInfo
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder

from tests.conftest import assert_ir_parses, make_module

SINGLE_DISPATCH_ENTRY_COUNT = 1
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


def _mangled_method_name(
    module: astx.Module,
    class_node: astx.ClassDefStmt,
    method_name: str,
) -> str:
    """
    title: Return the overload-aware LLVM symbol for one class method.
    parameters:
      module:
        type: astx.Module
      class_node:
        type: astx.ClassDefStmt
      method_name:
        type: str
    returns:
      type: str
    """
    analyze(module)
    resolved = _semantic(class_node).resolved_class
    assert resolved is not None
    member = resolved.declared_member_table[method_name]
    assert member.signature_key is not None
    return mangle_class_method_name(
        module.name,
        class_node.name,
        method_name,
        member.signature_key,
    )


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
    )
    if is_static:
        prototype.is_static = True
    body = astx.Block()
    body.append(astx.FunctionReturn(return_value))
    return astx.FunctionDef(prototype=prototype, body=body)


def _single_return_body(value: astx.AST) -> astx.Block:
    """
    title: Build one single-return block.
    parameters:
      value:
        type: astx.AST
    returns:
      type: astx.Block
    """
    body = astx.Block()
    body.append(astx.FunctionReturn(value))
    return body


def _main_int32(*body_nodes: astx.AST) -> astx.FunctionDef:
    """
    title: Build a simple int32-returning main function.
    parameters:
      body_nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.FunctionDef
    """
    body = astx.Block()
    for node in body_nodes:
        body.append(node)
    if not any(isinstance(node, astx.FunctionReturn) for node in body_nodes):
        body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    return astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=body,
    )


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_instance_method_definition_emits_hidden_receiver_and_dispatch_table(
    builder_class: type[Builder],
) -> None:
    """
    title: Instance methods lower to functions plus one class dispatch table.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    area = _returning_method("area", astx.LiteralInt32(1))
    shape = astx.ClassDefStmt(name="Shape", methods=[area])
    module = make_module("main", shape, _main_int32())

    ir_text = builder.translate(module)
    method_name = _mangled_method_name(module, shape, "area")
    dispatch_name = mangle_class_dispatch_name("main", "Shape")

    assert f'define i32 @"{method_name}"(%"main__Shape"* %"self")' in ir_text
    assert (
        f'@"{dispatch_name}" = internal constant '
        f"[{SINGLE_DISPATCH_ENTRY_COUNT} x i8*] "
        f'[i8* bitcast (i32 (%"main__Shape"*)* @"{method_name}" to i8*)]'
    ) in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_instance_method_call_uses_dispatch_lookup(
    builder_class: type[Builder],
) -> None:
    """
    title: Instance method calls lower through the dispatch-table slot.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    shape = astx.ClassDefStmt(
        name="Shape",
        methods=[_returning_method("area", astx.LiteralInt32(1))],
    )
    measure = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="measure",
            args=astx.Arguments(astx.Argument("shape", _class_type("Shape"))),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.MethodCall(astx.Identifier("shape"), "area", [])
        ),
    )
    module = make_module("main", shape, measure, _main_int32())

    ir_text = builder.translate(module)

    assert "area_dispatch_addr" in ir_text
    assert "area_dispatch_ptr" in ir_text
    assert "area_slot" in ir_text
    assert "area_callee" in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_static_method_call_lowers_to_direct_call_without_receiver(
    builder_class: type[Builder],
) -> None:
    """
    title: Static method calls lower directly without a hidden receiver.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    identity = _returning_method(
        "identity",
        astx.Identifier("value"),
        astx.Argument("value", astx.Int32()),
        is_static=True,
    )
    math = astx.ClassDefStmt(name="Math", methods=[identity])
    module = make_module(
        "main",
        math,
        _main_int32(
            astx.FunctionReturn(
                astx.StaticMethodCall(
                    "Math",
                    "identity",
                    [astx.LiteralInt32(4)],
                )
            )
        ),
    )

    ir_text = builder.translate(module)
    method_name = _mangled_method_name(module, math, "identity")

    assert f'define i32 @"{method_name}"(i32 %"value")' in ir_text
    assert f'call i32 @"{method_name}"(i32 4)' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_base_typed_method_dispatch_uses_upcast_and_shared_slot(
    builder_class: type[Builder],
) -> None:
    """
    title: Base-typed calls bitcast derived values and dispatch indirectly.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    base = astx.ClassDefStmt(
        name="Base",
        methods=[_returning_method("area", astx.LiteralInt32(1))],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[_returning_method("area", astx.LiteralInt32(2))],
    )
    measure = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="measure",
            args=astx.Arguments(astx.Argument("shape", _class_type("Base"))),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.MethodCall(astx.Identifier("shape"), "area", [])
        ),
    )
    wrap = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="wrap",
            args=astx.Arguments(astx.Argument("shape", _class_type("Child"))),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.FunctionCall("measure", [astx.Identifier("shape")])
        ),
    )
    module = make_module("poly", base, child, measure, wrap, _main_int32())

    ir_text = builder.translate(module)

    assert 'bitcast %"poly__Child"*' in ir_text
    assert 'to %"poly__Base"*' in ir_text
    assert "area_slot" in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_instance_method_body_reads_class_field_slot(
    builder_class: type[Builder],
) -> None:
    """
    title: Method bodies use the flattened class layout for self field reads.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    read = _returning_method(
        "read",
        astx.FieldAccess(astx.Identifier("self"), "value"),
    )
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[_attribute("value", astx.Int32())],
        methods=[read],
    )
    module = make_module("main", counter, _main_int32())

    ir_text = builder.translate(module)

    assert '"value_addr" = getelementptr inbounds %"main__Counter"' in ir_text
    assert f"i32 0, i32 {FIRST_INSTANCE_STORAGE_INDEX}" in ir_text
    assert_ir_parses(ir_text)

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
    mangle_class_static_name,
)
from irx.analysis.resolved_nodes import SemanticInfo
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder

from tests.conftest import (
    assert_ir_parses,
    assert_jit_int_main_result,
    make_module,
)

SINGLE_DISPATCH_ENTRY_COUNT = 1
CLASS_HEADER_SLOT_COUNT = 2
FIRST_INSTANCE_STORAGE_INDEX = CLASS_HEADER_SLOT_COUNT
BASE_METHOD_RESULT = 21
CHILD_METHOD_RESULT = 34
BASE_FIELD_VALUE = 13
STATIC_LITERAL_VALUE = 7
STATIC_ASSIGNED_VALUE = 19
INSTANCE_ASSIGNED_VALUE = 23
BASE_ASSIGNED_VALUE = 29
INCREMENTED_STATIC_VALUE = 8
RUNTIME_BASE_OVERLOAD_RESULT = 41
RUNTIME_CHILD_OVERLOAD_RESULT = 52
RUNTIME_RECEIVER_OVERLOAD_RESULT = 63


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


def _attribute(
    name: str,
    type_: astx.DataType,
    *,
    mutability: astx.MutabilityKind = astx.MutabilityKind.mutable,
    is_static: bool = False,
    value: astx.AST | None = None,
) -> astx.VariableDeclaration:
    """
    title: Build one instance attribute declaration.
    parameters:
      name:
        type: str
      type_:
        type: astx.DataType
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
def test_runtime_multimethod_dispatch_uses_dynamic_argument_class(
    builder_class: type[Builder],
) -> None:
    """
    title: Base-typed arguments dispatch to the most specific runtime overload.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    value_base = astx.ClassDefStmt(name="Value")
    value_child = astx.ClassDefStmt(
        name="ChildValue",
        bases=[_class_type("Value")],
    )
    renderer = astx.ClassDefStmt(
        name="Renderer",
        methods=[
            _returning_method(
                "render",
                astx.LiteralInt32(RUNTIME_BASE_OVERLOAD_RESULT),
                astx.Argument("value", _class_type("Value")),
            ),
            _returning_method(
                "render",
                astx.LiteralInt32(RUNTIME_CHILD_OVERLOAD_RESULT),
                astx.Argument("value", _class_type("ChildValue")),
            ),
        ],
    )
    dispatch = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="dispatch",
            args=astx.Arguments(astx.Argument("value", _class_type("Value"))),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.MethodCall(
                astx.ClassConstruct("Renderer"),
                "render",
                [astx.Identifier("value")],
            )
        ),
    )
    module = make_module(
        "main",
        value_base,
        value_child,
        renderer,
        dispatch,
        _main_int32(
            astx.FunctionReturn(
                astx.FunctionCall(
                    "dispatch",
                    [astx.ClassConstruct("ChildValue")],
                )
            )
        ),
    )

    ir_text = builder.translate(module)

    assert "multimethod" in ir_text
    assert_ir_parses(ir_text)
    assert_jit_int_main_result(
        builder,
        module,
        RUNTIME_CHILD_OVERLOAD_RESULT,
    )


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_runtime_multimethod_dispatch_uses_dynamic_receiver_and_arg(
    builder_class: type[Builder],
) -> None:
    """
    title: Receiver branches can refine overload choice at runtime too.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    value_base = astx.ClassDefStmt(name="Value")
    value_child = astx.ClassDefStmt(
        name="ChildValue",
        bases=[_class_type("Value")],
    )
    base_renderer = astx.ClassDefStmt(
        name="BaseRenderer",
        methods=[
            _returning_method(
                "render",
                astx.LiteralInt32(RUNTIME_BASE_OVERLOAD_RESULT),
                astx.Argument("value", _class_type("Value")),
            )
        ],
    )
    child_renderer = astx.ClassDefStmt(
        name="ChildRenderer",
        bases=[_class_type("BaseRenderer")],
        methods=[
            _returning_method(
                "render",
                astx.LiteralInt32(RUNTIME_RECEIVER_OVERLOAD_RESULT),
                astx.Argument("value", _class_type("ChildValue")),
            )
        ],
    )
    measure = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="measure",
            args=astx.Arguments(
                astx.Argument("renderer", _class_type("BaseRenderer")),
                astx.Argument("value", _class_type("Value")),
            ),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.MethodCall(
                astx.Identifier("renderer"),
                "render",
                [astx.Identifier("value")],
            )
        ),
    )
    module = make_module(
        "main",
        value_base,
        value_child,
        base_renderer,
        child_renderer,
        measure,
        _main_int32(
            astx.FunctionReturn(
                astx.FunctionCall(
                    "measure",
                    [
                        astx.ClassConstruct("ChildRenderer"),
                        astx.ClassConstruct("ChildValue"),
                    ],
                )
            )
        ),
    )

    ir_text = builder.translate(module)

    assert "receiver_case_" in ir_text
    assert_ir_parses(ir_text)
    assert_jit_int_main_result(
        builder,
        module,
        RUNTIME_RECEIVER_OVERLOAD_RESULT,
    )


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_runtime_multimethod_dispatch_uses_dynamic_static_args(
    builder_class: type[Builder],
) -> None:
    """
    title: Static method overload families dispatch on runtime class args.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    value_base = astx.ClassDefStmt(name="Value")
    value_child = astx.ClassDefStmt(
        name="ChildValue",
        bases=[_class_type("Value")],
    )
    renderer = astx.ClassDefStmt(
        name="Renderer",
        methods=[
            _returning_method(
                "render",
                astx.LiteralInt32(RUNTIME_BASE_OVERLOAD_RESULT),
                astx.Argument("value", _class_type("Value")),
                is_static=True,
            ),
            _returning_method(
                "render",
                astx.LiteralInt32(RUNTIME_CHILD_OVERLOAD_RESULT),
                astx.Argument("value", _class_type("ChildValue")),
                is_static=True,
            ),
        ],
    )
    dispatch = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="dispatch",
            args=astx.Arguments(astx.Argument("value", _class_type("Value"))),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.StaticMethodCall(
                "Renderer",
                "render",
                [astx.Identifier("value")],
            )
        ),
    )
    module = make_module(
        "main",
        value_base,
        value_child,
        renderer,
        dispatch,
        _main_int32(
            astx.FunctionReturn(
                astx.FunctionCall(
                    "dispatch",
                    [astx.ClassConstruct("ChildValue")],
                )
            )
        ),
    )

    ir_text = builder.translate(module)

    assert "multimethod" in ir_text
    assert_ir_parses(ir_text)
    assert_jit_int_main_result(
        builder,
        module,
        RUNTIME_CHILD_OVERLOAD_RESULT,
    )


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


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_inherited_base_method_call_on_derived_receiver_uses_upcast(
    builder_class: type[Builder],
) -> None:
    """
    title: Inherited base-method calls upcast derived receivers.
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
    )
    probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(astx.Argument("value", _class_type("Child"))),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.MethodCall(astx.Identifier("value"), "area", [])
        ),
    )
    module = make_module("main", base, child, probe, _main_int32())

    ir_text = builder.translate(module)

    assert '%"classcast" = bitcast %"main__Child"*' in ir_text
    assert 'to %"main__Base"*' in ir_text
    assert "area_callee" in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_base_method_call_lowers_directly_to_selected_base(
    builder_class: type[Builder],
) -> None:
    """
    title: Base-qualified calls bypass override dispatch and call the base.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    base = astx.ClassDefStmt(
        name="Base",
        methods=[
            _returning_method("area", astx.LiteralInt32(BASE_METHOD_RESULT))
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[
            _returning_method("area", astx.LiteralInt32(CHILD_METHOD_RESULT))
        ],
    )
    main_fn = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.BaseMethodCall(
                astx.ClassConstruct("Child"),
                "Base",
                "area",
                [],
            )
        ),
    )
    module = make_module("main", base, child, main_fn)

    ir_text = builder.translate(module)
    base_method_name = _mangled_method_name(module, base, "area")

    assert f'call i32 @"{base_method_name}"' in ir_text
    assert_jit_int_main_result(builder, module, BASE_METHOD_RESULT)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_base_field_access_reads_inherited_storage_from_receiver(
    builder_class: type[Builder],
) -> None:
    """
    title: Base-qualified field reads use the receiver's flattened storage.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[
            _attribute(
                "value",
                astx.Int32(),
                value=astx.LiteralInt32(BASE_FIELD_VALUE),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
    )
    main_fn = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.BaseFieldAccess(
                astx.ClassConstruct("Child"),
                "Base",
                "value",
            )
        ),
    )
    module = make_module("main", base, child, main_fn)

    ir_text = builder.translate(module)

    assert '"value_addr" = getelementptr inbounds %"main__Child"' in ir_text
    assert_jit_int_main_result(builder, module, BASE_FIELD_VALUE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_static_field_assignment_updates_global_and_reads_back(
    builder_class: type[Builder],
) -> None:
    """
    title: Static field assignment stores through the emitted global.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "instances",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(BASE_FIELD_VALUE),
            )
        ],
    )
    static_name = mangle_class_static_name("main", "Counter", "instances")
    main_fn = _main_int32(
        astx.BinaryOp(
            "=",
            astx.StaticFieldAccess("Counter", "instances"),
            astx.LiteralInt32(STATIC_ASSIGNED_VALUE),
        ),
        astx.FunctionReturn(astx.StaticFieldAccess("Counter", "instances")),
    )
    module = make_module("main", counter, main_fn)

    ir_text = builder.translate(module)

    assert (
        f'store i32 {STATIC_ASSIGNED_VALUE}, i32* @"{static_name}"' in ir_text
    )
    assert_jit_int_main_result(builder, module, STATIC_ASSIGNED_VALUE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_inherited_static_field_assignment_reuses_base_global(
    builder_class: type[Builder],
) -> None:
    """
    title: Derived static writes store through inherited base globals.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[
            _attribute(
                "instances",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(BASE_FIELD_VALUE),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
    )
    base_name = mangle_class_static_name("main", "Base", "instances")
    child_name = mangle_class_static_name("main", "Child", "instances")
    main_fn = _main_int32(
        astx.BinaryOp(
            "=",
            astx.StaticFieldAccess("Child", "instances"),
            astx.LiteralInt32(STATIC_ASSIGNED_VALUE),
        ),
        astx.FunctionReturn(astx.StaticFieldAccess("Child", "instances")),
    )
    module = make_module("main", base, child, main_fn)

    ir_text = builder.translate(module)

    assert child_name not in ir_text
    assert f'store i32 {STATIC_ASSIGNED_VALUE}, i32* @"{base_name}"' in ir_text
    assert_jit_int_main_result(builder, module, STATIC_ASSIGNED_VALUE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_instance_field_assignment_updates_value(
    builder_class: type[Builder],
) -> None:
    """
    title: Instance field assignment stores through the receiver layout slot.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    body = astx.Block()
    body.append(
        astx.BinaryOp(
            "=",
            astx.FieldAccess(astx.Identifier("self"), "value"),
            astx.LiteralInt32(INSTANCE_ASSIGNED_VALUE),
        )
    )
    body.append(
        astx.FunctionReturn(astx.FieldAccess(astx.Identifier("self"), "value"))
    )
    write = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="write",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=body,
    )
    counter = astx.ClassDefStmt(
        name="Counter",
        attributes=[_attribute("value", astx.Int32())],
        methods=[write],
    )
    main_fn = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.MethodCall(astx.ClassConstruct("Counter"), "write", [])
        ),
    )
    module = make_module("main", counter, main_fn)

    ir_text = builder.translate(module)

    assert (
        f'store i32 {INSTANCE_ASSIGNED_VALUE}, i32* %"value_addr"' in ir_text
    )
    assert_jit_int_main_result(builder, module, INSTANCE_ASSIGNED_VALUE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_base_field_assignment_updates_shared_receiver_storage(
    builder_class: type[Builder],
) -> None:
    """
    title: Base-qualified writes update the receiver's flattened storage.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    body = astx.Block()
    body.append(
        astx.BinaryOp(
            "=",
            astx.BaseFieldAccess(astx.Identifier("self"), "Base", "value"),
            astx.LiteralInt32(BASE_ASSIGNED_VALUE),
        )
    )
    body.append(
        astx.FunctionReturn(astx.FieldAccess(astx.Identifier("self"), "value"))
    )
    write = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="write",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=body,
    )
    base = astx.ClassDefStmt(
        name="Base",
        attributes=[_attribute("value", astx.Int32())],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
        methods=[write],
    )
    main_fn = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.MethodCall(astx.ClassConstruct("Child"), "write", [])
        ),
    )
    module = make_module("main", base, child, main_fn)

    ir_text = builder.translate(module)

    assert f'store i32 {BASE_ASSIGNED_VALUE}, i32* %"value_addr"' in ir_text
    assert_jit_int_main_result(builder, module, BASE_ASSIGNED_VALUE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_static_field_increment_updates_global(
    builder_class: type[Builder],
) -> None:
    """
    title: Unary mutation stores the updated value back to static globals.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
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
    main_fn = _main_int32(
        astx.UnaryOp(
            op_code="++",
            operand=astx.StaticFieldAccess("Counter", "instances"),
        ),
        astx.FunctionReturn(astx.StaticFieldAccess("Counter", "instances")),
    )
    module = make_module("main", counter, main_fn)

    ir_text = builder.translate(module)

    assert "add i32" in ir_text
    assert_jit_int_main_result(builder, module, INCREMENTED_STATIC_VALUE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_default_class_construction_initializes_fields_and_dispatch(
    builder_class: type[Builder],
) -> None:
    """
    title: Default construction allocates, initializes, and dispatches.
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
        attributes=[
            _attribute("value", astx.Int32(), value=astx.LiteralInt32(7))
        ],
        methods=[read],
    )
    main_fn = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=_single_return_body(
            astx.MethodCall(astx.ClassConstruct("Counter"), "read", [])
        ),
    )
    module = make_module("main", counter, main_fn)

    assert_jit_int_main_result(builder, module, 7)

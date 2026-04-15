"""
title: Stable class layout and storage lowering tests.
"""

from __future__ import annotations

import ctypes
import re

import pytest

from irx import astx
from irx.analysis.module_symbols import (
    mangle_class_descriptor_name,
    mangle_class_name,
    mangle_class_static_name,
)
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder
from llvmlite import binding as llvm

from tests.conftest import (
    assert_ir_parses,
    assert_jit_int_main_result,
    make_module,
)

GLOBAL_ADDRESS_MISSING = 0
STATIC_LITERAL_VALUE = 7
STATIC_DEFAULT_VALUE = 0
INHERITED_STATIC_VALUE = 9


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
    title: Build one class attribute declaration.
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


def _mutable_var(
    name: str,
    type_: astx.DataType,
    value: astx.AST | astx.Undefined = astx.Undefined(),
) -> astx.VariableDeclaration:
    """
    title: Build one mutable local variable declaration.
    parameters:
      name:
        type: str
      type_:
        type: astx.DataType
      value:
        type: astx.AST | astx.Undefined
    returns:
      type: astx.VariableDeclaration
    """
    return astx.VariableDeclaration(
        name=name,
        type_=type_,
        mutability=astx.MutabilityKind.mutable,
        value=value,
    )


def _main_int32(*body_nodes: astx.AST) -> astx.FunctionDef:
    """
    title: Build a small int32-returning main function.
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


def _jit_int32_globals(
    ir_text: str,
    *global_names: str,
) -> dict[str, int]:
    """
    title: Read Int32 class-static globals from one JIT-compiled module.
    parameters:
      ir_text:
        type: str
      global_names:
        type: str
        variadic: positional
    returns:
      type: dict[str, int]
    """
    llvm_module = llvm.parse_assembly(ir_text)
    llvm_module.verify()

    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_module = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_module, target_machine)

    engine.add_module(llvm_module)
    engine.finalize_object()
    engine.run_static_constructors()

    values: dict[str, int] = {}
    for global_name in global_names:
        address = engine.get_global_value_address(global_name)
        assert address != GLOBAL_ADDRESS_MISSING, (
            f"Expected JIT-visible global '{global_name}'"
        )
        values[global_name] = int(ctypes.c_int32.from_address(address).value)
    return values


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_class_definition_emits_header_and_instance_layout(
    builder_class: type[Builder],
) -> None:
    """
    title: Class definitions lower to pointer-oriented object structs.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    node = astx.ClassDefStmt(
        name="Vector",
        attributes=[
            _attribute("x", astx.Int32()),
            _attribute("ready", astx.Boolean()),
        ],
    )
    module = make_module("main", node, _main_int32())

    ir_text = builder.translate(module)
    llvm_name = mangle_class_name("main", "Vector")

    assert f'%"{llvm_name}" = type {{i8*, i8*, i32, i1}}' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_class_construction_initializes_descriptor_header(
    builder_class: type[Builder],
) -> None:
    """
    title: Class construction stores the per-class descriptor pointer.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    node = astx.ClassDefStmt(name="Vector")
    main_fn = _main_int32(
        _mutable_var(
            "value",
            _class_type("Vector"),
            astx.ClassConstruct("Vector"),
        ),
    )
    module = make_module("main", node, main_fn)

    ir_text = builder.translate(module)
    descriptor_name = mangle_class_descriptor_name("main", "Vector")

    assert f'@"{descriptor_name}" = internal constant i8 0' in ir_text
    assert (
        f'store i8* @"{descriptor_name}", i8** %"Vector_descriptor_addr"'
    ) in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_class_layout_flattens_canonical_shared_ancestor_storage(
    builder_class: type[Builder],
) -> None:
    """
    title: Shared ancestors appear once and bases precede derived fields.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
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
    module = make_module("main", root, left, right, child, _main_int32())

    ir_text = builder.translate(module)
    llvm_name = mangle_class_name("main", "Child")

    assert re.search(
        rf'%"{re.escape(llvm_name)}(?:\.\d+)?" = type '
        r"\{i8\*, i8\*, i32, i1, double, i8\}",
        ir_text,
    )
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_class_values_lower_as_pointers_and_static_members_as_globals(
    builder_class: type[Builder],
) -> None:
    """
    title: Class locals use pointer storage and statics emit globals.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    node = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute("value", astx.Int32()),
            _attribute(
                "instances",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(7),
            ),
            _attribute(
                "limit",
                astx.Int32(),
                mutability=astx.MutabilityKind.constant,
                is_static=True,
                value=astx.LiteralInt32(99),
            ),
        ],
    )
    main_fn = _main_int32(_mutable_var("counter", _class_type("Counter")))
    module = make_module("main", node, main_fn)

    ir_text = builder.translate(module)

    assert 'store %"main__Counter"* null' in ir_text
    assert (
        f'@"{mangle_class_static_name("main", "Counter", "instances")}" '
        "= internal global i32 7"
    ) in ir_text
    assert (
        f'@"{mangle_class_static_name("main", "Counter", "limit")}" '
        "= internal constant i32 99"
    ) in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_ir_defined_functions_use_pointer_abi_for_class_values(
    builder_class: type[Builder],
) -> None:
    """
    title: >-
      IR-defined function signatures pass and return class values by pointer.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    widget = astx.ClassDefStmt(name="Widget")
    identity_body = astx.Block()
    identity_body.append(astx.FunctionReturn(astx.Identifier("value")))
    identity = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="identity",
            args=astx.Arguments(
                astx.Argument("value", _class_type("Widget")),
            ),
            return_type=_class_type("Widget"),
        ),
        body=identity_body,
    )
    module = make_module("main", widget, identity, _main_int32())

    ir_text = builder.translate(module)

    assert (
        'define %"main__Widget"* @"main__identity"(%"main__Widget"* %"value")'
    ) in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_class_static_globals_keep_literal_and_default_values(
    builder_class: type[Builder],
) -> None:
    """
    title: Static class globals preserve literal and default init values.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    node = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "instances",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(7),
            ),
            _attribute(
                "pending",
                astx.Int32(),
                is_static=True,
            ),
        ],
    )
    module = make_module("main", node, _main_int32())

    ir_text = builder.translate(module)
    literal_name = mangle_class_static_name(
        "main",
        "Counter",
        "instances",
    )
    default_name = mangle_class_static_name(
        "main",
        "Counter",
        "pending",
    )

    assert f'@"{literal_name}" = internal global i32 7' in ir_text
    assert f'@"{default_name}" = internal global i32 0' in ir_text
    assert _jit_int32_globals(
        ir_text,
        literal_name,
        default_name,
    ) == {
        literal_name: 7,
        default_name: 0,
    }


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_static_field_access_reads_class_global_storage(
    builder_class: type[Builder],
) -> None:
    """
    title: Static field reads lower to loads from analyzed class globals.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    node = astx.ClassDefStmt(
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
        astx.FunctionReturn(astx.StaticFieldAccess("Counter", "instances"))
    )
    module = make_module("main", node, main_fn)

    ir_text = builder.translate(module)
    global_name = mangle_class_static_name(
        "main",
        "Counter",
        "instances",
    )

    assert f'load i32, i32* @"{global_name}"' in ir_text
    assert_ir_parses(ir_text)
    assert_jit_int_main_result(builder, module, STATIC_LITERAL_VALUE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_static_field_access_reads_default_initialized_class_global(
    builder_class: type[Builder],
) -> None:
    """
    title: Static field reads preserve zero/default-initialized globals.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    node = astx.ClassDefStmt(
        name="Counter",
        attributes=[
            _attribute(
                "pending",
                astx.Int32(),
                is_static=True,
            )
        ],
    )
    main_fn = _main_int32(
        astx.FunctionReturn(astx.StaticFieldAccess("Counter", "pending"))
    )
    module = make_module("main", node, main_fn)

    ir_text = builder.translate(module)
    global_name = mangle_class_static_name(
        "main",
        "Counter",
        "pending",
    )

    assert f'load i32, i32* @"{global_name}"' in ir_text
    assert_ir_parses(ir_text)
    assert_jit_int_main_result(builder, module, STATIC_DEFAULT_VALUE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_inherited_static_field_access_reuses_ancestor_storage(
    builder_class: type[Builder],
) -> None:
    """
    title: Inherited static field reads do not duplicate ancestor globals.
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
                value=astx.LiteralInt32(INHERITED_STATIC_VALUE),
            )
        ],
    )
    child = astx.ClassDefStmt(
        name="Child",
        bases=[_class_type("Base")],
    )
    main_fn = _main_int32(
        astx.FunctionReturn(astx.StaticFieldAccess("Child", "instances"))
    )
    module = make_module("main", base, child, main_fn)

    ir_text = builder.translate(module)
    base_global_name = mangle_class_static_name(
        "main",
        "Base",
        "instances",
    )
    child_global_name = mangle_class_static_name(
        "main",
        "Child",
        "instances",
    )

    assert f'load i32, i32* @"{base_global_name}"' in ir_text
    assert f'@"{child_global_name}"' not in ir_text
    assert_ir_parses(ir_text)
    assert_jit_int_main_result(builder, module, INHERITED_STATIC_VALUE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_class_type_bodies_do_not_leak_across_translations(
    builder_class: type[Builder],
) -> None:
    """
    title: Reused class names keep per-translation LLVM layouts isolated.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    first = make_module(
        "main",
        astx.ClassDefStmt(
            name="Widget",
            attributes=[_attribute("x", astx.Int32())],
        ),
        _main_int32(),
    )
    second = make_module(
        "main",
        astx.ClassDefStmt(
            name="Widget",
            attributes=[
                _attribute("x", astx.Int32()),
                _attribute("ready", astx.Boolean()),
            ],
        ),
        _main_int32(),
    )

    first_ir = builder.translate(first)
    second_ir = builder.translate(second)
    llvm_name = mangle_class_name("main", "Widget")

    assert f'%"{llvm_name}" = type {{i8*, i8*, i32}}' in first_ir
    assert f'%"{llvm_name}" = type {{i8*, i8*, i32, i1}}' in second_ir
    assert_ir_parses(second_ir)

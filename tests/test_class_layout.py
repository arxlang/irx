"""
title: Stable class layout and storage lowering tests.
"""

from __future__ import annotations

import ctypes
import re

import pytest

from irx import astx
from irx.analysis.module_symbols import (
    mangle_class_name,
    mangle_class_static_name,
)
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder
from llvmlite import binding as llvm

from tests.conftest import assert_ir_parses, make_module

GLOBAL_ADDRESS_MISSING = 0


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

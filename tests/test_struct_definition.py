"""
title: Stable struct semantics and lowering tests.
"""

from __future__ import annotations

import pytest

from irx import astx
from irx.analysis import SemanticError
from irx.analysis.module_symbols import mangle_struct_name
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder

from tests.conftest import assert_ir_parses, make_module


def _struct_type(name: str) -> astx.StructType:
    """
    title: Build one named struct type reference.
    parameters:
      name:
        type: str
    returns:
      type: astx.StructType
    """
    return astx.StructType(name)


def _field(value: astx.AST, name: str) -> astx.FieldAccess:
    """
    title: Build one field access node.
    parameters:
      value:
        type: astx.AST
      name:
        type: str
    returns:
      type: astx.FieldAccess
    """
    return astx.FieldAccess(value, name)


def _mutable_var(
    name: str,
    type_: astx.DataType,
    value: astx.DataType | astx.Undefined = astx.Undefined(),
) -> astx.VariableDeclaration:
    """
    title: Build one mutable local variable declaration.
    parameters:
      name:
        type: str
      type_:
        type: astx.DataType
      value:
        type: astx.DataType | astx.Undefined
    returns:
      type: astx.VariableDeclaration
    """
    return astx.VariableDeclaration(
        name=name,
        type_=type_,
        mutability=astx.MutabilityKind.mutable,
        value=value,
    )


def _point_struct() -> astx.StructDefStmt:
    """
    title: Build a simple point struct.
    returns:
      type: astx.StructDefStmt
    """
    return astx.StructDefStmt(
        name="Point",
        attributes=[
            astx.VariableDeclaration(name="x", type_=astx.Int32()),
            astx.VariableDeclaration(name="y", type_=astx.Int32()),
        ],
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


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_struct_definition_simple(builder_class: type[Builder]) -> None:
    """
    title: Simple struct definitions lower to identified LLVM structs.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = make_module("main", _point_struct(), _main_int32())

    ir_text = builder.translate(module)
    point_name = mangle_struct_name("main", "Point")

    assert f'%"{point_name}" = type {{i32, i32}}' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_struct_field_layout_stays_in_declaration_order(
    builder_class: type[Builder],
) -> None:
    """
    title: Field order stays stable in the emitted LLVM type.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    record = astx.StructDefStmt(
        name="Record",
        attributes=[
            astx.VariableDeclaration(name="flag", type_=astx.Boolean()),
            astx.VariableDeclaration(name="count", type_=astx.Int32()),
            astx.VariableDeclaration(name="weight", type_=astx.Float64()),
        ],
    )
    module = make_module("main", record, _main_int32())

    ir_text = builder.translate(module)

    assert '%"main__Record" = type {i1, i32, double}' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_nested_struct_layout_is_stable(builder_class: type[Builder]) -> None:
    """
    title: Nested structs preserve declaration-order layout transitively.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    inner = astx.StructDefStmt(
        name="Inner",
        attributes=[
            astx.VariableDeclaration(name="tag", type_=astx.UInt8()),
            astx.VariableDeclaration(name="value", type_=astx.Int32()),
        ],
    )
    outer = astx.StructDefStmt(
        name="Outer",
        attributes=[
            astx.VariableDeclaration(
                name="inner",
                type_=_struct_type("Inner"),
            ),
            astx.VariableDeclaration(name="ready", type_=astx.Boolean()),
        ],
    )
    module = make_module("main", inner, outer, _main_int32())

    ir_text = builder.translate(module)

    assert '%"main__Inner" = type {i8, i32}' in ir_text
    assert '%"main__Outer" = type {%"main__Inner", i1}' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_struct_field_reads_and_writes_use_stable_indices(
    builder_class: type[Builder],
) -> None:
    """
    title: Field access uses declaration-order indices for reads and writes.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    triple = astx.StructDefStmt(
        name="Triple",
        attributes=[
            astx.VariableDeclaration(name="first", type_=astx.Int32()),
            astx.VariableDeclaration(name="middle", type_=astx.Int32()),
            astx.VariableDeclaration(name="last", type_=astx.Int32()),
        ],
    )
    main_fn = _main_int32(
        _mutable_var("t", _struct_type("Triple")),
        astx.BinaryOp(
            "=",
            _field(astx.Identifier("t"), "first"),
            astx.LiteralInt32(1),
        ),
        astx.BinaryOp(
            "=",
            _field(astx.Identifier("t"), "middle"),
            astx.LiteralInt32(2),
        ),
        astx.BinaryOp(
            "=",
            _field(astx.Identifier("t"), "last"),
            astx.LiteralInt32(3),
        ),
        astx.FunctionReturn(_field(astx.Identifier("t"), "last")),
    )
    module = make_module("main", triple, main_fn)

    ir_text = builder.translate(module)

    gep_first = (
        'getelementptr inbounds %"main__Triple", '
        '%"main__Triple"* %"t", i32 0, i32 0'
    )
    gep_middle = (
        'getelementptr inbounds %"main__Triple", '
        '%"main__Triple"* %"t", i32 0, i32 1'
    )
    gep_last = (
        'getelementptr inbounds %"main__Triple", '
        '%"main__Triple"* %"t", i32 0, i32 2'
    )
    assert gep_first in ir_text
    assert gep_middle in ir_text
    assert gep_last in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_nested_field_access_reads_and_writes(
    builder_class: type[Builder],
) -> None:
    """
    title: Nested field access lowers through semantic field indices.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    inner = astx.StructDefStmt(
        name="Payload",
        attributes=[
            astx.VariableDeclaration(name="value", type_=astx.Int32()),
        ],
    )
    outer = astx.StructDefStmt(
        name="Container",
        attributes=[
            astx.VariableDeclaration(
                name="inner",
                type_=_struct_type("Payload"),
            ),
            astx.VariableDeclaration(name="enabled", type_=astx.Boolean()),
        ],
    )
    nested_value = _field(_field(astx.Identifier("o"), "inner"), "value")
    main_fn = _main_int32(
        _mutable_var("o", _struct_type("Container")),
        astx.BinaryOp("=", nested_value, astx.LiteralInt32(9)),
        astx.FunctionReturn(
            _field(_field(astx.Identifier("o"), "inner"), "value")
        ),
    )
    module = make_module("main", inner, outer, main_fn)

    ir_text = builder.translate(module)

    assert '%"main__Container" = type {%"main__Payload", i1}' in ir_text
    gep_inner = (
        'getelementptr inbounds %"main__Container", '
        '%"main__Container"* %"o", i32 0, i32 0'
    )
    assert gep_inner in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_struct_parameter_by_value_is_supported(
    builder_class: type[Builder],
) -> None:
    """
    title: Struct parameters lower by value in IRx-defined functions.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    read_point = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="read_x",
            args=astx.Arguments(astx.Argument("p", _struct_type("Point"))),
            return_type=astx.Int32(),
        ),
        body=astx.Block(),
    )
    read_point.body.append(
        astx.FunctionReturn(_field(astx.Identifier("p"), "x"))
    )
    main_fn = _main_int32(
        _mutable_var("p", _struct_type("Point")),
        astx.BinaryOp(
            "=",
            _field(astx.Identifier("p"), "x"),
            astx.LiteralInt32(4),
        ),
        astx.FunctionReturn(
            astx.FunctionCall("read_x", [astx.Identifier("p")])
        ),
    )
    module = make_module("main", _point_struct(), read_point, main_fn)

    ir_text = builder.translate(module)

    assert 'define i32 @"main__read_x"(%"main__Point" %"p")' in ir_text
    assert 'call i32 @"main__read_x"(%"main__Point" ' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_struct_return_by_value_and_assignment_from_call(
    builder_class: type[Builder],
) -> None:
    """
    title: Struct-returning functions round-trip by value through locals.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    make_point_body = astx.Block()
    make_point_body.append(_mutable_var("p", _struct_type("Point")))
    make_point_body.append(
        astx.BinaryOp(
            "=",
            _field(astx.Identifier("p"), "x"),
            astx.LiteralInt32(11),
        )
    )
    make_point_body.append(astx.FunctionReturn(astx.Identifier("p")))
    make_point = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="make_point",
            args=astx.Arguments(),
            return_type=_struct_type("Point"),
        ),
        body=make_point_body,
    )
    main_fn = _main_int32(
        _mutable_var(
            "p",
            _struct_type("Point"),
            value=astx.FunctionCall("make_point", []),
        ),
        astx.FunctionReturn(_field(astx.Identifier("p"), "x")),
    )
    module = make_module("main", _point_struct(), make_point, main_fn)

    ir_text = builder.translate(module)

    assert 'define %"main__Point" @"main__make_point"()' in ir_text
    assert 'call %"main__Point" @"main__make_point"()' in ir_text
    assert 'store %"main__Point" %"calltmp", %"main__Point"* %"p"' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_nested_struct_return_by_value_is_supported(
    builder_class: type[Builder],
) -> None:
    """
    title: Nested structs can be returned by value without hidden fields.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    inner = astx.StructDefStmt(
        name="Payload",
        attributes=[
            astx.VariableDeclaration(name="value", type_=astx.Int32()),
        ],
    )
    outer = astx.StructDefStmt(
        name="Container",
        attributes=[
            astx.VariableDeclaration(
                name="inner",
                type_=_struct_type("Payload"),
            ),
            astx.VariableDeclaration(name="ready", type_=astx.Boolean()),
        ],
    )
    make_outer_body = astx.Block()
    make_outer_body.append(_mutable_var("o", _struct_type("Container")))
    make_outer_body.append(
        astx.BinaryOp(
            "=",
            _field(_field(astx.Identifier("o"), "inner"), "value"),
            astx.LiteralInt32(21),
        )
    )
    make_outer_body.append(astx.FunctionReturn(astx.Identifier("o")))
    make_outer = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="make_outer",
            args=astx.Arguments(),
            return_type=_struct_type("Container"),
        ),
        body=make_outer_body,
    )
    main_fn = _main_int32(
        _mutable_var(
            "o",
            _struct_type("Container"),
            value=astx.FunctionCall("make_outer", []),
        ),
        astx.FunctionReturn(
            _field(_field(astx.Identifier("o"), "inner"), "value")
        ),
    )
    module = make_module("main", inner, outer, make_outer, main_fn)

    ir_text = builder.translate(module)

    assert '%"main__Container" = type {%"main__Payload", i1}' in ir_text
    assert 'define %"main__Container" @"main__make_outer"()' in ir_text
    assert_ir_parses(ir_text)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_duplicate_struct_name_is_rejected(
    builder_class: type[Builder],
) -> None:
    """
    title: Duplicate struct names are semantic errors.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = make_module(
        "main",
        astx.StructDefStmt(
            name="Duplicate",
            attributes=[
                astx.VariableDeclaration(name="x", type_=astx.Int32())
            ],
        ),
        astx.StructDefStmt(
            name="Duplicate",
            attributes=[
                astx.VariableDeclaration(name="y", type_=astx.Int32())
            ],
        ),
        _main_int32(),
    )

    with pytest.raises(
        SemanticError,
        match=r"Struct 'Duplicate' already defined\.",
    ):
        builder.translate(module)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_duplicate_field_name_is_rejected(
    builder_class: type[Builder],
) -> None:
    """
    title: Duplicate field names are semantic errors.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = make_module(
        "main",
        astx.StructDefStmt(
            name="Broken",
            attributes=[
                astx.VariableDeclaration(name="x", type_=astx.Int32()),
                astx.VariableDeclaration(name="x", type_=astx.Int32()),
            ],
        ),
        _main_int32(),
    )

    with pytest.raises(
        SemanticError,
        match=r"Struct field 'x' already defined\.",
    ):
        builder.translate(module)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_unknown_field_type_is_rejected(builder_class: type[Builder]) -> None:
    """
    title: Unknown field types are semantic errors.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = make_module(
        "main",
        astx.StructDefStmt(
            name="Broken",
            attributes=[
                astx.VariableDeclaration(
                    name="child",
                    type_=_struct_type("Missing"),
                ),
            ],
        ),
        _main_int32(),
    )

    with pytest.raises(SemanticError, match="Unknown field type 'Missing'"):
        builder.translate(module)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_empty_structs_are_rejected(builder_class: type[Builder]) -> None:
    """
    title: Empty structs are explicitly unsupported for now.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = make_module(
        "main",
        astx.StructDefStmt(name="Empty", attributes=[]),
        _main_int32(),
    )

    with pytest.raises(
        SemanticError,
        match="Struct 'Empty' must declare at least one field",
    ):
        builder.translate(module)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_invalid_field_access_is_rejected(
    builder_class: type[Builder],
) -> None:
    """
    title: Unknown fields are rejected before lowering.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    main_fn = _main_int32(
        _mutable_var("p", _struct_type("Point")),
        astx.FunctionReturn(_field(astx.Identifier("p"), "missing")),
    )
    module = make_module("main", _point_struct(), main_fn)

    with pytest.raises(
        SemanticError,
        match="struct 'Point' has no field 'missing'",
    ):
        builder.translate(module)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_direct_recursive_struct_is_rejected(
    builder_class: type[Builder],
) -> None:
    """
    title: Direct by-value recursive structs are forbidden.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = make_module(
        "main",
        astx.StructDefStmt(
            name="Node",
            attributes=[
                astx.VariableDeclaration(
                    name="next",
                    type_=_struct_type("Node"),
                ),
            ],
        ),
        _main_int32(),
    )

    with pytest.raises(
        SemanticError,
        match="direct by-value recursive struct 'Node' is forbidden",
    ):
        builder.translate(module)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_mutual_recursive_structs_are_rejected(
    builder_class: type[Builder],
) -> None:
    """
    title: Mutual by-value recursive structs are forbidden.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = make_module(
        "main",
        astx.StructDefStmt(
            name="Left",
            attributes=[
                astx.VariableDeclaration(
                    name="right",
                    type_=_struct_type("Right"),
                ),
            ],
        ),
        astx.StructDefStmt(
            name="Right",
            attributes=[
                astx.VariableDeclaration(
                    name="left",
                    type_=_struct_type("Left"),
                ),
            ],
        ),
        _main_int32(),
    )

    with pytest.raises(
        SemanticError,
        match=(
            "mutual by-value recursive structs are forbidden: "
            "Left -> Right -> Left"
        ),
    ):
        builder.translate(module)

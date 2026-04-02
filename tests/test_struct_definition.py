"""
title: Test Struct Definition
summary: Verify StructDefStmt generates an LLVM identified struct type.
"""

import pytest

from irx import astx
from irx.analysis import SemanticError
from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_struct_definition(builder_class: type[Builder]) -> None:
    """
    title: Struct definition code generation
    summary: Ensure StructDefStmt translates to an LLVM struct type.
    parameters:
      builder_class:
        type: type[Builder]
    """

    builder = builder_class()
    module = builder.module()

    # Define struct: Point { x: int32, y: int32 }
    struct_def = astx.StructDefStmt(
        name="Point",
        attributes=[
            astx.VariableDeclaration(name="x", type_=astx.Int32()),
            astx.VariableDeclaration(name="y", type_=astx.Int32()),
        ],
    )

    # Define main() -> int32
    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )

    main_block = astx.Block()
    main_block.append(struct_def)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    main_fn = astx.FunctionDef(
        prototype=main_proto,
        body=main_block,
    )

    module.block.append(main_fn)

    ir_text = builder.translate(module)
    assert '%"Point" = type {i32, i32}' in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_struct_definition_single_field(builder_class: type[Builder]) -> None:
    """
    title: Single field struct definition
    summary: Ensure StructDefStmt works with a single attribute.
    parameters:
      builder_class:
        type: type[Builder]
    """

    builder = builder_class()
    module = builder.module()

    struct_def = astx.StructDefStmt(
        name="Value",
        attributes=[
            astx.VariableDeclaration(name="x", type_=astx.Int32()),
        ],
    )

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )

    main_block = astx.Block()
    main_block.append(struct_def)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    main_fn = astx.FunctionDef(
        prototype=main_proto,
        body=main_block,
    )

    module.block.append(main_fn)

    ir_text = builder.translate(module)
    assert '%"Value" = type {i32}' in ir_text


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_struct_definition_duplicate_name(
    builder_class: type[Builder],
) -> None:
    """
    title: Duplicate struct name raises error
    summary: >-
      Ensure defining a struct with the same name twice raises ValueError.
    parameters:
      builder_class:
        type: type[Builder]
    """

    builder = builder_class()
    module = builder.module()

    struct_a = astx.StructDefStmt(
        name="Duplicate",
        attributes=[
            astx.VariableDeclaration(name="x", type_=astx.Int32()),
        ],
    )

    struct_b = astx.StructDefStmt(
        name="Duplicate",
        attributes=[
            astx.VariableDeclaration(name="y", type_=astx.Int32()),
        ],
    )

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )

    main_block = astx.Block()
    main_block.append(struct_a)
    main_block.append(struct_b)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    main_fn = astx.FunctionDef(
        prototype=main_proto,
        body=main_block,
    )

    module.block.append(main_fn)

    with pytest.raises(SemanticError, match="already defined"):
        builder.translate(module)

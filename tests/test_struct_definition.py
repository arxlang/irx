"""
title: Struct Definition Tests
summary: Tests for StructDefStmt LLVM IR generation.
"""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_struct_definition(builder_class: Type[Builder]) -> None:
    """
    title: Basic Struct Definition
    summary: Verify StructDefStmt generates correct LLVM struct type.
    parameters:
      builder_class:
        type: Type[Builder]
    """

    builder = builder_class()
    module = builder.module()

    # Define struct
    struct_def = astx.StructDefStmt(
        name="Point",
        attributes=[
            astx.VariableDeclaration(name="x", type_=astx.Int32()),
            astx.VariableDeclaration(name="y", type_=astx.Int32()),
        ],
    )

    # main() function
    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )

    main_block = astx.Block()
    main_block.append(struct_def)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="0")

"""
title: Test Struct Definition
summary: Verify StructDefStmt generates an LLVM identified struct type.
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
    title: Struct definition code generation
    summary: Ensure StructDefStmt translates to an LLVM struct type.
    parameters:
      builder_class:
        type: Type[Builder]
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

    # Verify LLVM IR translation
    check_result("translate", builder, module)

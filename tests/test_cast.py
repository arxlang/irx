from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import Cast

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type_from, literal_type_from, int_type_to",
    [
        (astx.Int8, astx.LiteralInt8, astx.Int32),
        (astx.Int16, astx.LiteralInt16, astx.Int8),
        (astx.Int32, astx.LiteralInt32, astx.Int16),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_cast_basic.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_cast_basic(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type_from: type,
    literal_type_from: type,
    int_type_to: type,
) -> None:
    """Test casting int types between different widths."""
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a", type_=int_type_from(), value=literal_type_from(42)
    )
    a = astx.Variable("a")
    cast_expr = Cast(value=a, target_type=int_type_to())

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type_to()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    cast_var = astx.InlineVariableDeclaration(
        name="r", type_=int_type_to(), value=cast_expr
    )
    main_block.append(cast_var)
    main_block.append(astx.FunctionReturn(astx.Variable("r")))
    main_fn = astx.Function(prototype=main_proto, body=main_block)

    module.block.append(main_fn)
    check_result(action, builder, module, expected_file)

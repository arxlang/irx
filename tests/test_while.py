"""Test While Loop statements."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [(astx.Int32, astx.LiteralInt32), (astx.Int16, astx.LiteralInt16)],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_while_expr.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_while_expr(
    action: str,
    expected_file: str,
    builder_class: Type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """Test the While expression translation."""
    builder = builder_class()

    # Variable declaration and initialization: int a = 0
    init_var = astx.InlineVariableDeclaration(
        "a", type_=int_type(), value=literal_type(0)
    )

    # Condition: a < 5
    var_a = astx.Variable("a")
    cond = astx.BinaryOp(op_code="<", lhs=var_a, rhs=literal_type(5))

    # Update: ++a
    update = astx.UnaryOp(op_code="++", operand=var_a)

    # Body
    body = astx.Block()
    body.append(update)
    body.append(literal_type(2))

    while_expr = astx.WhileStmt(condition=cond, body=body)

    # Main function
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    fn_block = astx.Block()
    fn_block.append(init_var)
    fn_block.append(while_expr)
    fn_block.append(astx.FunctionReturn(literal_type(0)))

    fn_main = astx.Function(prototype=proto, body=fn_block)

    module = builder.module()
    module.block.append(fn_main)

    check_result(action, builder, module, expected_file)

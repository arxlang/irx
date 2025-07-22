# """Tests for floating-point operations using different IR builders."""

# import subprocess

# from typing import Type

# import astx
# import pytest

# from irx.builders.base import Builder
# from irx.builders.llvmliteir import LLVMLiteIR
# from irx.system import Cast

# from .conftest import check_result


# @pytest.mark.parametrize(
#     "lhs,op,rhs,expected",
#     [
#         (1.0, "+", 2.5, "3"),  # 1.0 + 2.5 = 3.5 (expect 3 due to cast to int)
#         (6.2, "-", 4.2, "2"),
#         (2.1, "*", 3.0, "6"),  # 2.0 * 3.0 = 6
#         (2.0, "/", 1.0, "3"),
#     ],
# )
# @pytest.mark.parametrize("builder_class", [LLVMLiteIR])
# def test_float_operations(
#     builder_class: Type[Builder],
#     lhs: float,
#     op: str,
#     rhs: float,
#     expected: str,
# ) -> None:
#     """Test float operations for different IR builders."""
#     builder = builder_class()
#     module = builder.module()

#     # Construct float operation and cast to int
#     left = astx.LiteralFloat32(lhs)
#     right = astx.LiteralFloat32(rhs)
#     expr = astx.BinaryOp(op, left, right)
#     cast_expr = Cast(value=expr, target_type=astx.Int32())

#     # Declare and return variable
#     decl_tmp = astx.VariableDeclaration(
#         name="tmp", type_=astx.Int32(), value=cast_expr
#     )

#     proto = astx.FunctionPrototype(
#         name="main", args=astx.Arguments(), return_type=astx.Int32()
#     )
#     block = astx.Block()
#     block.append(decl_tmp)
#     block.append(astx.FunctionReturn(astx.Variable("tmp")))
#     fn = astx.FunctionDef(prototype=proto, body=block)
#     module.block.append(fn)

#     success = True
#     try:
#         check_result("build", builder, module, expected_output=expected)
#     except subprocess.CalledProcessError as e:
#         success = False
#         assert e.returncode == int(expected)
#     assert not success

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result  # we'll update this too


@pytest.mark.parametrize(
    "lhs, op, rhs, expected",
    [
        (1.0, "+", 2.5, 3.5),
        (6.2, "-", 4.2, 2.0),
        (2.1, "*", 3.0, 6.3),
        (2.0, "/", 1.0, 2.0),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_float_operations_with_print(
    builder_class: Type[Builder],
    lhs: float,
    op: str,
    rhs: float,
    expected: float,
) -> None:
    """Test float operations by printing result to stdout."""
    builder = builder_class()
    module = builder.module()

    # Build expression: lhs <op> rhs
    left = astx.LiteralFloat32(lhs)
    right = astx.LiteralFloat32(rhs)
    expr = astx.BinaryOp(op, left, right)

    # Declare tmp: float32 = expr
    decl_tmp = astx.VariableDeclaration(
        name="tmp", type_=astx.Float32(), value=expr
    )

    # Return block that prints float then returns 0
    block = astx.Block()
    block.append(decl_tmp)
    block.append(PrintExpr(astx.Variable("tmp"), name="print_tmp"))
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    # Define: float main() -> returns 0
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, expected_output=expected)

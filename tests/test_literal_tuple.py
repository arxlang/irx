"""
title: Tests for LiteralTuple lowering using project conventions.
"""

from __future__ import annotations

from typing import Type, cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

HAS_LITERAL_TUPLE = hasattr(astx, "LiteralTuple")


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_empty(builder_class: Type[Builder]) -> None:
    """
    title: Empty tuple lowers to constant empty literal struct {}.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTuple(elements=()))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert not const.type.elements
    expected = ir.Constant(ir.LiteralStructType([]), [])
    assert str(const) == str(expected)
    assert not visitor.result_stack


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_homogeneous_constants(
    builder_class: Type[Builder],
) -> None:
    """
    title: Homogeneous tuple constants lower to constant struct.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralTuple(
            elements=(
                astx.LiteralInt32(1),
                astx.LiteralInt32(2),
                astx.LiteralInt32(3),
            )
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == 3  # noqa: PLR2004
    assert [str(t) for t in const.type.elements] == ["i32", "i32", "i32"]
    i32 = ir.IntType(32)
    expected = ir.Constant(
        ir.LiteralStructType([i32, i32, i32]),
        [ir.Constant(i32, v) for v in (1, 2, 3)],
    )
    assert str(const) == str(expected)
    assert not visitor.result_stack


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_heterogeneous_constants(
    builder_class: Type[Builder],
) -> None:
    """
    title: Heterogeneous tuple constants lower to constant struct.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralTuple(
            elements=(
                astx.LiteralInt32(1),
                astx.LiteralFloat32(2.5),
            )
        )
    )
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == 2  # noqa: PLR2004
    assert [str(t) for t in const.type.elements] == ["i32", "float"]
    expected = ir.Constant(
        ir.LiteralStructType([ir.IntType(32), ir.FloatType()]),
        [ir.Constant(ir.IntType(32), 1), ir.Constant(ir.FloatType(), 2.5)],
    )
    assert str(const) == str(expected)
    assert not visitor.result_stack


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_single_element(builder_class: Type[Builder]) -> None:
    """
    title: Single element tuple lowers to single field struct.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTuple(elements=(astx.LiteralBoolean(True),)))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == 1
    assert [str(t) for t in const.type.elements] == ["i1"]
    expected = ir.Constant(
        ir.LiteralStructType([ir.IntType(1)]),
        [ir.Constant(ir.IntType(1), 1)],
    )
    assert str(const) == str(expected)
    assert not visitor.result_stack


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_global_value_constant(
    builder_class: Type[Builder],
) -> None:
    """
    title: Tuple containing GlobalValues is treated as a constant initializer.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    # Mock elements that lower to GlobalVariables
    class MockGlobalNode(astx.LiteralNone):
        pass

    i32 = ir.IntType(32)
    # Create two global variables without an explicit ir_builder function body
    gv1 = ir.GlobalVariable(visitor._llvm.module, i32, name="mock_gv_1")
    gv2 = ir.GlobalVariable(visitor._llvm.module, i32, name="mock_gv_2")

    def mock_visit(node: MockGlobalNode) -> None:
        if not hasattr(mock_visit, "toggle"):
            mock_visit.toggle = True  # type: ignore
            visitor.result_stack.append(gv1)
        else:
            visitor.result_stack.append(gv2)

    original_visit = visitor.visit

    def proxy_visit(node: object) -> None:
        if isinstance(node, MockGlobalNode):
            mock_visit(node)
        else:
            original_visit(node)

    # We patch visitor.visit locally for this test execution only
    visitor.visit = proxy_visit  # type: ignore

    tuple_node = astx.LiteralTuple(
        elements=(MockGlobalNode(), MockGlobalNode())
    )

    try:
        visitor.visit(tuple_node)
    finally:
        # Restore original visitor function
        visitor.visit = original_visit  # type: ignore

    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == 2  # noqa: PLR2004

    # gv1 and gv2 are i32*, embedding directly produces i32* fields.
    ptr_type_str = str(i32.as_pointer())
    assert [str(t) for t in const.type.elements] == [
        ptr_type_str,
        ptr_type_str,
    ]
    s = str(const)
    assert '@"mock_gv_1"' in s and '@"mock_gv_2"' in s
    assert not visitor.result_stack


@pytest.mark.skipif(
    not HAS_LITERAL_TUPLE, reason="astx.LiteralTuple not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_builder_guard(
    builder_class: Type[Builder],
) -> None:
    """
    title: Exception is raised if builder is not ready for dynamic tuple.
    parameters:
      builder_class:
        type: Type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    # Intentionally wipe builder context
    visitor._llvm.ir_builder = None

    # Mock node that generates a non-constant instruction
    class MockDynNode(astx.LiteralNone):
        pass

    i32 = ir.IntType(32)
    f = ir.Function(
        visitor._llvm.module, ir.FunctionType(i32, [i32]), name="dummy"
    )
    dummy_val = f.args[0]

    original_visit = visitor.visit

    def proxy_visit(node: object) -> None:
        if isinstance(node, MockDynNode):
            visitor.result_stack.append(dummy_val)
        else:
            original_visit(node)

    visitor.visit = proxy_visit  # type: ignore

    tuple_node = astx.LiteralTuple(elements=(MockDynNode(),))

    try:
        with pytest.raises(Exception, match="global initializer context"):
            visitor.visit(tuple_node)
    finally:
        visitor.visit = original_visit  # type: ignore

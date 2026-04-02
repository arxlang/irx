"""
title: Tests for LiteralList lowering using project conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import pytest

from irx import astx
from irx.builders.llvmliteir import Visitor as LLVMVisitor
from llvmlite import ir

HAS_LITERAL_LIST = hasattr(astx, "LiteralList")

pytestmark = pytest.mark.skipif(
    not HAS_LITERAL_LIST,
    reason="astx.LiteralList not available",
)


@dataclass(frozen=True)
class LiteralListScenario:
    """
    title: One positive LiteralList lowering scenario.
    attributes:
      elements_factory:
        type: Callable[[], list[astx.Literal]]
      requires_live_builder:
        type: bool
      expected_length:
        type: int
      expected_element_type:
        type: ir.Type
      expected_values:
        type: list[int | float]
    """

    elements_factory: Callable[[], list[astx.Literal]]
    requires_live_builder: bool
    expected_length: int
    expected_element_type: ir.Type
    expected_values: list[int | float]


def _constant_scalars(const: ir.Constant) -> list[int | float]:
    """
    title: Extract scalar values from an array constant payload.
    parameters:
      const:
        type: ir.Constant
    returns:
      type: list[int | float]
    """
    items = cast(list[ir.Constant], const.constant)
    return [cast(int | float, item.constant) for item in items]


POSITIVE_SCENARIOS = [
    pytest.param(
        LiteralListScenario(
            elements_factory=lambda: [],
            requires_live_builder=False,
            expected_length=0,
            expected_element_type=ir.IntType(32),
            expected_values=[],
        ),
        id="empty",
    ),
    pytest.param(
        LiteralListScenario(
            elements_factory=lambda: [
                astx.LiteralInt32(1),
                astx.LiteralInt32(2),
                astx.LiteralInt32(3),
            ],
            requires_live_builder=False,
            expected_length=3,
            expected_element_type=ir.IntType(32),
            expected_values=[1, 2, 3],
        ),
        id="homogeneous-ints",
    ),
    pytest.param(
        LiteralListScenario(
            elements_factory=lambda: [
                astx.LiteralInt16(1),
                astx.LiteralInt32(2),
            ],
            requires_live_builder=True,
            expected_length=2,
            expected_element_type=ir.IntType(32),
            expected_values=[1, 2],
        ),
        id="mixed-int-widths",
    ),
    pytest.param(
        LiteralListScenario(
            elements_factory=lambda: [
                astx.LiteralFloat32(1.0),
                astx.LiteralFloat32(2.0),
                astx.LiteralFloat32(3.0),
            ],
            requires_live_builder=True,
            expected_length=3,
            expected_element_type=ir.FloatType(),
            expected_values=[1.0, 2.0, 3.0],
        ),
        id="homogeneous-floats",
    ),
    pytest.param(
        LiteralListScenario(
            elements_factory=lambda: [
                astx.LiteralInt32(1),
                astx.LiteralFloat32(2.0),
            ],
            requires_live_builder=True,
            expected_length=2,
            expected_element_type=ir.FloatType(),
            expected_values=[1.0, 2.0],
        ),
        id="mixed-int-and-float",
    ),
    pytest.param(
        LiteralListScenario(
            elements_factory=lambda: [
                astx.LiteralFloat32(1.0),
                astx.LiteralFloat64(2.0),
            ],
            requires_live_builder=True,
            expected_length=2,
            expected_element_type=ir.DoubleType(),
            expected_values=[1.0, 2.0],
        ),
        id="mixed-float-widths",
    ),
]


@pytest.mark.parametrize("scenario", POSITIVE_SCENARIOS)
def test_literal_list_positive_scenarios(
    request: pytest.FixtureRequest,
    scenario: LiteralListScenario,
) -> None:
    """
    title: LiteralList should lower expected array constants across scenarios.
    parameters:
      request:
        type: pytest.FixtureRequest
      scenario:
        type: LiteralListScenario
    """
    fixture_name = (
        "llvm_visitor_in_function"
        if scenario.requires_live_builder
        else "llvm_visitor"
    )
    visitor = cast(LLVMVisitor, request.getfixturevalue(fixture_name))

    visitor.visit(astx.LiteralList(elements=scenario.elements_factory()))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.ArrayType)
    assert const.type.count == scenario.expected_length
    assert const.type.element == scenario.expected_element_type

    actual_values = _constant_scalars(const)
    if any(isinstance(value, float) for value in scenario.expected_values):
        assert actual_values == pytest.approx(scenario.expected_values)
    else:
        assert actual_values == scenario.expected_values

    assert not visitor.result_stack


def test_literal_list_incompatible_types_raise(
    llvm_visitor_in_function: LLVMVisitor,
) -> None:
    """
    title: Incompatible LiteralList element types should raise TypeError.
    parameters:
      llvm_visitor_in_function:
        type: LLVMVisitor
    """
    with pytest.raises(
        TypeError, match="LiteralList: cannot find common type"
    ):
        llvm_visitor_in_function.visit(
            astx.LiteralList(
                elements=[
                    astx.LiteralUTF8String("hello"),
                    astx.LiteralInt32(1),
                ]
            )
        )


def test_literal_list_nested_unsupported_raises_at_ast_construction() -> None:
    """
    title: Nested LiteralList values are not currently constructible.
    """
    with pytest.raises(TypeError, match=r"missing.*argument.*element_types"):
        astx.LiteralList(elements=[astx.LiteralList(elements=[])])

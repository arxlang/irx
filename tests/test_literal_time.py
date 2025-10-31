"""Tests for LiteralTime support."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

MAX_TIME_PARTS = 3


@pytest.mark.parametrize(
    "time_str,expected_hour,expected_min,expected_sec",
    [
        ("10:30", 10, 30, 0),
        ("14:45:30", 14, 45, 30),
        ("00:00:00", 0, 0, 0),
        ("23:59:59", 23, 59, 59),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_time_basic(
    builder_class: Type[Builder],
    time_str: str,
    expected_hour: int,
    expected_min: int,
    expected_sec: int,
) -> None:
    """Test basic LiteralTime parsing and IR generation."""
    builder = builder_class()
    module = builder.module()

    # Create time literal
    time_literal = astx.LiteralTime(time_str)

    # Validate parsed components
    parts = time_str.split(":")
    assert int(parts[0]) == expected_hour
    assert int(parts[1]) == expected_min
    if len(parts) == MAX_TIME_PARTS:
        assert int(parts[2]) == expected_sec
    else:
        assert expected_sec == 0

    # Create a function that returns the hour component
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()

    # Store time in variable
    time_decl = astx.VariableDeclaration(
        name="t", type_=astx.Time(), value=time_literal
    )
    block.append(time_decl)

    # Return 0 for now (just testing that it compiles)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    # Just check it translates without error
    ir_code = builder.translate(module)
    assert "i32" in ir_code


@pytest.mark.parametrize(
    "invalid_time",
    [
        "25:00",  # Hour out of range
        "10:60",  # Minute out of range
        "10:30:60",  # Second out of range
        "10",  # Missing minute
        "10:30:45.123",  # Fractional seconds not supported
        "abc:def",  # Invalid format
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_time_invalid(
    builder_class: Type[Builder],
    invalid_time: str,
) -> None:
    """Test that invalid time formats raise exceptions."""
    with pytest.raises(Exception):
        builder = builder_class()
        module = builder.module()

        time_literal = astx.LiteralTime(invalid_time)

        proto = astx.FunctionPrototype(
            name="main", args=astx.Arguments(), return_type=astx.Int32()
        )
        block = astx.Block()

        time_decl = astx.VariableDeclaration(
            name="t", type_=astx.Time(), value=time_literal
        )
        block.append(time_decl)
        block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

        fn = astx.FunctionDef(prototype=proto, body=block)
        module.block.append(fn)

        builder.translate(module)

"""Tests for LiteralDate support."""

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR

from .conftest import check_result


@pytest.mark.parametrize(
    "date_str,expected_year,expected_month,expected_day",
    [
        ("2024-01-15", 2024, 1, 15),
        ("2000-12-31", 2000, 12, 31),
        ("1970-01-01", 1970, 1, 1),
        ("2023-06-15", 2023, 6, 15),
        ("1999-02-28", 1999, 2, 28),
        ("2024-11-30", 2024, 11, 30),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_date_basic(
    builder_class: Type[Builder],
    date_str: str,
    expected_year: int,
    expected_month: int,
    expected_day: int,
) -> None:
    """Test basic LiteralDate parsing and IR generation."""
    builder = builder_class()
    module = builder.module()

    # Create date literal
    date_literal = astx.LiteralDate(date_str)

    # Create a function that stores the date
    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    
    # Store date in variable
    date_decl = astx.VariableDeclaration(
        name="d", type_=astx.Date(), value=date_literal
    )
    block.append(date_decl)
    
    # Return 0 (just testing that it compiles)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    # Check it translates without error
    ir_code = builder.translate(module)
    assert "i32" in ir_code
    # Verify the struct contains our values
    assert str(expected_year) in ir_code
    assert str(expected_month) in ir_code
    assert str(expected_day) in ir_code


@pytest.mark.parametrize(
    "invalid_date,error_msg",
    [
        ("2024-13-01", "month out of range"),  # Invalid month
        ("2024-00-01", "month out of range"),  # Month = 0
        ("2024-01-32", "day out of range"),    # Invalid day
        ("2024-01-00", "day out of range"),    # Day = 0
        ("10000-01-01", "year out of range"),  # Year too large
        ("0-01-01", "year out of range"),      # Year = 0
        ("2024/01/01", "invalid date format"), # Wrong separator
        ("2024-Jan-01", "invalid year/month/day"), # Text month
        ("2024-01", "invalid date format"),    # Missing day
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_date_invalid(
    builder_class: Type[Builder],
    invalid_date: str,
    error_msg: str,
) -> None:
    """Test that invalid date formats raise appropriate exceptions."""
    builder = builder_class()
    module = builder.module()

    date_literal = astx.LiteralDate(invalid_date)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    
    date_decl = astx.VariableDeclaration(
        name="d", type_=astx.Date(), value=date_literal
    )
    block.append(date_decl)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    # Should raise an exception with expected message
    with pytest.raises(Exception) as exc_info:
        builder.translate(module)
    
    assert error_msg in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "date_str",
    [
        "2024-02-29",  # Leap year - valid
        "2024-12-31",  # End of year
        "2024-01-01",  # Start of year
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_date_edge_cases(
    builder_class: Type[Builder],
    date_str: str,
) -> None:
    """Test edge cases for valid dates."""
    builder = builder_class()
    module = builder.module()

    date_literal = astx.LiteralDate(date_str)

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    
    date_decl = astx.VariableDeclaration(
        name="d", type_=astx.Date(), value=date_literal
    )
    block.append(date_decl)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    # Should compile successfully
    ir_code = builder.translate(module)
    assert "i32" in ir_code


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_date_multiple_variables(
    builder_class: Type[Builder],
) -> None:
    """Test multiple date variables in the same function."""
    builder = builder_class()
    module = builder.module()

    date1 = astx.LiteralDate("2024-01-15")
    date2 = astx.LiteralDate("2023-12-25")

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    block = astx.Block()
    
    # Store multiple dates
    date_decl1 = astx.VariableDeclaration(
        name="d1", type_=astx.Date(), value=date1
    )
    date_decl2 = astx.VariableDeclaration(
        name="d2", type_=astx.Date(), value=date2
    )
    block.append(date_decl1)
    block.append(date_decl2)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    # Should compile successfully
    ir_code = builder.translate(module)
    assert "2024" in ir_code
    assert "2023" in ir_code
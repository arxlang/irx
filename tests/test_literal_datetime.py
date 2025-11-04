"""Tests for LiteralDateTime lowering using project conventions."""

from __future__ import annotations

import re

from typing import Type

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from llvmlite import ir

from .conftest import check_result

HAS_LITERAL_DATETIME = hasattr(astx, "LiteralDateTime")


def _datetime_values(const: ir.Constant) -> list[int]:
    """Extract i32 values from the literal struct constant."""
    return [int(v) for v in re.findall(r"i32\s+(-?\d+)", str(const))]


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_basic_hms(builder_class: Type[Builder]) -> None:
    """Integration: lowering succeeds and program builds."""
    builder = builder_class()
    module = builder.module()

    datetime_node = astx.LiteralDateTime("2025-10-30T12:34:56")
    block = astx.Block()
    block.append(datetime_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, "")


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_basic_hm(builder_class: Type[Builder]) -> None:
    """Integration: HH:MM defaults seconds to 0 and builds."""
    builder = builder_class()
    module = builder.module()

    datetime_node = astx.LiteralDateTime("2025-10-30 12:34")
    block = astx.Block()
    block.append(datetime_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, "")


@pytest.mark.parametrize(
    "datetime_str, expected_values",
    [
        ("2025-10-30T12:34:56", [2025, 10, 30, 12, 34, 56]),
        ("2025-01-02 03:04:05", [2025, 1, 2, 3, 4, 5]),
        ("2024-12-31T23:59:59", [2024, 12, 31, 23, 59, 59]),
    ],
)
@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_parsing(
    builder_class: Type[Builder],
    datetime_str: str,
    expected_values: list[int],
) -> None:
    """Integration: various formats build."""
    builder = builder_class()
    module = builder.module()

    datetime_node = astx.LiteralDateTime(datetime_str)
    block = astx.Block()
    block.append(datetime_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    check_result("build", builder, module, "")


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_fractional_rejected(
    builder_class: Type[Builder],
) -> None:
    """Integration: fractional seconds rejected during build."""
    builder = builder_class()
    module = builder.module()

    datetime_node = astx.LiteralDateTime("2025-10-30T12:34:56.123")
    block = astx.Block()
    block.append(datetime_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="fractional seconds"):
        check_result("build", builder, module, "")


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_timezone_rejected(
    builder_class: Type[Builder],
) -> None:
    """Integration: timezone markers rejected during build."""
    builder = builder_class()
    module = builder.module()

    datetime_node = astx.LiteralDateTime("2025-10-30T12:34:56Z")
    block = astx.Block()
    block.append(datetime_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="timezone"):
        check_result("build", builder, module, "")


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_invalid_month(builder_class: Type[Builder]) -> None:
    """Integration: invalid month rejected during build."""
    builder = builder_class()
    module = builder.module()

    datetime_node = astx.LiteralDateTime("2025-13-01T00:00:00")
    block = astx.Block()
    block.append(datetime_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="calendar date"):
        check_result("build", builder, module, "")


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_invalid_day(builder_class: Type[Builder]) -> None:
    """Integration: impossible calendar dates rejected during build."""
    builder = builder_class()
    module = builder.module()

    datetime_node = astx.LiteralDateTime("2025-12-32T00:00:00")
    block = astx.Block()
    block.append(datetime_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="calendar date"):
        check_result("build", builder, module, "")


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_invalid_hour(builder_class: Type[Builder]) -> None:
    """Integration: hour out of range rejected during build."""
    builder = builder_class()
    module = builder.module()

    datetime_node = astx.LiteralDateTime("2025-10-30T24:00:00")
    block = astx.Block()
    block.append(datetime_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="hour out of range"):
        check_result("build", builder, module, "")


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_invalid_minute(builder_class: Type[Builder]) -> None:
    """Integration: minute out of range rejected during build."""
    builder = builder_class()
    module = builder.module()

    datetime_node = astx.LiteralDateTime("2025-10-30T12:60:00")
    block = astx.Block()
    block.append(datetime_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="minute out of range"):
        check_result("build", builder, module, "")


@pytest.mark.skipif(
    not HAS_LITERAL_DATETIME, reason="astx.LiteralDateTime not available"
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_datetime_invalid_second(builder_class: Type[Builder]) -> None:
    """Integration: second out of range rejected during build."""
    builder = builder_class()
    module = builder.module()

    datetime_node = astx.LiteralDateTime("2025-10-30T12:34:60")
    block = astx.Block()
    block.append(datetime_node)
    block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    fn = astx.FunctionDef(prototype=proto, body=block)
    module.block.append(fn)

    with pytest.raises(Exception, match="second out of range"):
        check_result("build", builder, module, "")

"""
title: Runtime-oriented llvmliteir helpers.
"""

from __future__ import annotations

from llvmlite import ir

from irx.typecheck import typechecked


@typechecked
def safe_pop(
    values: list[ir.Value | ir.Function | None],
) -> ir.Value | ir.Function | None:
    try:
        return values.pop()
    except IndexError:
        return None


__all__ = ["safe_pop"]

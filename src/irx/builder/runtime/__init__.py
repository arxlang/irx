"""
title: Builder runtime helpers and runtime feature support.
"""

from __future__ import annotations

import builtins

from llvmlite import ir

from irx.typecheck import typechecked


@typechecked
def safe_pop(
    values: builtins.list[ir.Value | ir.Function | None],
) -> ir.Value | ir.Function | None:
    """
    title: Safe pop.
    parameters:
      values:
        type: builtins.list[ir.Value | ir.Function | None]
    returns:
      type: ir.Value | ir.Function | None
    """
    try:
        return values.pop()
    except IndexError:
        return None


__all__ = ["safe_pop"]

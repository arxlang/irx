"""
title: Shared state typing for LLVMLiteIR codegen.
"""

from __future__ import annotations

from typing import Any

from llvmlite import ir

ResultStackValue = ir.Value | ir.Function | None
NamedValueMap = dict[str, Any]

__all__ = ["NamedValueMap", "ResultStackValue"]

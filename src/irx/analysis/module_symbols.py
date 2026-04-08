"""
title: Module-aware semantic identity and LLVM mangling helpers.
summary: >-
  Centralize module-qualified semantic ids and deterministic LLVM mangling
  rules for cross-module declarations.
"""

from __future__ import annotations

import re

from public import public

from irx.analysis.module_interfaces import ModuleKey
from irx.typecheck import typechecked

_SEGMENT_RE = re.compile(r"[A-Za-z0-9_]+")


@typechecked
def _split_segments(value: str) -> list[str]:
    """
    title: Split a string into LLVM-friendly segments.
    parameters:
      value:
        type: str
    returns:
      type: list[str]
    """
    segments = _SEGMENT_RE.findall(value)
    if segments:
        return segments
    if not value:
        return ["module"]
    return [f"x{ord(char):02x}" for char in value]


@typechecked
def _mangle_parts(*parts: str) -> str:
    """
    title: Mangle string parts into a deterministic LLVM name.
    parameters:
      parts:
        type: str
        variadic: positional
    returns:
      type: str
    """
    segments: list[str] = []
    for part in parts:
        segments.extend(_split_segments(part))
    return "__".join(segments)


@public
@typechecked
def function_key(module_key: ModuleKey, name: str) -> tuple[ModuleKey, str]:
    """
    title: Return a module-aware function registry key.
    parameters:
      module_key:
        type: ModuleKey
      name:
        type: str
    returns:
      type: tuple[ModuleKey, str]
    """
    return (module_key, name)


@public
@typechecked
def struct_key(module_key: ModuleKey, name: str) -> tuple[ModuleKey, str]:
    """
    title: Return a module-aware struct registry key.
    parameters:
      module_key:
        type: ModuleKey
      name:
        type: str
    returns:
      type: tuple[ModuleKey, str]
    """
    return (module_key, name)


@public
@typechecked
def qualified_function_name(module_key: ModuleKey, name: str) -> str:
    """
    title: Return a qualified semantic function name.
    parameters:
      module_key:
        type: ModuleKey
      name:
        type: str
    returns:
      type: str
    """
    return f"{module_key}::fn::{name}"


@public
@typechecked
def qualified_struct_name(module_key: ModuleKey, name: str) -> str:
    """
    title: Return a qualified semantic struct name.
    parameters:
      module_key:
        type: ModuleKey
      name:
        type: str
    returns:
      type: str
    """
    return f"{module_key}::struct::{name}"


@public
@typechecked
def qualified_local_name(
    module_key: ModuleKey,
    kind: str,
    name: str,
    symbol_id: str,
) -> str:
    """
    title: Return a qualified semantic local symbol name.
    parameters:
      module_key:
        type: ModuleKey
      kind:
        type: str
      name:
        type: str
      symbol_id:
        type: str
    returns:
      type: str
    """
    return f"{module_key}::{kind}::{name}::{symbol_id}"


@public
@typechecked
def mangle_function_name(module_key: ModuleKey, function_name: str) -> str:
    """
    title: Return a deterministic LLVM function name.
    parameters:
      module_key:
        type: ModuleKey
      function_name:
        type: str
    returns:
      type: str
    """
    return _mangle_parts(str(module_key), function_name)


@public
@typechecked
def mangle_struct_name(module_key: ModuleKey, struct_name: str) -> str:
    """
    title: Return a deterministic LLVM struct name.
    parameters:
      module_key:
        type: ModuleKey
      struct_name:
        type: str
    returns:
      type: str
    """
    return _mangle_parts(str(module_key), struct_name)

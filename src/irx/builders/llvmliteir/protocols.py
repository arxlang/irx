"""
title: Typing protocols for LLVMLiteIR visitors.
"""

from __future__ import annotations

from typing import Any, Protocol

import astx

from llvmlite import binding as llvm
from llvmlite import ir

from irx.builders.llvmliteir.types import VariablesLLVM


class LLVMLiteIRVisitorProtocol(Protocol):
    """
    title: Stable interface used by visitor mixins and runtime features.
    attributes:
      _llvm:
        type: VariablesLLVM
      named_values:
        type: dict[str, Any]
      const_vars:
        type: set[str]
      function_protos:
        type: dict[str, astx.FunctionPrototype]
      result_stack:
        type: list[ir.Value | ir.Function]
      target:
        type: llvm.TargetRef
      target_machine:
        type: llvm.TargetMachine
    """

    _llvm: VariablesLLVM
    named_values: dict[str, Any]
    const_vars: set[str]
    function_protos: dict[str, astx.FunctionPrototype]
    result_stack: list[ir.Value | ir.Function]
    target: llvm.TargetRef
    target_machine: llvm.TargetMachine

    def visit(self, node: astx.AST) -> None: ...

    def get_function(self, name: str) -> ir.Function | None: ...

    def create_entry_block_alloca(
        self, _var_name: str, _type_name: str
    ) -> Any: ...

    def require_runtime_symbol(
        self, _feature_name: str, _symbol_name: str
    ) -> ir.Function: ...

    def set_fast_math(self, _enabled: bool) -> None: ...

    def _apply_fast_math(self, _inst: ir.Instruction) -> None: ...

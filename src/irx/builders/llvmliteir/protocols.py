"""
title: Typing protocols for llvmliteir visitors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from llvmlite import binding as llvm
from llvmlite import ir

from irx import astx
from irx.base.visitors.protocols import BaseVisitorProtocol
from irx.builders.llvmliteir.state import NamedValueMap, ResultStackValue
from irx.builders.llvmliteir.types import VariablesLLVM


class VisitorProtocol(BaseVisitorProtocol, Protocol):
    """
    title: Stable interface used by visitor mixins and runtime features.
    attributes:
      _llvm:
        type: VariablesLLVM
      named_values:
        type: NamedValueMap
      const_vars:
        type: set[str]
      function_protos:
        type: dict[str, astx.FunctionPrototype]
      result_stack:
        type: list[ResultStackValue]
      loop_stack:
        type: list[dict[str, Any]]
      struct_types:
        type: dict[str, ir.Type]
      _fast_math_enabled:
        type: bool
      target:
        type: llvm.TargetRef
      target_machine:
        type: llvm.TargetMachine
    """

    _llvm: VariablesLLVM
    named_values: NamedValueMap
    const_vars: set[str]
    function_protos: dict[str, astx.FunctionPrototype]
    result_stack: list[ResultStackValue]
    loop_stack: list[dict[str, Any]]
    struct_types: dict[str, ir.Type]
    _fast_math_enabled: bool
    target: llvm.TargetRef
    target_machine: llvm.TargetMachine

    def get_function(self, _name: str) -> ir.Function | None: ...

    def create_entry_block_alloca(
        self, _var_name: str, _type_name: str
    ) -> Any: ...

    def require_runtime_symbol(
        self, _feature_name: str, _symbol_name: str
    ) -> ir.Function: ...

    def set_fast_math(self, _enabled: bool) -> None: ...

    def _apply_fast_math(self, _inst: ir.Instruction) -> None: ...

    def _emit_fma(
        self, _lhs: ir.Value, _rhs: ir.Value, _addend: ir.Value
    ) -> ir.Value: ...

    def _is_numeric_value(self, _value: ir.Value) -> bool: ...

    def _unify_numeric_operands(
        self,
        _lhs: ir.Value,
        _rhs: ir.Value,
        unsigned: bool = False,
    ) -> tuple[ir.Value, ir.Value]:
        _ = unsigned
        raise NotImplementedError

    def _try_set_binary_op(
        self,
        _lhs: ir.Value | None,
        _rhs: ir.Value | None,
        _op_code: str,
    ) -> bool: ...

    def _handle_string_concatenation(
        self, _lhs: ir.Value, _rhs: ir.Value
    ) -> ir.Value: ...

    def _handle_string_comparison(
        self, _lhs: ir.Value, _rhs: ir.Value, _op: str
    ) -> ir.Value: ...

    def _common_list_element_type(
        self, _lhs_ty: ir.Type, _rhs_ty: ir.Type
    ) -> ir.Type: ...

    def _coerce_to(
        self, _value: ir.Value, _target_ty: ir.Type
    ) -> ir.Value: ...

    def _mark_set_value(self, _value: ir.Value) -> ir.Value: ...

    def _subscript_uses_unsigned_semantics(
        self, _node: astx.SubscriptExpr
    ) -> bool: ...

    def _constant_subscript_key_matches(
        self, _entry_key: ir.Constant, _key_val: ir.Constant
    ) -> bool: ...

    def _emit_runtime_subscript_lookup(
        self,
        _dict_val: ir.Constant,
        _key_val: ir.Value,
        *,
        unsigned: bool,
    ) -> None:
        _ = unsigned
        raise NotImplementedError

    def _normalize_int_for_printf(
        self, _value: ir.Value
    ) -> tuple[ir.Value, str]: ...

    def _snprintf_heap(
        self, _fmt_gv: ir.GlobalVariable, _args: list[ir.Value]
    ) -> ir.Value: ...

    def _get_or_create_format_global(self, _fmt: str) -> ir.GlobalVariable: ...


if TYPE_CHECKING:

    class VisitorMixinBase:
        """
        title: Type-checking-only annotation carrier for llvmliteir mixins.
        attributes:
          _llvm:
            type: VariablesLLVM
          named_values:
            type: NamedValueMap
          const_vars:
            type: set[str]
          function_protos:
            type: dict[str, astx.FunctionPrototype]
          result_stack:
            type: list[ResultStackValue]
          loop_stack:
            type: list[dict[str, Any]]
          struct_types:
            type: dict[str, ir.Type]
          _fast_math_enabled:
            type: bool
          target:
            type: llvm.TargetRef
          target_machine:
            type: llvm.TargetMachine
        """

        _llvm: VariablesLLVM
        named_values: NamedValueMap
        const_vars: set[str]
        function_protos: dict[str, astx.FunctionPrototype]
        result_stack: list[ResultStackValue]
        loop_stack: list[dict[str, Any]]
        struct_types: dict[str, ir.Type]
        _fast_math_enabled: bool
        target: llvm.TargetRef
        target_machine: llvm.TargetMachine

        def visit(self, _node: astx.AST) -> None:
            raise NotImplementedError

        def visit_child(self, _node: astx.AST) -> None:
            raise NotImplementedError

        def get_function(self, _name: str) -> ir.Function | None:
            return cast(ir.Function | None, None)

        def create_entry_block_alloca(
            self, _var_name: str, _type_name: str
        ) -> Any:
            return cast(Any, None)

        def require_runtime_symbol(
            self, _feature_name: str, _symbol_name: str
        ) -> ir.Function:
            return cast(ir.Function, None)

        def set_fast_math(self, _enabled: bool) -> None:
            return None

        def _apply_fast_math(self, _inst: ir.Instruction) -> None:
            return None

        def _emit_fma(
            self, _lhs: ir.Value, _rhs: ir.Value, _addend: ir.Value
        ) -> ir.Value:
            return cast(ir.Value, None)

        def _is_numeric_value(self, _value: ir.Value) -> bool:
            return False

        def _unify_numeric_operands(
            self,
            _lhs: ir.Value,
            _rhs: ir.Value,
            unsigned: bool = False,
        ) -> tuple[ir.Value, ir.Value]:
            _ = unsigned
            return cast(tuple[ir.Value, ir.Value], (None, None))

        def _try_set_binary_op(
            self,
            _lhs: ir.Value | None,
            _rhs: ir.Value | None,
            _op_code: str,
        ) -> bool:
            return False

        def _handle_string_concatenation(
            self, _lhs: ir.Value, _rhs: ir.Value
        ) -> ir.Value:
            return cast(ir.Value, None)

        def _handle_string_comparison(
            self, _lhs: ir.Value, _rhs: ir.Value, _op: str
        ) -> ir.Value:
            return cast(ir.Value, None)

        def _common_list_element_type(
            self, _lhs_ty: ir.Type, _rhs_ty: ir.Type
        ) -> ir.Type:
            return cast(ir.Type, None)

        def _coerce_to(
            self, _value: ir.Value, _target_ty: ir.Type
        ) -> ir.Value:
            return cast(ir.Value, None)

        def _mark_set_value(self, _value: ir.Value) -> ir.Value:
            return cast(ir.Value, None)

        def _subscript_uses_unsigned_semantics(
            self, _node: astx.SubscriptExpr
        ) -> bool:
            return False

        def _constant_subscript_key_matches(
            self, _entry_key: ir.Constant, _key_val: ir.Constant
        ) -> bool:
            return False

        def _emit_runtime_subscript_lookup(
            self,
            _dict_val: ir.Constant,
            _key_val: ir.Value,
            *,
            unsigned: bool,
        ) -> None:
            _ = unsigned

        def _normalize_int_for_printf(
            self, _value: ir.Value
        ) -> tuple[ir.Value, str]:
            return cast(tuple[ir.Value, str], (None, ""))

        def _snprintf_heap(
            self, _fmt_gv: ir.GlobalVariable, _args: list[ir.Value]
        ) -> ir.Value:
            return cast(ir.Value, None)

        def _get_or_create_format_global(self, _fmt: str) -> ir.GlobalVariable:
            return cast(ir.GlobalVariable, None)

else:

    class VisitorMixinBase:
        """
        title: Runtime-empty base shared by llvmliteir visitor mixins.
        """

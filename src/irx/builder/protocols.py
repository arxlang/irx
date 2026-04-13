"""
title: Typing protocols for llvmliteir visitors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from llvmlite import binding as llvm
from llvmlite import ir

from irx import astx
from irx.analysis.resolved_nodes import FunctionSignature
from irx.base.visitors.protocols import BaseVisitorProtocol
from irx.builder.state import LoopTargets, NamedValueMap, ResultStackValue
from irx.builder.types import VariablesLLVM
from irx.typecheck import typechecked

if TYPE_CHECKING:
    from irx.builder.runtime.registry import RuntimeFeatureState


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
      llvm_functions_by_symbol_id:
        type: dict[str, ir.Function]
      result_stack:
        type: list[ResultStackValue]
      _buffer_view_global_counter:
        type: int
      loop_stack:
        type: list[LoopTargets]
      struct_types:
        type: dict[str, ir.Type]
      llvm_structs_by_qualified_name:
        type: dict[str, ir.IdentifiedStructType]
      runtime_features:
        type: RuntimeFeatureState
      entry_function_symbol_id:
        type: str | None
      _fast_math_enabled:
        type: bool
      _current_function_return_type:
        type: astx.DataType | None
      _current_function_signature:
        type: FunctionSignature | None
      target:
        type: llvm.TargetRef
      target_machine:
        type: llvm.TargetMachine
    """

    _llvm: VariablesLLVM
    named_values: NamedValueMap
    const_vars: set[str]
    function_protos: dict[str, astx.FunctionPrototype]
    llvm_functions_by_symbol_id: dict[str, ir.Function]
    result_stack: list[ResultStackValue]
    _buffer_view_global_counter: int
    loop_stack: list[LoopTargets]
    struct_types: dict[str, ir.Type]
    llvm_structs_by_qualified_name: dict[str, ir.IdentifiedStructType]
    runtime_features: RuntimeFeatureState
    entry_function_symbol_id: str | None
    _fast_math_enabled: bool
    _current_function_return_type: astx.DataType | None
    _current_function_signature: FunctionSignature | None
    target: llvm.TargetRef
    target_machine: llvm.TargetMachine

    def get_function(self, _name: str) -> ir.Function | None:
        """
        title: Get function.
        parameters:
          _name:
            type: str
        returns:
          type: ir.Function | None
        """
        ...

    def llvm_function_name_for_node(
        self,
        _node: astx.AST,
        _fallback: str,
    ) -> str:
        """
        title: Return the LLVM symbol name for a function node.
        parameters:
          _node:
            type: astx.AST
          _fallback:
            type: str
        returns:
          type: str
        """
        ...

    def create_entry_block_alloca(
        self, _var_name: str, _type_name: str | ir.Type
    ) -> Any:
        """
        title: Create entry block alloca.
        parameters:
          _var_name:
            type: str
          _type_name:
            type: str | ir.Type
        returns:
          type: Any
        """
        ...

    def _field_address(self, _node: astx.FieldAccess) -> ir.Value:
        """
        title: Lower one field access to an address.
        parameters:
          _node:
            type: astx.FieldAccess
        returns:
          type: ir.Value
        """
        ...

    def require_runtime_symbol(
        self, _feature_name: str, _symbol_name: str
    ) -> ir.Function:
        """
        title: Require runtime symbol.
        parameters:
          _feature_name:
            type: str
          _symbol_name:
            type: str
        returns:
          type: ir.Function
        """
        ...

    def activate_runtime_feature(self, _feature_name: str) -> None:
        """
        title: Activate runtime feature.
        parameters:
          _feature_name:
            type: str
        """
        ...

    def set_fast_math(self, _enabled: bool) -> None:
        """
        title: Set fast math.
        parameters:
          _enabled:
            type: bool
        """
        ...

    def _apply_fast_math(self, _inst: ir.Instruction) -> None:
        """
        title: Apply fast math.
        parameters:
          _inst:
            type: ir.Instruction
        """
        ...

    def _emit_fma(
        self, _lhs: ir.Value, _rhs: ir.Value, _addend: ir.Value
    ) -> ir.Value:
        """
        title: Emit fma.
        parameters:
          _lhs:
            type: ir.Value
          _rhs:
            type: ir.Value
          _addend:
            type: ir.Value
        returns:
          type: ir.Value
        """
        ...

    def _is_numeric_value(self, _value: ir.Value) -> bool:
        """
        title: Is numeric value.
        parameters:
          _value:
            type: ir.Value
        returns:
          type: bool
        """
        ...

    def _unify_numeric_operands(
        self,
        _lhs: ir.Value,
        _rhs: ir.Value,
        unsigned: bool = False,
    ) -> tuple[ir.Value, ir.Value]:
        """
        title: Unify numeric operands.
        parameters:
          _lhs:
            type: ir.Value
          _rhs:
            type: ir.Value
          unsigned:
            type: bool
        returns:
          type: tuple[ir.Value, ir.Value]
        """
        _ = unsigned
        raise NotImplementedError

    def _try_set_binary_op(
        self,
        _lhs: ir.Value | None,
        _rhs: ir.Value | None,
        _op_code: str,
    ) -> bool:
        """
        title: Try set binary op.
        parameters:
          _lhs:
            type: ir.Value | None
          _rhs:
            type: ir.Value | None
          _op_code:
            type: str
        returns:
          type: bool
        """
        ...

    def _handle_string_concatenation(
        self, _lhs: ir.Value, _rhs: ir.Value
    ) -> ir.Value:
        """
        title: Handle string concatenation.
        parameters:
          _lhs:
            type: ir.Value
          _rhs:
            type: ir.Value
        returns:
          type: ir.Value
        """
        ...

    def _handle_string_comparison(
        self, _lhs: ir.Value, _rhs: ir.Value, _op: str
    ) -> ir.Value:
        """
        title: Handle string comparison.
        parameters:
          _lhs:
            type: ir.Value
          _rhs:
            type: ir.Value
          _op:
            type: str
        returns:
          type: ir.Value
        """
        ...

    def _common_list_element_type(
        self, _lhs_ty: ir.Type, _rhs_ty: ir.Type
    ) -> ir.Type:
        """
        title: Common list element type.
        parameters:
          _lhs_ty:
            type: ir.Type
          _rhs_ty:
            type: ir.Type
        returns:
          type: ir.Type
        """
        ...

    def _coerce_to(self, _value: ir.Value, _target_ty: ir.Type) -> ir.Value:
        """
        title: Coerce to.
        parameters:
          _value:
            type: ir.Value
          _target_ty:
            type: ir.Type
        returns:
          type: ir.Value
        """
        ...

    def _mark_set_value(self, _value: ir.Value) -> ir.Value:
        """
        title: Mark set value.
        parameters:
          _value:
            type: ir.Value
        returns:
          type: ir.Value
        """
        ...

    def _subscript_uses_unsigned_semantics(
        self, _node: astx.SubscriptExpr
    ) -> bool:
        """
        title: Subscript uses unsigned semantics.
        parameters:
          _node:
            type: astx.SubscriptExpr
        returns:
          type: bool
        """
        ...

    def _constant_subscript_key_matches(
        self, _entry_key: ir.Constant, _key_val: ir.Constant
    ) -> bool:
        """
        title: Constant subscript key matches.
        parameters:
          _entry_key:
            type: ir.Constant
          _key_val:
            type: ir.Constant
        returns:
          type: bool
        """
        ...

    def _emit_runtime_subscript_lookup(
        self,
        _dict_val: ir.Constant,
        _key_val: ir.Value,
        *,
        unsigned: bool,
    ) -> None:
        """
        title: Emit runtime subscript lookup.
        parameters:
          _dict_val:
            type: ir.Constant
          _key_val:
            type: ir.Value
          unsigned:
            type: bool
        """
        _ = unsigned
        raise NotImplementedError

    def _normalize_int_for_printf(
        self,
        _value: ir.Value,
        *,
        unsigned: bool = False,
    ) -> tuple[ir.Value, str]:
        """
        title: Normalize int for printf.
        parameters:
          _value:
            type: ir.Value
          unsigned:
            type: bool
        returns:
          type: tuple[ir.Value, str]
        """
        _ = unsigned
        return cast(tuple[ir.Value, str], (None, ""))

    def _snprintf_heap(
        self, _fmt_gv: ir.GlobalVariable, _args: list[ir.Value]
    ) -> ir.Value:
        """
        title: Snprintf heap.
        parameters:
          _fmt_gv:
            type: ir.GlobalVariable
          _args:
            type: list[ir.Value]
        returns:
          type: ir.Value
        """
        ...

    def _get_or_create_format_global(self, _fmt: str) -> ir.GlobalVariable:
        """
        title: Get or create format global.
        parameters:
          _fmt:
            type: str
        returns:
          type: ir.GlobalVariable
        """
        ...


@typechecked
class VisitorMixinTypingBase:
    """
    title: Type-checking-only annotation carrier for llvmliteir visitor mixins.
    attributes:
      _llvm:
        type: VariablesLLVM
      named_values:
        type: NamedValueMap
      const_vars:
        type: set[str]
      function_protos:
        type: dict[str, astx.FunctionPrototype]
      llvm_functions_by_symbol_id:
        type: dict[str, ir.Function]
      result_stack:
        type: list[ResultStackValue]
      _buffer_view_global_counter:
        type: int
      loop_stack:
        type: list[LoopTargets]
      struct_types:
        type: dict[str, ir.Type]
      llvm_structs_by_qualified_name:
        type: dict[str, ir.IdentifiedStructType]
      runtime_features:
        type: RuntimeFeatureState
      entry_function_symbol_id:
        type: str | None
      _fast_math_enabled:
        type: bool
      _current_function_return_type:
        type: astx.DataType | None
      _current_function_signature:
        type: FunctionSignature | None
      target:
        type: llvm.TargetRef
      target_machine:
        type: llvm.TargetMachine
    """

    _llvm: VariablesLLVM
    named_values: NamedValueMap
    const_vars: set[str]
    function_protos: dict[str, astx.FunctionPrototype]
    llvm_functions_by_symbol_id: dict[str, ir.Function]
    result_stack: list[ResultStackValue]
    _buffer_view_global_counter: int
    loop_stack: list[LoopTargets]
    struct_types: dict[str, ir.Type]
    llvm_structs_by_qualified_name: dict[str, ir.IdentifiedStructType]
    runtime_features: RuntimeFeatureState
    entry_function_symbol_id: str | None
    _fast_math_enabled: bool
    _current_function_return_type: astx.DataType | None
    _current_function_signature: FunctionSignature | None
    target: llvm.TargetRef
    target_machine: llvm.TargetMachine

    def visit(self, _node: astx.AST) -> None:
        """
        title: Visit AST nodes.
        parameters:
          _node:
            type: astx.AST
        """
        raise NotImplementedError

    def visit_child(self, _node: astx.AST) -> None:
        """
        title: Visit one child AST node.
        parameters:
          _node:
            type: astx.AST
        """
        raise NotImplementedError

    def get_function(self, _name: str) -> ir.Function | None:
        """
        title: Get function.
        parameters:
          _name:
            type: str
        returns:
          type: ir.Function | None
        """
        return cast(ir.Function | None, None)

    def llvm_function_name_for_node(
        self,
        _node: astx.AST,
        _fallback: str,
    ) -> str:
        """
        title: Return the LLVM symbol name for a function node.
        parameters:
          _node:
            type: astx.AST
          _fallback:
            type: str
        returns:
          type: str
        """
        return _fallback

    def create_entry_block_alloca(
        self, _var_name: str, _type_name: str | ir.Type
    ) -> Any:
        """
        title: Create entry block alloca.
        parameters:
          _var_name:
            type: str
          _type_name:
            type: str | ir.Type
        returns:
          type: Any
        """
        return cast(Any, None)

    def _field_address(self, _node: astx.FieldAccess) -> ir.Value:
        """
        title: Lower one field access to an address.
        parameters:
          _node:
            type: astx.FieldAccess
        returns:
          type: ir.Value
        """
        return cast(ir.Value, None)

    def require_runtime_symbol(
        self, _feature_name: str, _symbol_name: str
    ) -> ir.Function:
        """
        title: Require runtime symbol.
        parameters:
          _feature_name:
            type: str
          _symbol_name:
            type: str
        returns:
          type: ir.Function
        """
        return cast(ir.Function, None)

    def activate_runtime_feature(self, _feature_name: str) -> None:
        """
        title: Activate runtime feature.
        parameters:
          _feature_name:
            type: str
        """
        return None

    def set_fast_math(self, _enabled: bool) -> None:
        """
        title: Set fast math.
        parameters:
          _enabled:
            type: bool
        """
        return None

    def _apply_fast_math(self, _inst: ir.Instruction) -> None:
        """
        title: Apply fast math.
        parameters:
          _inst:
            type: ir.Instruction
        """
        return None

    def _emit_fma(
        self, _lhs: ir.Value, _rhs: ir.Value, _addend: ir.Value
    ) -> ir.Value:
        """
        title: Emit fma.
        parameters:
          _lhs:
            type: ir.Value
          _rhs:
            type: ir.Value
          _addend:
            type: ir.Value
        returns:
          type: ir.Value
        """
        return cast(ir.Value, None)

    def _is_numeric_value(self, _value: ir.Value) -> bool:
        """
        title: Is numeric value.
        parameters:
          _value:
            type: ir.Value
        returns:
          type: bool
        """
        return False

    def _llvm_type_for_ast_type(
        self,
        _type: astx.DataType | None,
    ) -> ir.Type | None:
        """
        title: LLVM type for AST type.
        parameters:
          _type:
            type: astx.DataType | None
        returns:
          type: ir.Type | None
        """
        return cast(ir.Type | None, None)

    def _resolved_ast_type(
        self, _node: astx.AST | None
    ) -> astx.DataType | None:
        """
        title: Resolved ast type.
        parameters:
          _node:
            type: astx.AST | None
        returns:
          type: astx.DataType | None
        """
        return cast(astx.DataType | None, None)

    def _cast_ast_value(
        self,
        _value: ir.Value,
        *,
        source_type: astx.DataType | None,
        target_type: astx.DataType | None,
    ) -> ir.Value:
        """
        title: Cast ast value.
        parameters:
          _value:
            type: ir.Value
          source_type:
            type: astx.DataType | None
          target_type:
            type: astx.DataType | None
        returns:
          type: ir.Value
        """
        _ = source_type
        _ = target_type
        return cast(ir.Value, None)

    def _coerce_numeric_operands_for_types(
        self,
        _lhs: ir.Value,
        _rhs: ir.Value,
        *,
        lhs_type: astx.DataType | None,
        rhs_type: astx.DataType | None,
    ) -> tuple[ir.Value, ir.Value]:
        """
        title: Coerce numeric operands for types.
        parameters:
          _lhs:
            type: ir.Value
          _rhs:
            type: ir.Value
          lhs_type:
            type: astx.DataType | None
          rhs_type:
            type: astx.DataType | None
        returns:
          type: tuple[ir.Value, ir.Value]
        """
        _ = lhs_type
        _ = rhs_type
        return cast(tuple[ir.Value, ir.Value], (None, None))

    def _emit_numeric_compare(
        self,
        _op_code: str,
        _lhs: ir.Value,
        _rhs: ir.Value,
        *,
        unsigned: bool,
        name: str,
    ) -> ir.Value:
        """
        title: Emit numeric compare.
        parameters:
          _op_code:
            type: str
          _lhs:
            type: ir.Value
          _rhs:
            type: ir.Value
          unsigned:
            type: bool
          name:
            type: str
        returns:
          type: ir.Value
        """
        _ = unsigned
        _ = name
        return cast(ir.Value, None)

    def _unify_numeric_operands(
        self,
        _lhs: ir.Value,
        _rhs: ir.Value,
        unsigned: bool = False,
    ) -> tuple[ir.Value, ir.Value]:
        """
        title: Unify numeric operands.
        parameters:
          _lhs:
            type: ir.Value
          _rhs:
            type: ir.Value
          unsigned:
            type: bool
        returns:
          type: tuple[ir.Value, ir.Value]
        """
        _ = unsigned
        return cast(tuple[ir.Value, ir.Value], (None, None))

    def _try_set_binary_op(
        self,
        _lhs: ir.Value | None,
        _rhs: ir.Value | None,
        _op_code: str,
    ) -> bool:
        """
        title: Try set binary op.
        parameters:
          _lhs:
            type: ir.Value | None
          _rhs:
            type: ir.Value | None
          _op_code:
            type: str
        returns:
          type: bool
        """
        return False

    def _handle_string_concatenation(
        self, _lhs: ir.Value, _rhs: ir.Value
    ) -> ir.Value:
        """
        title: Handle string concatenation.
        parameters:
          _lhs:
            type: ir.Value
          _rhs:
            type: ir.Value
        returns:
          type: ir.Value
        """
        return cast(ir.Value, None)

    def _handle_string_comparison(
        self, _lhs: ir.Value, _rhs: ir.Value, _op: str
    ) -> ir.Value:
        """
        title: Handle string comparison.
        parameters:
          _lhs:
            type: ir.Value
          _rhs:
            type: ir.Value
          _op:
            type: str
        returns:
          type: ir.Value
        """
        return cast(ir.Value, None)

    def _common_list_element_type(
        self, _lhs_ty: ir.Type, _rhs_ty: ir.Type
    ) -> ir.Type:
        """
        title: Common list element type.
        parameters:
          _lhs_ty:
            type: ir.Type
          _rhs_ty:
            type: ir.Type
        returns:
          type: ir.Type
        """
        return cast(ir.Type, None)

    def _coerce_to(self, _value: ir.Value, _target_ty: ir.Type) -> ir.Value:
        """
        title: Coerce to.
        parameters:
          _value:
            type: ir.Value
          _target_ty:
            type: ir.Type
        returns:
          type: ir.Value
        """
        return cast(ir.Value, None)

    def _mark_set_value(self, _value: ir.Value) -> ir.Value:
        """
        title: Mark set value.
        parameters:
          _value:
            type: ir.Value
        returns:
          type: ir.Value
        """
        return cast(ir.Value, None)

    def _subscript_uses_unsigned_semantics(
        self, _node: astx.SubscriptExpr
    ) -> bool:
        """
        title: Subscript uses unsigned semantics.
        parameters:
          _node:
            type: astx.SubscriptExpr
        returns:
          type: bool
        """
        return False

    def _constant_subscript_key_matches(
        self, _entry_key: ir.Constant, _key_val: ir.Constant
    ) -> bool:
        """
        title: Constant subscript key matches.
        parameters:
          _entry_key:
            type: ir.Constant
          _key_val:
            type: ir.Constant
        returns:
          type: bool
        """
        return False

    def _emit_runtime_subscript_lookup(
        self,
        _dict_val: ir.Constant,
        _key_val: ir.Value,
        *,
        unsigned: bool,
    ) -> None:
        """
        title: Emit runtime subscript lookup.
        parameters:
          _dict_val:
            type: ir.Constant
          _key_val:
            type: ir.Value
          unsigned:
            type: bool
        """
        _ = unsigned

    def _normalize_int_for_printf(
        self,
        _value: ir.Value,
        *,
        unsigned: bool = False,
    ) -> tuple[ir.Value, str]:
        """
        title: Normalize int for printf.
        parameters:
          _value:
            type: ir.Value
          unsigned:
            type: bool
        returns:
          type: tuple[ir.Value, str]
        """
        _ = unsigned
        return cast(tuple[ir.Value, str], (None, ""))

    def _snprintf_heap(
        self, _fmt_gv: ir.GlobalVariable, _args: list[ir.Value]
    ) -> ir.Value:
        """
        title: Snprintf heap.
        parameters:
          _fmt_gv:
            type: ir.GlobalVariable
          _args:
            type: list[ir.Value]
        returns:
          type: ir.Value
        """
        return cast(ir.Value, None)

    def _get_or_create_format_global(self, _fmt: str) -> ir.GlobalVariable:
        """
        title: Get or create format global.
        parameters:
          _fmt:
            type: str
        returns:
          type: ir.GlobalVariable
        """
        return cast(ir.GlobalVariable, None)


@typechecked
class VisitorMixinRuntimeBase:
    """
    title: Runtime-empty base shared by llvmliteir visitor mixins.
    """


if TYPE_CHECKING:
    VisitorMixinBase = VisitorMixinTypingBase
else:
    VisitorMixinBase = VisitorMixinRuntimeBase

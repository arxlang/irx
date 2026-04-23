# mypy: disable-error-code=no-redef

"""
title: Dynamic-list visitor mixins for llvmliteir.
"""

from __future__ import annotations

from typing import cast

from llvmlite import ir

from irx import astx
from irx.builder.core import VisitorCore
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builtins.collections.list import (
    LIST_APPEND_SYMBOL,
    LIST_AT_SYMBOL,
    LIST_RUNTIME_FEATURE,
    list_element_type,
)
from irx.typecheck import typechecked


@typechecked
class ListVisitorMixin(VisitorMixinBase):
    """
    title: Dynamic-list visitor mixin.
    """

    def _llvm_list_type(self) -> ir.Type:
        """
        title: Return the canonical lowered list value type.
        returns:
          type: ir.Type
        """
        return cast(
            ir.Type,
            ir.LiteralStructType(
                [
                    self._llvm.INT8_TYPE.as_pointer(),
                    self._llvm.INT64_TYPE,
                    self._llvm.INT64_TYPE,
                    self._llvm.INT64_TYPE,
                ]
            ),
        )

    def _list_element_llvm_type_from_type(
        self,
        type_: astx.DataType | None,
    ) -> ir.Type:
        """
        title: Return the lowered LLVM type for one concrete list element type.
        parameters:
          type_:
            type: astx.DataType | None
        returns:
          type: ir.Type
        """
        element_type = list_element_type(type_)
        if element_type is None:
            raise Exception(
                "dynamic list lowering requires a single concrete element type"
            )
        llvm_type = self._llvm_type_for_ast_type(element_type)
        if llvm_type is None:
            raise Exception(
                "dynamic list lowering requires a lowerable element type"
            )
        return llvm_type

    def _list_element_llvm_type(self, node: astx.AST) -> ir.Type:
        """
        title: Return the lowered element type for one list-valued AST node.
        parameters:
          node:
            type: astx.AST
        returns:
          type: ir.Type
        """
        return self._list_element_llvm_type_from_type(
            self._resolved_ast_type(node)
        )

    def _list_element_size_from_type(
        self,
        type_: astx.DataType | None,
    ) -> int:
        """
        title: Return the ABI size in bytes for one concrete list type.
        parameters:
          type_:
            type: astx.DataType | None
        returns:
          type: int
        """
        llvm_type = self._list_element_llvm_type_from_type(type_)
        return cast(
            int,
            llvm_type.get_abi_size(self.target_machine.target_data),
        )

    def _list_element_size(self, node: astx.AST) -> int:
        """
        title: Return the ABI size in bytes for one list element type.
        parameters:
          node:
            type: astx.AST
        returns:
          type: int
        """
        return self._list_element_size_from_type(self._resolved_ast_type(node))

    def _empty_list_value_for_type(
        self,
        type_: astx.DataType | None,
    ) -> ir.Constant:
        """
        title: Return one zero-length lowered list value for one list type.
        parameters:
          type_:
            type: astx.DataType | None
        returns:
          type: ir.Constant
        """
        return ir.Constant(
            cast(ir.LiteralStructType, self._llvm_list_type()),
            [
                ir.Constant(self._llvm.INT8_TYPE.as_pointer(), None),
                ir.Constant(self._llvm.INT64_TYPE, 0),
                ir.Constant(self._llvm.INT64_TYPE, 0),
                ir.Constant(
                    self._llvm.INT64_TYPE,
                    self._list_element_size_from_type(type_),
                ),
            ],
        )

    def _empty_list_value(self, node: astx.AST) -> ir.Constant:
        """
        title: Return one zero-length lowered list value.
        parameters:
          node:
            type: astx.AST
        returns:
          type: ir.Constant
        """
        return self._empty_list_value_for_type(self._resolved_ast_type(node))

    def _list_pointer_for_call(self, node: astx.AST, *, name: str) -> ir.Value:
        """
        title: Return an addressable lowered list value for runtime calls.
        parameters:
          node:
            type: astx.AST
          name:
            type: str
        returns:
          type: ir.Value
        """
        if isinstance(
            node,
            (
                astx.Identifier,
                astx.FieldAccess,
                astx.BaseFieldAccess,
                astx.StaticFieldAccess,
            ),
        ):
            return self._lvalue_address(node)

        self.visit_child(node)
        value = safe_pop(self.result_stack)
        llvm_list_type = self._llvm_list_type()
        if value is None or value.type != llvm_list_type:
            raise Exception("dynamic list lowering requires a list value")
        temp = self._llvm.ir_builder.alloca(llvm_list_type, name=name)
        self._llvm.ir_builder.store(value, temp)
        return temp

    def _lower_list_subscript(self, node: astx.SubscriptExpr) -> None:
        """
        title: Lower one list indexing operation.
        parameters:
          node:
            type: astx.SubscriptExpr
        """
        if list_element_type(self._resolved_ast_type(node.value)) is None:
            raise Exception(
                "dynamic list subscript requires a single concrete element "
                "type"
            )

        at_fn = self.require_runtime_symbol(
            LIST_RUNTIME_FEATURE,
            LIST_AT_SYMBOL,
        )
        list_ptr = self._list_pointer_for_call(
            node.value,
            name="irx_list_index_value",
        )

        self.visit_child(node.index)
        index = safe_pop(self.result_stack)
        if index is None:
            raise Exception("dynamic list subscript requires an index value")
        index = self._cast_ast_value(
            index,
            source_type=self._resolved_ast_type(node.index),
            target_type=astx.Int64(),
        )

        raw_ptr = self._llvm.ir_builder.call(
            at_fn,
            [list_ptr, index],
            name="irx_list_at_ptr",
        )
        llvm_element_type = self._list_element_llvm_type(node.value)
        typed_ptr = self._llvm.ir_builder.bitcast(
            raw_ptr,
            llvm_element_type.as_pointer(),
            name="irx_list_at_typed_ptr",
        )
        self.result_stack.append(
            self._llvm.ir_builder.load(
                typed_ptr,
                name="irx_list_at_load",
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.ListCreate) -> None:
        """
        title: Visit ListCreate nodes.
        parameters:
          node:
            type: astx.ListCreate
        """
        self.result_stack.append(self._empty_list_value(node))

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.ListAppend) -> None:
        """
        title: Visit ListAppend nodes.
        parameters:
          node:
            type: astx.ListAppend
        """
        append_fn = self.require_runtime_symbol(
            LIST_RUNTIME_FEATURE,
            LIST_APPEND_SYMBOL,
        )
        list_ptr = self._lvalue_address(node.base)

        self.visit_child(node.value)
        value = safe_pop(self.result_stack)
        if value is None:
            raise Exception("dynamic list append requires a value")

        element_type = list_element_type(self._resolved_ast_type(node.base))
        if element_type is None:
            raise Exception(
                "dynamic list append requires a single concrete element type"
            )
        value = self._cast_ast_value(
            value,
            source_type=self._resolved_ast_type(node.value),
            target_type=element_type,
        )

        llvm_element_type = self._list_element_llvm_type(node.base)
        value_ptr = self._llvm.ir_builder.alloca(
            llvm_element_type,
            name="irx_list_append_value",
        )
        self._llvm.ir_builder.store(value, value_ptr)
        raw_value_ptr = self._llvm.ir_builder.bitcast(
            value_ptr,
            self._llvm.INT8_TYPE.as_pointer(),
            name="irx_list_append_bytes",
        )

        result = self._llvm.ir_builder.call(
            append_fn,
            [list_ptr, raw_value_ptr],
            name="irx_list_append_status",
        )
        self.result_stack.append(result)

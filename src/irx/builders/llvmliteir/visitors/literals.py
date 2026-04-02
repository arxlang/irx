# mypy: disable-error-code=no-redef

"""
title: Literal visitor mixins for llvmliteir.
"""

from __future__ import annotations

from typing import Any

import astx

from llvmlite import ir

from irx.builders.base import BuilderVisitor
from irx.builders.llvmliteir.protocols import VisitorMixinBase


class LiteralVisitorMixin(VisitorMixinBase):
    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralInt32) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.INT32_TYPE, node.value)
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, expr: astx.LiteralFloat32) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.FLOAT_TYPE, expr.value)
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, expr: astx.LiteralFloat64) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.DOUBLE_TYPE, expr.value)
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralFloat16) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.FLOAT16_TYPE, node.value)
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, expr: astx.LiteralNone) -> None:
        self.result_stack.append(None)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralBoolean) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.BOOLEAN_TYPE, int(node.value))
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralInt64) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.INT64_TYPE, node.value)
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralInt8) -> None:
        self.result_stack.append(ir.Constant(self._llvm.INT8_TYPE, node.value))

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralUInt8) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.UINT8_TYPE, node.value)
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralUInt16) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.UINT16_TYPE, node.value)
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralUInt32) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.UINT32_TYPE, node.value)
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralUInt64) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.UINT64_TYPE, node.value)
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralUInt128) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.UINT128_TYPE, node.value)
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, expr: astx.LiteralUTF8Char) -> None:
        string_value = expr.value
        utf8_bytes = string_value.encode("utf-8")
        string_length = len(utf8_bytes)

        string_data_type = ir.ArrayType(
            self._llvm.INT8_TYPE, string_length + 1
        )
        string_data = ir.GlobalVariable(
            self._llvm.module,
            string_data_type,
            name=f"str_ascii_{id(expr)}",
        )
        string_data.linkage = "internal"
        string_data.global_constant = True
        string_data.initializer = ir.Constant(
            string_data_type, bytearray(utf8_bytes + b"\0")
        )

        ptr = self._llvm.ir_builder.gep(
            string_data,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)],
            inbounds=True,
        )
        self.result_stack.append(ptr)

    @BuilderVisitor.visit.dispatch
    def visit(self, expr: astx.LiteralUTF8String) -> None:
        string_value = expr.value
        utf8_bytes = string_value.encode("utf-8")
        string_length = len(utf8_bytes)

        string_data_type = ir.ArrayType(
            self._llvm.INT8_TYPE, string_length + 1
        )
        unique_name = f"str_utf8_{abs(hash(string_value))}_{id(expr)}"
        string_data = ir.GlobalVariable(
            self._llvm.module, string_data_type, name=unique_name
        )
        string_data.linkage = "internal"
        string_data.global_constant = True
        string_data.initializer = ir.Constant(
            string_data_type, bytearray(utf8_bytes + b"\0")
        )

        data_ptr = self._llvm.ir_builder.gep(
            string_data,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)],
            inbounds=True,
        )
        self.result_stack.append(data_ptr)

    @BuilderVisitor.visit.dispatch
    def visit(self, expr: astx.LiteralString) -> None:
        self.visit_child(astx.LiteralUTF8String(value=expr.value))

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralList) -> None:
        llvm_elems: list[ir.Value] = []
        for elem in node.elements:
            self.visit_child(elem)
            value = self.result_stack.pop()
            if value is None:
                raise Exception("LiteralList: invalid element lowering.")
            llvm_elems.append(value)

        count = len(llvm_elems)
        if count == 0:
            empty_ty = ir.ArrayType(self._llvm.INT32_TYPE, 0)
            self.result_stack.append(ir.Constant(empty_ty, []))
            return

        target_ty = llvm_elems[0].type
        for value in llvm_elems[1:]:
            target_ty = self._common_list_element_type(target_ty, value.type)

        coerced = [self._coerce_to(value, target_ty) for value in llvm_elems]
        if all(isinstance(value, ir.Constant) for value in coerced):
            arr_ty = ir.ArrayType(target_ty, count)
            self.result_stack.append(ir.Constant(arr_ty, coerced))
            return

        arr_ty = ir.ArrayType(target_ty, count)
        alloca = self._llvm.ir_builder.alloca(arr_ty, name="list_tmp")
        zero = ir.Constant(self._llvm.INT32_TYPE, 0)
        for index, value in enumerate(coerced):
            idx = ir.Constant(self._llvm.INT32_TYPE, index)
            slot = self._llvm.ir_builder.gep(
                alloca, [zero, idx], inbounds=True
            )
            self._llvm.ir_builder.store(value, slot)

        first_ptr = self._llvm.ir_builder.gep(
            alloca, [zero, zero], inbounds=True
        )
        self.result_stack.append(first_ptr)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralSet) -> None:
        def sort_key(lit: astx.Literal) -> tuple[str, Any]:
            type_name = type(lit).__name__
            value = getattr(lit, "value", None)
            comparable = (
                value if isinstance(value, (int, float, str)) else repr(lit)
            )
            return type_name, comparable

        elems_sorted = sorted(node.elements, key=sort_key)
        llvm_elems: list[ir.Value] = []
        for elem in elems_sorted:
            self.visit_child(elem)
            value = self.result_stack.pop()
            if value is None:
                raise Exception("LiteralSet: invalid element lowering.")
            llvm_elems.append(value)

        count = len(llvm_elems)
        if count == 0:
            empty_ty = ir.ArrayType(self._llvm.INT32_TYPE, 0)
            self.result_stack.append(
                self._mark_set_value(ir.Constant(empty_ty, []))
            )
            return

        is_ints = all(
            isinstance(value.type, ir.IntType) for value in llvm_elems
        )
        all_constants = all(
            isinstance(value, ir.Constant) for value in llvm_elems
        )
        if is_ints and all_constants:
            widest = max(value.type.width for value in llvm_elems)
            elem_ty = ir.IntType(widest)
            arr_ty = ir.ArrayType(elem_ty, count)
            promoted_vals: list[ir.Constant] = []
            for value in llvm_elems:
                if value.type.width != widest:
                    promoted_vals.append(ir.Constant(elem_ty, value.constant))
                else:
                    promoted_vals.append(value)

            const_arr = self._mark_set_value(
                ir.Constant(arr_ty, promoted_vals)
            )
            self.result_stack.append(const_arr)
            return

        raise TypeError(
            "LiteralSet: only integer constants are currently supported "
            "(homogeneous or mixed-width)"
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralTuple) -> None:
        llvm_elems: list[ir.Value] = []
        for elem in node.elements:
            self.visit_child(elem)
            value = self.result_stack.pop()
            if value is None:
                raise Exception("LiteralTuple: invalid element lowering.")
            llvm_elems.append(value)

        count = len(llvm_elems)
        if count == 0:
            struct_ty = ir.LiteralStructType([])
            self.result_stack.append(ir.Constant(struct_ty, []))
            return

        first_ty = llvm_elems[0].type
        homogeneous = all(value.type == first_ty for value in llvm_elems)
        all_constants = all(
            isinstance(value, ir.Constant) for value in llvm_elems
        )
        if homogeneous and all_constants:
            struct_ty = ir.LiteralStructType([first_ty] * count)
            self.result_stack.append(ir.Constant(struct_ty, llvm_elems))
            return

        raise TypeError(
            "LiteralTuple: only empty or homogeneous constant tuples "
            "are supported"
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralDict) -> None:
        llvm_pairs: list[tuple[ir.Value, ir.Value]] = []
        for key_node, value_node in node.elements.items():
            self.visit_child(key_node)
            key_val = self.result_stack.pop()
            if key_val is None:
                raise Exception("LiteralDict: failed to lower key.")

            self.visit_child(value_node)
            val_val = self.result_stack.pop()
            if val_val is None:
                raise Exception("LiteralDict: failed to lower value.")

            llvm_pairs.append((key_val, val_val))

        count = len(llvm_pairs)
        if count == 0:
            pair_ty = ir.LiteralStructType(
                [self._llvm.INT32_TYPE, self._llvm.INT32_TYPE]
            )
            arr_ty = ir.ArrayType(pair_ty, 0)
            self.result_stack.append(ir.Constant(arr_ty, []))
            return

        all_constants = all(
            isinstance(key, ir.Constant) and isinstance(value, ir.Constant)
            for key, value in llvm_pairs
        )
        if all_constants:
            first_key_ty = llvm_pairs[0][0].type
            first_val_ty = llvm_pairs[0][1].type
            pair_ty = ir.LiteralStructType([first_key_ty, first_val_ty])
            arr_ty = ir.ArrayType(pair_ty, count)

            struct_consts: list[ir.Constant] = []
            for key_val, val_val in llvm_pairs:
                if (
                    key_val.type != first_key_ty
                    or val_val.type != first_val_ty
                ):
                    raise TypeError(
                        "LiteralDict: heterogeneous constant key/value types "
                        "are not yet supported"
                    )
                struct_consts.append(ir.Constant(pair_ty, [key_val, val_val]))

            self.result_stack.append(ir.Constant(arr_ty, struct_consts))
            return

        raise TypeError(
            "LiteralDict: only empty or all-constant dictionaries "
            "are supported in this version"
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.SubscriptExpr) -> None:
        dict_pair_fields = 2
        self.visit_child(node.value)
        dict_val = self.result_stack.pop()

        if not (
            isinstance(dict_val, ir.Constant)
            and isinstance(dict_val.type, ir.ArrayType)
            and isinstance(dict_val.type.element, ir.LiteralStructType)
            and len(dict_val.type.element.elements) == dict_pair_fields
        ):
            raise TypeError(
                "SubscriptExpr: only constant LiteralDict subscript "
                "is supported in this version"
            )

        self.visit_child(node.index)
        key_val = self.result_stack.pop()
        if key_val is None:
            raise Exception("SubscriptExpr: invalid index lowering.")

        if dict_val.type.count == 0:
            raise KeyError("SubscriptExpr: key lookup on empty dict")

        if isinstance(key_val, ir.Constant):
            for entry in dict_val.constant:
                entry_key = entry.constant[0]
                if self._constant_subscript_key_matches(entry_key, key_val):
                    self.result_stack.append(entry.constant[1])
                    return
            raise KeyError(
                f"SubscriptExpr: key {key_val.constant!r} not found in dict"
            )

        self._emit_runtime_subscript_lookup(
            dict_val,
            key_val,
            unsigned=self._subscript_uses_unsigned_semantics(node),
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.LiteralInt16) -> None:
        self.result_stack.append(
            ir.Constant(self._llvm.INT16_TYPE, node.value)
        )

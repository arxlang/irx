# mypy: disable-error-code=no-redef

"""
title: Literal visitor mixins for llvmliteir.
"""

from __future__ import annotations

from typing import Any, cast

from llvmlite import ir

from irx import astx
from irx.analysis.resolved_nodes import (
    ClassHeaderFieldKind,
    ResolvedClassConstruction,
)
from irx.builder.core import VisitorCore
from irx.builder.diagnostics import (
    raise_lowering_internal_error,
    require_semantic_metadata,
)
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.typecheck import typechecked


@typechecked
class LiteralVisitorMixin(VisitorMixinBase):
    def _semantic_class_construction(
        self,
        node: astx.ClassConstruct,
    ) -> ResolvedClassConstruction:
        """
        title: Return the resolved semantic class construction metadata.
        parameters:
          node:
            type: astx.ClassConstruct
        returns:
          type: ResolvedClassConstruction
        """
        semantic = getattr(node, "semantic", None)
        resolution = getattr(semantic, "resolved_class_construction", None)
        return require_semantic_metadata(
            cast(ResolvedClassConstruction | None, resolution),
            node=node,
            metadata="resolved_class_construction",
            context="class construction lowering",
        )

    def _empty_string_pointer(self, name_hint: str) -> ir.Value:
        """
        title: Return one canonical empty-string pointer.
        parameters:
          name_hint:
            type: str
        returns:
          type: ir.Value
        """
        empty_str_type = ir.ArrayType(self._llvm.INT8_TYPE, 1)
        global_name = f"class_init_empty_str_{name_hint}"
        global_value = self._llvm.module.globals.get(global_name)
        if global_value is None:
            global_value = ir.GlobalVariable(
                self._llvm.module,
                empty_str_type,
                name=global_name,
            )
            global_value.linkage = "internal"
            global_value.global_constant = True
            global_value.initializer = ir.Constant(
                empty_str_type,
                bytearray(b"\0"),
            )
        return self._llvm.ir_builder.gep(
            global_value,
            [
                ir.Constant(self._llvm.INT32_TYPE, 0),
                ir.Constant(self._llvm.INT32_TYPE, 0),
            ],
            inbounds=True,
            name=f"{name_hint}_empty",
        )

    def _default_runtime_initializer(
        self,
        type_: astx.DataType,
        *,
        name_hint: str,
    ) -> ir.Value:
        """
        title: Return one runtime default value for class field construction.
        parameters:
          type_:
            type: astx.DataType
          name_hint:
            type: str
        returns:
          type: ir.Value
        """
        llvm_type = self._llvm_type_for_ast_type(type_)
        if llvm_type is None:
            raise_lowering_internal_error(
                f"cannot lower default initializer for {type_!r}",
                node=None,
            )
        type_name = type_.__class__.__name__.lower()
        if type_name == "string":
            return self._empty_string_pointer(name_hint)
        if "float" in type_name:
            return ir.Constant(self._llvm.get_data_type(type_name), 0.0)
        if isinstance(type_, astx.ClassType):
            return ir.Constant(llvm_type, None)
        if isinstance(
            type_,
            (
                astx.StructType,
                astx.ListType,
                astx.BufferViewType,
                astx.NDArrayType,
                astx.PointerType,
                astx.OpaqueHandleType,
                astx.BufferOwnerType,
            ),
        ):
            return ir.Constant(llvm_type, None)
        return ir.Constant(self._llvm.get_data_type(type_name), 0)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.ClassConstruct) -> None:
        """
        title: Visit ClassConstruct nodes.
        parameters:
          node:
            type: astx.ClassConstruct
        """
        resolution = self._semantic_class_construction(node)
        class_ = resolution.class_
        layout = class_.layout
        result_type = self._resolved_ast_type(node)
        llvm_type = self._llvm_type_for_ast_type(result_type)
        if layout is None or not isinstance(llvm_type, ir.PointerType):
            raise_lowering_internal_error(
                "class construction is missing resolved object layout",
                node=node,
            )

        malloc = self._create_malloc_decl()
        object_size_ptr = self._llvm.ir_builder.gep(
            ir.Constant(llvm_type, None),
            [ir.Constant(self._llvm.INT32_TYPE, 1)],
            name=f"{class_.name}_size_ptr",
        )
        object_size = self._llvm.ir_builder.ptrtoint(
            object_size_ptr,
            self._llvm.SIZE_T_TYPE,
            f"{class_.name}_size",
        )
        raw_ptr = self._llvm.ir_builder.call(
            malloc,
            [object_size],
            f"{class_.name}_raw",
        )
        object_ptr = self._llvm.ir_builder.bitcast(
            raw_ptr,
            llvm_type,
            f"{class_.name}_obj",
        )

        for header in layout.header_fields:
            header_addr = self._llvm.ir_builder.gep(
                object_ptr,
                [
                    ir.Constant(self._llvm.INT32_TYPE, 0),
                    ir.Constant(self._llvm.INT32_TYPE, header.storage_index),
                ],
                inbounds=True,
                name=f"{class_.name}_{header.name}_addr",
            )
            header_value: ir.Value = ir.Constant(
                self._llvm.OPAQUE_POINTER_TYPE,
                None,
            )
            if (
                header.kind is ClassHeaderFieldKind.DISPATCH_TABLE
                and layout.dispatch_table_size > 0
            ):
                dispatch_global = self._llvm.module.globals.get(
                    layout.dispatch_global_name
                )
                if dispatch_global is None:
                    raise_lowering_internal_error(
                        "class construction is missing dispatch metadata",
                        node=node,
                    )
                header_value = self._llvm.ir_builder.bitcast(
                    dispatch_global,
                    self._llvm.OPAQUE_POINTER_TYPE,
                    name=f"{class_.name}_{header.name}_init",
                )
            self._llvm.ir_builder.store(header_value, header_addr)

        for initializer in resolution.initialization.instance_initializers:
            field = initializer.field
            field_addr = self._llvm.ir_builder.gep(
                object_ptr,
                [
                    ir.Constant(self._llvm.INT32_TYPE, 0),
                    ir.Constant(
                        self._llvm.INT32_TYPE,
                        field.storage_index,
                    ),
                ],
                inbounds=True,
                name=f"{field.member.name}_init_addr",
            )
            if initializer.value is None:
                field_value = self._default_runtime_initializer(
                    field.member.type_,
                    name_hint=f"{class_.name}_{field.member.name}",
                )
            else:
                self.visit_child(initializer.value)
                raw_value = safe_pop(self.result_stack)
                if raw_value is None:
                    raise_lowering_internal_error(
                        "class field initializer did not lower to a value",
                        node=initializer.value,
                    )
                field_value = self._cast_ast_value(
                    raw_value,
                    source_type=self._resolved_ast_type(initializer.value),
                    target_type=field.member.type_,
                )
            self._llvm.ir_builder.store(field_value, field_addr)

        self.result_stack.append(object_ptr)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralInt32) -> None:
        """
        title: Visit LiteralInt32 nodes.
        parameters:
          node:
            type: astx.LiteralInt32
        """
        self.result_stack.append(
            ir.Constant(self._llvm.INT32_TYPE, node.value)
        )

    @VisitorCore.visit.dispatch
    def visit(self, expr: astx.LiteralFloat32) -> None:
        """
        title: Visit LiteralFloat32 nodes.
        parameters:
          expr:
            type: astx.LiteralFloat32
        """
        self.result_stack.append(
            ir.Constant(self._llvm.FLOAT_TYPE, expr.value)
        )

    @VisitorCore.visit.dispatch
    def visit(self, expr: astx.LiteralFloat64) -> None:
        """
        title: Visit LiteralFloat64 nodes.
        parameters:
          expr:
            type: astx.LiteralFloat64
        """
        self.result_stack.append(
            ir.Constant(self._llvm.DOUBLE_TYPE, expr.value)
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralFloat16) -> None:
        """
        title: Visit LiteralFloat16 nodes.
        parameters:
          node:
            type: astx.LiteralFloat16
        """
        self.result_stack.append(
            ir.Constant(self._llvm.FLOAT16_TYPE, node.value)
        )

    @VisitorCore.visit.dispatch
    def visit(self, expr: astx.LiteralNone) -> None:
        """
        title: Visit LiteralNone nodes.
        parameters:
          expr:
            type: astx.LiteralNone
        """
        self.result_stack.append(None)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralBoolean) -> None:
        """
        title: Visit LiteralBoolean nodes.
        parameters:
          node:
            type: astx.LiteralBoolean
        """
        self.result_stack.append(
            ir.Constant(self._llvm.BOOLEAN_TYPE, int(node.value))
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralInt64) -> None:
        """
        title: Visit LiteralInt64 nodes.
        parameters:
          node:
            type: astx.LiteralInt64
        """
        self.result_stack.append(
            ir.Constant(self._llvm.INT64_TYPE, node.value)
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralInt8) -> None:
        """
        title: Visit LiteralInt8 nodes.
        parameters:
          node:
            type: astx.LiteralInt8
        """
        self.result_stack.append(ir.Constant(self._llvm.INT8_TYPE, node.value))

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralUInt8) -> None:
        """
        title: Visit LiteralUInt8 nodes.
        parameters:
          node:
            type: astx.LiteralUInt8
        """
        self.result_stack.append(
            ir.Constant(self._llvm.UINT8_TYPE, node.value)
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralUInt16) -> None:
        """
        title: Visit LiteralUInt16 nodes.
        parameters:
          node:
            type: astx.LiteralUInt16
        """
        self.result_stack.append(
            ir.Constant(self._llvm.UINT16_TYPE, node.value)
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralUInt32) -> None:
        """
        title: Visit LiteralUInt32 nodes.
        parameters:
          node:
            type: astx.LiteralUInt32
        """
        self.result_stack.append(
            ir.Constant(self._llvm.UINT32_TYPE, node.value)
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralUInt64) -> None:
        """
        title: Visit LiteralUInt64 nodes.
        parameters:
          node:
            type: astx.LiteralUInt64
        """
        self.result_stack.append(
            ir.Constant(self._llvm.UINT64_TYPE, node.value)
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralUInt128) -> None:
        """
        title: Visit LiteralUInt128 nodes.
        parameters:
          node:
            type: astx.LiteralUInt128
        """
        self.result_stack.append(
            ir.Constant(self._llvm.UINT128_TYPE, node.value)
        )

    @VisitorCore.visit.dispatch
    def visit(self, expr: astx.LiteralUTF8Char) -> None:
        """
        title: Visit LiteralUTF8Char nodes.
        parameters:
          expr:
            type: astx.LiteralUTF8Char
        """
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

    @VisitorCore.visit.dispatch
    def visit(self, expr: astx.LiteralUTF8String) -> None:
        """
        title: Visit LiteralUTF8String nodes.
        parameters:
          expr:
            type: astx.LiteralUTF8String
        """
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

    @VisitorCore.visit.dispatch
    def visit(self, expr: astx.LiteralString) -> None:
        """
        title: Visit LiteralString nodes.
        parameters:
          expr:
            type: astx.LiteralString
        """
        self.visit_child(astx.LiteralUTF8String(value=expr.value))

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralList) -> None:
        """
        title: Visit LiteralList nodes.
        parameters:
          node:
            type: astx.LiteralList
        """
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

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralSet) -> None:
        """
        title: Visit LiteralSet nodes.
        parameters:
          node:
            type: astx.LiteralSet
        """

        def sort_key(lit: astx.Literal) -> tuple[str, Any]:
            """
            title: Sort key.
            parameters:
              lit:
                type: astx.Literal
            returns:
              type: tuple[str, Any]
            """
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

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralTuple) -> None:
        """
        title: Visit LiteralTuple nodes.
        parameters:
          node:
            type: astx.LiteralTuple
        """
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

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralDict) -> None:
        """
        title: Visit LiteralDict nodes.
        parameters:
          node:
            type: astx.LiteralDict
        """
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

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.SubscriptExpr) -> None:
        """
        title: Visit SubscriptExpr nodes.
        parameters:
          node:
            type: astx.SubscriptExpr
        """
        if isinstance(self._resolved_ast_type(node.value), astx.ListType):
            cast(Any, self)._lower_list_subscript(node)
            return

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

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.LiteralInt16) -> None:
        """
        title: Visit LiteralInt16 nodes.
        parameters:
          node:
            type: astx.LiteralInt16
        """
        self.result_stack.append(
            ir.Constant(self._llvm.INT16_TYPE, node.value)
        )

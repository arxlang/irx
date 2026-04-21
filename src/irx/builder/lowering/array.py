# mypy: disable-error-code=no-redef

"""
title: Array and ndarray visitor mixins for llvmliteir.
"""

from __future__ import annotations

from typing import Any, cast

from llvmlite import ir

from irx import astx
from irx.analysis.types import is_float_type, is_unsigned_type
from irx.array import (
    NDARRAY_ELEMENT_TYPE_EXTRA,
    NDARRAY_FLAGS_EXTRA,
    NDARRAY_LAYOUT_EXTRA,
    NdarrayLayout,
    ndarray_element_count,
    ndarray_primitive_type_name,
)
from irx.buffer import BUFFER_VIEW_FIELD_INDICES
from irx.builder.core import VisitorCore
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builder.runtime.array.feature import (
    ARRAY_PRIMITIVE_TYPE_SPECS,
    ArrayPrimitiveTypeSpec,
)
from irx.builder.types import is_int_type
from irx.typecheck import typechecked


@typechecked
class ArrayVisitorMixin(VisitorMixinBase):
    """
    title: Array and ndarray visitor mixin.
    """

    def _static_ndarray_layout(self, node: astx.AST) -> NdarrayLayout:
        """
        title: Return static ndarray layout metadata for lowering.
        parameters:
          node:
            type: astx.AST
        returns:
          type: NdarrayLayout
        """
        semantic = getattr(node, "semantic", None)
        extras = getattr(semantic, "extras", {})
        layout = extras.get(NDARRAY_LAYOUT_EXTRA)
        if isinstance(layout, NdarrayLayout):
            return layout

        symbol = getattr(semantic, "resolved_symbol", None)
        declaration = getattr(symbol, "declaration", None)
        initializer = getattr(declaration, "value", None)
        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        layout = initializer_extras.get(NDARRAY_LAYOUT_EXTRA)
        if isinstance(layout, NdarrayLayout):
            return layout
        raise Exception("ndarray lowering requires static layout metadata")

    def _static_ndarray_element_type(self, node: astx.AST) -> astx.DataType:
        """
        title: Return static ndarray element type for lowering.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.DataType
        """
        semantic = getattr(node, "semantic", None)
        extras = getattr(semantic, "extras", {})
        element_type = extras.get(NDARRAY_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type

        resolved_type = getattr(
            semantic,
            "resolved_type",
            getattr(node, "type_", None),
        )
        if (
            isinstance(resolved_type, astx.NdarrayType)
            and resolved_type.element_type is not None
        ):
            return resolved_type.element_type

        symbol = getattr(semantic, "resolved_symbol", None)
        declaration = getattr(symbol, "declaration", None)
        initializer = getattr(declaration, "value", None)
        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        element_type = initializer_extras.get(NDARRAY_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type
        raise Exception("ndarray lowering requires a known element type")

    def _static_ndarray_flags(self, node: astx.AST) -> int:
        """
        title: Return static ndarray flags for lowering.
        parameters:
          node:
            type: astx.AST
        returns:
          type: int
        """
        semantic = getattr(node, "semantic", None)
        extras = getattr(semantic, "extras", {})
        flags = extras.get(NDARRAY_FLAGS_EXTRA)
        if isinstance(flags, int):
            return flags

        symbol = getattr(semantic, "resolved_symbol", None)
        declaration = getattr(symbol, "declaration", None)
        initializer = getattr(declaration, "value", None)
        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        flags = initializer_extras.get(NDARRAY_FLAGS_EXTRA)
        if isinstance(flags, int):
            return flags
        raise Exception("ndarray lowering requires static flags metadata")

    def _array_i64_array_pointer(
        self,
        values: tuple[int, ...],
        *,
        purpose: str,
    ) -> ir.Value:
        """
        title: Lower one tuple of i64 values to a stable global pointer.
        parameters:
          values:
            type: tuple[int, Ellipsis]
          purpose:
            type: str
        returns:
          type: ir.Value
        """
        ptr_type = self._llvm.INT64_TYPE.as_pointer()
        if not values:
            return ir.Constant(ptr_type, None)

        array_type = ir.ArrayType(self._llvm.INT64_TYPE, len(values))
        initializer = ir.Constant(
            array_type,
            [ir.Constant(self._llvm.INT64_TYPE, value) for value in values],
        )
        index = self._buffer_view_global_counter
        self._buffer_view_global_counter += 1
        global_value = ir.GlobalVariable(
            self._llvm.module,
            array_type,
            name=f"irx_ndarray_{purpose}_{index}",
        )
        global_value.linkage = "internal"
        global_value.global_constant = True
        global_value.initializer = initializer
        return self._llvm.ir_builder.gep(
            global_value,
            [
                ir.Constant(self._llvm.INT32_TYPE, 0),
                ir.Constant(self._llvm.INT32_TYPE, 0),
            ],
            inbounds=True,
            name=f"irx_ndarray_{purpose}_ptr_{index}",
        )

    def _extract_view_field(
        self,
        view: ir.Value,
        field_name: str,
        *,
        name: str,
    ) -> ir.Value:
        """
        title: Extract one canonical buffer-view field from an ndarray value.
        parameters:
          view:
            type: ir.Value
          field_name:
            type: str
          name:
            type: str
        returns:
          type: ir.Value
        """
        return self._llvm.ir_builder.extract_value(
            view,
            BUFFER_VIEW_FIELD_INDICES[field_name],
            name=name,
        )

    def _ndarray_value_from_parts(
        self,
        *,
        data: ir.Value,
        owner: ir.Value,
        dtype: ir.Value,
        layout: NdarrayLayout,
        flags: int,
        offset_value: ir.Value | None = None,
    ) -> ir.Value:
        """
        title: >-
          Assemble one lowered ndarray value from storage and layout parts.
        parameters:
          data:
            type: ir.Value
          owner:
            type: ir.Value
          dtype:
            type: ir.Value
          layout:
            type: NdarrayLayout
          flags:
            type: int
          offset_value:
            type: ir.Value | None
        returns:
          type: ir.Value
        """
        fields: list[ir.Value] = [
            data,
            owner,
            dtype,
            ir.Constant(self._llvm.INT32_TYPE, layout.ndim),
            self._array_i64_array_pointer(layout.shape, purpose="shape"),
            self._array_i64_array_pointer(layout.strides, purpose="strides"),
            (
                ir.Constant(self._llvm.INT64_TYPE, layout.offset_bytes)
                if offset_value is None
                else offset_value
            ),
            ir.Constant(self._llvm.INT32_TYPE, flags),
        ]

        value: ir.Value = ir.Constant(self._llvm.BUFFER_VIEW_TYPE, None)
        for index, field in enumerate(fields):
            value = self._llvm.ir_builder.insert_value(
                value,
                field,
                index,
                name=f"irx_ndarray_field_{index}",
            )
        return value

    def _require_ndarray_value(self, node: astx.AST) -> ir.Value:
        """
        title: Lower one ndarray expression and require buffer-view storage.
        parameters:
          node:
            type: astx.AST
        returns:
          type: ir.Value
        """
        self.visit_child(node)
        value = safe_pop(self.result_stack)
        if value is None or value.type != self._llvm.BUFFER_VIEW_TYPE:
            raise Exception("ndarray lowering requires a NdarrayType value")
        return value

    def _array_primitive_spec(
        self,
        element_type: astx.DataType,
    ) -> ArrayPrimitiveTypeSpec:
        """
        title: Return one runtime primitive storage spec for ndarray lowering.
        parameters:
          element_type:
            type: astx.DataType
        returns:
          type: ArrayPrimitiveTypeSpec
        """
        primitive_name = ndarray_primitive_type_name(element_type)
        if primitive_name is None:
            raise Exception("ndarray lowering has unsupported element type")

        spec = ARRAY_PRIMITIVE_TYPE_SPECS.get(primitive_name)
        if spec is None or not spec.buffer_view_compatible:
            raise Exception(
                "ndarray lowering requires a buffer-view-compatible primitive "
                "element type"
            )
        return spec

    def _append_value_to_array_builder(
        self,
        *,
        builder_handle: ir.Value,
        value_node: astx.AST,
        element_type: astx.DataType,
    ) -> None:
        """
        title: Append one lowered scalar to the generic Arrow array builder.
        parameters:
          builder_handle:
            type: ir.Value
          value_node:
            type: astx.AST
          element_type:
            type: astx.DataType
        """
        self.visit_child(value_node)
        value = safe_pop(self.result_stack)
        if value is None:
            raise Exception("ndarray literal expected a scalar value")

        value = self._cast_ast_value(
            value,
            source_type=self._resolved_ast_type(value_node),
            target_type=element_type,
        )

        if is_float_type(element_type):
            append = self.require_runtime_symbol(
                "array",
                "irx_arrow_array_builder_append_double",
            )
            if value.type != self._llvm.DOUBLE_TYPE:
                value = self._llvm.ir_builder.fpext(
                    value,
                    self._llvm.DOUBLE_TYPE,
                    name="irx_ndarray_double_promote",
                )
            self._llvm.ir_builder.call(append, [builder_handle, value])
            return

        if not is_int_type(value.type):
            raise Exception("ndarray builder requires integer or float values")

        if is_unsigned_type(element_type):
            append = self.require_runtime_symbol(
                "array",
                "irx_arrow_array_builder_append_uint",
            )
            if value.type.width < self._llvm.INT64_TYPE.width:
                value = self._llvm.ir_builder.zext(
                    value,
                    self._llvm.INT64_TYPE,
                    name="irx_ndarray_uint_promote",
                )
            elif value.type.width > self._llvm.INT64_TYPE.width:
                value = self._llvm.ir_builder.trunc(
                    value,
                    self._llvm.INT64_TYPE,
                    name="irx_ndarray_uint_trunc",
                )
            self._llvm.ir_builder.call(append, [builder_handle, value])
            return

        append = self.require_runtime_symbol(
            "array",
            "irx_arrow_array_builder_append_int",
        )
        if value.type.width < self._llvm.INT64_TYPE.width:
            value = self._llvm.ir_builder.sext(
                value,
                self._llvm.INT64_TYPE,
                name="irx_ndarray_int_promote",
            )
        elif value.type.width > self._llvm.INT64_TYPE.width:
            value = self._llvm.ir_builder.trunc(
                value,
                self._llvm.INT64_TYPE,
                name="irx_ndarray_int_trunc",
            )
        self._llvm.ir_builder.call(append, [builder_handle, value])

    def _build_arrow_array_from_values(
        self,
        values: list[astx.AST],
        element_type: astx.DataType,
    ) -> ir.Value:
        """
        title: Build one Arrow array handle from scalar AST values.
        parameters:
          values:
            type: list[astx.AST]
          element_type:
            type: astx.DataType
        returns:
          type: ir.Value
        """
        spec = self._array_primitive_spec(element_type)
        builder_new = self.require_runtime_symbol(
            "array",
            "irx_arrow_array_builder_new",
        )
        finish_builder = self.require_runtime_symbol(
            "array",
            "irx_arrow_array_builder_finish",
        )

        builder_slot = self._llvm.ir_builder.alloca(
            self._llvm.ARRAY_BUILDER_HANDLE_TYPE,
            name="irx_ndarray_builder_slot",
        )
        self._llvm.ir_builder.call(
            builder_new,
            [
                ir.Constant(self._llvm.INT32_TYPE, spec.type_id),
                builder_slot,
            ],
        )
        builder_handle = self._llvm.ir_builder.load(
            builder_slot,
            name="irx_ndarray_builder",
        )

        for value in values:
            self._append_value_to_array_builder(
                builder_handle=builder_handle,
                value_node=value,
                element_type=element_type,
            )

        array_slot = self._llvm.ir_builder.alloca(
            self._llvm.ARRAY_HANDLE_TYPE,
            name="irx_ndarray_array_slot",
        )
        self._llvm.ir_builder.call(
            finish_builder,
            [builder_handle, array_slot],
        )
        return self._llvm.ir_builder.load(
            array_slot,
            name="irx_ndarray_array_handle",
        )

    def _wrap_arrow_array_handle_as_ndarray(
        self,
        *,
        array_handle: ir.Value,
        layout: NdarrayLayout,
        flags: int,
    ) -> ir.Value:
        """
        title: Wrap one Arrow array handle as an owned ndarray value.
        parameters:
          array_handle:
            type: ir.Value
          layout:
            type: NdarrayLayout
          flags:
            type: int
        returns:
          type: ir.Value
        """
        borrow_view = self.require_runtime_symbol(
            "array",
            "irx_arrow_array_borrow_buffer_view",
        )
        owner_new = self.require_runtime_symbol(
            "buffer",
            "irx_buffer_owner_external_new",
        )
        release_array = self.require_runtime_symbol(
            "array",
            "irx_arrow_array_release",
        )

        borrowed_slot = self._llvm.ir_builder.alloca(
            self._llvm.BUFFER_VIEW_TYPE,
            name="irx_ndarray_borrowed_view",
        )
        self._llvm.ir_builder.call(
            borrow_view,
            [array_handle, borrowed_slot],
        )
        borrowed_view = self._llvm.ir_builder.load(
            borrowed_slot,
            name="irx_ndarray_borrowed_value",
        )

        owner_slot = self._llvm.ir_builder.alloca(
            self._llvm.BUFFER_OWNER_HANDLE_TYPE,
            name="irx_ndarray_owner_slot",
        )
        release_fn = self._llvm.ir_builder.bitcast(
            release_array,
            self._llvm.OPAQUE_POINTER_TYPE,
            name="irx_ndarray_release_fn",
        )
        self._llvm.ir_builder.call(
            owner_new,
            [array_handle, release_fn, owner_slot],
        )
        owner_handle = self._llvm.ir_builder.load(
            owner_slot,
            name="irx_ndarray_owner",
        )

        borrowed_offset = self._extract_view_field(
            borrowed_view,
            "offset_bytes",
            name="irx_ndarray_borrowed_offset",
        )
        layout_offset = ir.Constant(
            self._llvm.INT64_TYPE,
            layout.offset_bytes,
        )
        final_offset = self._llvm.ir_builder.add(
            borrowed_offset,
            layout_offset,
            name="irx_ndarray_offset_bytes",
        )
        return self._ndarray_value_from_parts(
            data=self._extract_view_field(
                borrowed_view,
                "data",
                name="irx_ndarray_data",
            ),
            owner=owner_handle,
            dtype=self._extract_view_field(
                borrowed_view,
                "dtype",
                name="irx_ndarray_dtype",
            ),
            layout=layout,
            flags=flags,
            offset_value=final_offset,
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.ArrayInt32ArrayLength) -> None:
        """
        title: Visit ArrayInt32ArrayLength nodes.
        parameters:
          node:
            type: astx.ArrayInt32ArrayLength
        """
        builder_new = self.require_runtime_symbol(
            "array", "irx_arrow_array_builder_int32_new"
        )
        append_int32 = self.require_runtime_symbol(
            "array", "irx_arrow_array_builder_append_int32"
        )
        finish_builder = self.require_runtime_symbol(
            "array", "irx_arrow_array_builder_finish"
        )
        array_length = self.require_runtime_symbol(
            "array", "irx_arrow_array_length"
        )
        release_array = self.require_runtime_symbol(
            "array", "irx_arrow_array_release"
        )

        builder_slot = self._llvm.ir_builder.alloca(
            self._llvm.ARRAY_BUILDER_HANDLE_TYPE,
            name="array_builder_slot",
        )
        self._llvm.ir_builder.call(builder_new, [builder_slot])
        builder_handle = self._llvm.ir_builder.load(
            builder_slot, "array_builder"
        )

        for item in node.values:
            self.visit_child(item)
            value = safe_pop(self.result_stack)
            if value is None:
                raise Exception("Array helper expected an integer value")
            if not is_int_type(value.type):
                raise Exception(
                    "Array helper supports only integer expressions"
                )

            if value.type.width < self._llvm.INT32_TYPE.width:
                value = self._llvm.ir_builder.sext(
                    value, self._llvm.INT32_TYPE, "array_i32_promote"
                )
            elif value.type.width > self._llvm.INT32_TYPE.width:
                value = self._llvm.ir_builder.trunc(
                    value, self._llvm.INT32_TYPE, "array_i32_trunc"
                )

            self._llvm.ir_builder.call(append_int32, [builder_handle, value])

        array_slot = self._llvm.ir_builder.alloca(
            self._llvm.ARRAY_HANDLE_TYPE,
            name="array_slot",
        )
        self._llvm.ir_builder.call(
            finish_builder, [builder_handle, array_slot]
        )
        array_handle = self._llvm.ir_builder.load(array_slot, "array_handle")
        length_i64 = self._llvm.ir_builder.call(
            array_length, [array_handle], "array_length"
        )
        self._llvm.ir_builder.call(release_array, [array_handle])

        length_i32 = self._llvm.ir_builder.trunc(
            length_i64, self._llvm.INT32_TYPE, "array_length_i32"
        )
        self.result_stack.append(length_i32)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayLiteral) -> None:
        """
        title: Visit NdarrayLiteral nodes.
        parameters:
          node:
            type: astx.NdarrayLiteral
        """
        layout = self._static_ndarray_layout(node)
        flags = self._static_ndarray_flags(node)
        element_type = self._static_ndarray_element_type(node)
        array_handle = self._build_arrow_array_from_values(
            node.values,
            element_type,
        )
        self.result_stack.append(
            self._wrap_arrow_array_handle_as_ndarray(
                array_handle=array_handle,
                layout=layout,
                flags=flags,
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayView) -> None:
        """
        title: Visit NdarrayView nodes.
        parameters:
          node:
            type: astx.NdarrayView
        """
        base_value = self._require_ndarray_value(node.base)
        layout = self._static_ndarray_layout(node)
        flags = self._static_ndarray_flags(node)

        self.result_stack.append(
            self._ndarray_value_from_parts(
                data=self._extract_view_field(
                    base_value,
                    "data",
                    name="irx_ndarray_view_data",
                ),
                owner=self._extract_view_field(
                    base_value,
                    "owner",
                    name="irx_ndarray_view_owner",
                ),
                dtype=self._extract_view_field(
                    base_value,
                    "dtype",
                    name="irx_ndarray_view_dtype",
                ),
                layout=layout,
                flags=flags,
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayIndex) -> None:
        """
        title: Visit NdarrayIndex nodes.
        parameters:
          node:
            type: astx.NdarrayIndex
        """
        view = self._require_ndarray_value(node.base)
        indices = cast(Any, self)._lower_buffer_index_indices(node.indices)
        element_ptr = cast(Any, self).lower_buffer_element_pointer(
            view,
            indices,
            self._static_ndarray_element_type(node.base),
            index_nodes=node.indices,
        )
        result = self._llvm.ir_builder.load(
            element_ptr,
            name="irx_ndarray_index_load",
        )
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayStore) -> None:
        """
        title: Visit NdarrayStore nodes.
        parameters:
          node:
            type: astx.NdarrayStore
        """
        view = self._require_ndarray_value(node.base)
        indices = cast(Any, self)._lower_buffer_index_indices(node.indices)
        element_type = self._static_ndarray_element_type(node.base)
        element_ptr = cast(Any, self).lower_buffer_element_pointer(
            view,
            indices,
            element_type,
            index_nodes=node.indices,
        )

        self.visit_child(node.value)
        value = safe_pop(self.result_stack)
        if value is None:
            raise Exception("ndarray indexed store requires a value")
        value = self._cast_ast_value(
            value,
            source_type=self._resolved_ast_type(node.value),
            target_type=element_type,
        )
        self._llvm.ir_builder.store(value, element_ptr)
        self.result_stack.append(ir.Constant(self._llvm.INT32_TYPE, 0))

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayNdim) -> None:
        """
        title: Visit NdarrayNdim nodes.
        parameters:
          node:
            type: astx.NdarrayNdim
        """
        view = self._require_ndarray_value(node.base)
        self.result_stack.append(
            self._extract_view_field(
                view,
                "ndim",
                name="irx_ndarray_ndim",
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayShape) -> None:
        """
        title: Visit NdarrayShape nodes.
        parameters:
          node:
            type: astx.NdarrayShape
        """
        view = self._require_ndarray_value(node.base)
        shape_ptr = self._extract_view_field(
            view,
            "shape",
            name="irx_ndarray_shape_ptr",
        )
        axis_ptr = self._llvm.ir_builder.gep(
            shape_ptr,
            [ir.Constant(self._llvm.INT64_TYPE, node.axis)],
            name="irx_ndarray_shape_axis_ptr",
        )
        self.result_stack.append(
            self._llvm.ir_builder.load(
                axis_ptr,
                name="irx_ndarray_shape_axis",
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayStride) -> None:
        """
        title: Visit NdarrayStride nodes.
        parameters:
          node:
            type: astx.NdarrayStride
        """
        view = self._require_ndarray_value(node.base)
        stride_ptr = self._extract_view_field(
            view,
            "strides",
            name="irx_ndarray_stride_ptr",
        )
        axis_ptr = self._llvm.ir_builder.gep(
            stride_ptr,
            [ir.Constant(self._llvm.INT64_TYPE, node.axis)],
            name="irx_ndarray_stride_axis_ptr",
        )
        self.result_stack.append(
            self._llvm.ir_builder.load(
                axis_ptr,
                name="irx_ndarray_stride_axis",
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayElementCount) -> None:
        """
        title: Visit NdarrayElementCount nodes.
        parameters:
          node:
            type: astx.NdarrayElementCount
        """
        layout = self._static_ndarray_layout(node.base)
        self.result_stack.append(
            ir.Constant(
                self._llvm.INT64_TYPE,
                ndarray_element_count(layout),
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayByteOffset) -> None:
        """
        title: Visit NdarrayByteOffset nodes.
        parameters:
          node:
            type: astx.NdarrayByteOffset
        """
        view = self._require_ndarray_value(node.base)
        indices = cast(Any, self)._lower_buffer_index_indices(node.indices)
        offset = cast(Any, self).lower_buffer_byte_offset(
            view,
            indices,
            index_nodes=node.indices,
        )
        self.result_stack.append(offset)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayRetain) -> None:
        """
        title: Visit NdarrayRetain nodes.
        parameters:
          node:
            type: astx.NdarrayRetain
        """
        retain = self.require_runtime_symbol(
            "buffer",
            "irx_buffer_view_retain",
        )
        value = self._require_ndarray_value(node.base)
        slot = self._llvm.ir_builder.alloca(
            self._llvm.BUFFER_VIEW_TYPE,
            name="irx_ndarray_retain_view",
        )
        self._llvm.ir_builder.store(value, slot)
        self.result_stack.append(
            self._llvm.ir_builder.call(
                retain,
                [slot],
                name="irx_ndarray_retain_status",
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.NdarrayRelease) -> None:
        """
        title: Visit NdarrayRelease nodes.
        parameters:
          node:
            type: astx.NdarrayRelease
        """
        release = self.require_runtime_symbol(
            "buffer",
            "irx_buffer_view_release",
        )
        value = self._require_ndarray_value(node.base)
        slot = self._llvm.ir_builder.alloca(
            self._llvm.BUFFER_VIEW_TYPE,
            name="irx_ndarray_release_view",
        )
        self._llvm.ir_builder.store(value, slot)
        self.result_stack.append(
            self._llvm.ir_builder.call(
                release,
                [slot],
                name="irx_ndarray_release_status",
            )
        )


__all__ = ["ArrayVisitorMixin"]

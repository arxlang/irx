# mypy: disable-error-code=no-redef

"""
title: Tensor visitor mixin for llvmliteir.
"""

from __future__ import annotations

from typing import Any, cast

from llvmlite import ir

from irx import astx
from irx.analysis.types import is_float_type, is_unsigned_type
from irx.buffer import BUFFER_VIEW_FIELD_INDICES
from irx.builder.core import VisitorCore
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builder.types import is_int_type
from irx.builtins.collections.array_primitives import (
    ARRAY_PRIMITIVE_TYPE_SPECS,
    ArrayPrimitiveTypeSpec,
)
from irx.builtins.collections.tensor import (
    TENSOR_ELEMENT_TYPE_EXTRA,
    TENSOR_FLAGS_EXTRA,
    TENSOR_LAYOUT_EXTRA,
    TensorLayout,
    tensor_element_count,
    tensor_primitive_type_name,
)
from irx.typecheck import typechecked


@typechecked
class TensorVisitorMixin(VisitorMixinBase):
    """
    title: Tensor visitor mixin.
    """

    def _static_tensor_layout(self, node: astx.AST) -> TensorLayout:
        """
        title: Return static Tensor layout metadata for lowering.
        parameters:
          node:
            type: astx.AST
        returns:
          type: TensorLayout
        """
        semantic = getattr(node, "semantic", None)
        extras = getattr(semantic, "extras", {})
        layout = extras.get(TENSOR_LAYOUT_EXTRA)
        if isinstance(layout, TensorLayout):
            return layout

        symbol = getattr(semantic, "resolved_symbol", None)
        declaration = getattr(symbol, "declaration", None)
        initializer = getattr(declaration, "value", None)
        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        layout = initializer_extras.get(TENSOR_LAYOUT_EXTRA)
        if isinstance(layout, TensorLayout):
            return layout
        raise Exception("tensor lowering requires static layout metadata")

    def _static_tensor_element_type(self, node: astx.AST) -> astx.DataType:
        """
        title: Return static Tensor element type for lowering.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.DataType
        """
        semantic = getattr(node, "semantic", None)
        extras = getattr(semantic, "extras", {})
        element_type = extras.get(TENSOR_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type

        resolved_type = getattr(
            semantic,
            "resolved_type",
            getattr(node, "type_", None),
        )
        if (
            isinstance(resolved_type, astx.TensorType)
            and resolved_type.element_type is not None
        ):
            return resolved_type.element_type

        symbol = getattr(semantic, "resolved_symbol", None)
        declaration = getattr(symbol, "declaration", None)
        initializer = getattr(declaration, "value", None)
        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        element_type = initializer_extras.get(TENSOR_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type
        raise Exception("tensor lowering requires a known element type")

    def _static_tensor_flags(self, node: astx.AST) -> int:
        """
        title: Return static Tensor flags for lowering.
        parameters:
          node:
            type: astx.AST
        returns:
          type: int
        """
        semantic = getattr(node, "semantic", None)
        extras = getattr(semantic, "extras", {})
        flags = extras.get(TENSOR_FLAGS_EXTRA)
        if isinstance(flags, int):
            return flags

        symbol = getattr(semantic, "resolved_symbol", None)
        declaration = getattr(symbol, "declaration", None)
        initializer = getattr(declaration, "value", None)
        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        flags = initializer_extras.get(TENSOR_FLAGS_EXTRA)
        if isinstance(flags, int):
            return flags
        raise Exception("tensor lowering requires static flags metadata")

    def _extract_view_field(
        self,
        view: ir.Value,
        field_name: str,
        *,
        name: str,
    ) -> ir.Value:
        """
        title: Extract one canonical buffer-view field from an Tensor value.
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

    def _tensor_value_from_parts(
        self,
        *,
        data: ir.Value,
        owner: ir.Value,
        dtype: ir.Value,
        layout: TensorLayout,
        flags: int,
        offset_value: ir.Value | None = None,
    ) -> ir.Value:
        """
        title: Assemble one lowered tensor value from storage and layout parts.
        parameters:
          data:
            type: ir.Value
          owner:
            type: ir.Value
          dtype:
            type: ir.Value
          layout:
            type: TensorLayout
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
            self._i64_array_pointer(
                layout.shape,
                purpose="shape",
                symbol_namespace="tensor",
            ),
            self._i64_array_pointer(
                layout.strides,
                purpose="strides",
                symbol_namespace="tensor",
            ),
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
                name=f"irx_tensor_field_{index}",
            )
        return value

    def _require_tensor_value(self, node: astx.AST) -> ir.Value:
        """
        title: Lower one Tensor expression and require buffer-view storage.
        parameters:
          node:
            type: astx.AST
        returns:
          type: ir.Value
        """
        self.visit_child(node)
        value = safe_pop(self.result_stack)
        if value is None or value.type != self._llvm.BUFFER_VIEW_TYPE:
            raise Exception("tensor lowering requires a TensorType value")
        return value

    def _tensor_primitive_spec(
        self,
        element_type: astx.DataType,
    ) -> ArrayPrimitiveTypeSpec:
        """
        title: Return one runtime primitive storage spec for Tensor lowering.
        parameters:
          element_type:
            type: astx.DataType
        returns:
          type: ArrayPrimitiveTypeSpec
        """
        primitive_name = tensor_primitive_type_name(element_type)
        if primitive_name is None:
            raise Exception("tensor lowering has unsupported element type")

        spec = ARRAY_PRIMITIVE_TYPE_SPECS.get(primitive_name)
        if spec is None or not spec.buffer_view_compatible:
            raise Exception(
                "tensor lowering requires a buffer-view-compatible primitive "
                "element type"
            )
        return spec

    def _append_value_to_tensor_builder(
        self,
        *,
        builder_handle: ir.Value,
        value_node: astx.AST,
        element_type: astx.DataType,
    ) -> None:
        """
        title: Append one lowered scalar to the Arrow tensor builder.
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
            raise Exception("tensor literal expected a scalar value")

        value = self._cast_ast_value(
            value,
            source_type=self._resolved_ast_type(value_node),
            target_type=element_type,
        )

        if is_float_type(element_type):
            append = self.require_runtime_symbol(
                "tensor",
                "irx_arrow_tensor_builder_append_double",
            )
            if value.type != self._llvm.DOUBLE_TYPE:
                value = self._llvm.ir_builder.fpext(
                    value,
                    self._llvm.DOUBLE_TYPE,
                    name="irx_tensor_double_promote",
                )
            self._llvm.ir_builder.call(append, [builder_handle, value])
            return

        if not is_int_type(value.type):
            raise Exception("tensor builder requires integer or float values")

        if is_unsigned_type(element_type):
            append = self.require_runtime_symbol(
                "tensor",
                "irx_arrow_tensor_builder_append_uint",
            )
            if value.type.width < self._llvm.INT64_TYPE.width:
                value = self._llvm.ir_builder.zext(
                    value,
                    self._llvm.INT64_TYPE,
                    name="irx_tensor_uint_promote",
                )
            elif value.type.width > self._llvm.INT64_TYPE.width:
                value = self._llvm.ir_builder.trunc(
                    value,
                    self._llvm.INT64_TYPE,
                    name="irx_tensor_uint_trunc",
                )
            self._llvm.ir_builder.call(append, [builder_handle, value])
            return

        append = self.require_runtime_symbol(
            "tensor",
            "irx_arrow_tensor_builder_append_int",
        )
        if value.type.width < self._llvm.INT64_TYPE.width:
            value = self._llvm.ir_builder.sext(
                value,
                self._llvm.INT64_TYPE,
                name="irx_tensor_int_promote",
            )
        elif value.type.width > self._llvm.INT64_TYPE.width:
            value = self._llvm.ir_builder.trunc(
                value,
                self._llvm.INT64_TYPE,
                name="irx_tensor_int_trunc",
            )
        self._llvm.ir_builder.call(append, [builder_handle, value])

    def _build_arrow_tensor_from_values(
        self,
        values: list[astx.AST],
        element_type: astx.DataType,
        layout: TensorLayout,
    ) -> ir.Value:
        """
        title: Build one Arrow tensor handle from scalar AST values.
        parameters:
          values:
            type: list[astx.AST]
          element_type:
            type: astx.DataType
          layout:
            type: TensorLayout
        returns:
          type: ir.Value
        """
        spec = self._tensor_primitive_spec(element_type)
        builder_new = self.require_runtime_symbol(
            "tensor",
            "irx_arrow_tensor_builder_new",
        )
        finish_builder = self.require_runtime_symbol(
            "tensor",
            "irx_arrow_tensor_builder_finish",
        )

        builder_slot = self._llvm.ir_builder.alloca(
            self._llvm.TENSOR_BUILDER_HANDLE_TYPE,
            name="irx_tensor_builder_slot",
        )
        self._llvm.ir_builder.call(
            builder_new,
            [
                ir.Constant(self._llvm.INT32_TYPE, spec.type_id),
                ir.Constant(self._llvm.INT32_TYPE, layout.ndim),
                self._i64_array_pointer(
                    layout.shape,
                    purpose="builder_shape",
                    symbol_namespace="tensor",
                ),
                self._i64_array_pointer(
                    layout.strides,
                    purpose="builder_strides",
                    symbol_namespace="tensor",
                ),
                builder_slot,
            ],
        )
        builder_handle = self._llvm.ir_builder.load(
            builder_slot,
            name="irx_tensor_builder",
        )

        for value in values:
            self._append_value_to_tensor_builder(
                builder_handle=builder_handle,
                value_node=value,
                element_type=element_type,
            )

        tensor_slot = self._llvm.ir_builder.alloca(
            self._llvm.TENSOR_HANDLE_TYPE,
            name="irx_tensor_handle_slot",
        )
        self._llvm.ir_builder.call(
            finish_builder,
            [builder_handle, tensor_slot],
        )
        return self._llvm.ir_builder.load(
            tensor_slot,
            name="irx_tensor_handle",
        )

    def _wrap_arrow_tensor_handle_as_tensor(
        self,
        *,
        tensor_handle: ir.Value,
        layout: TensorLayout,
        flags: int,
    ) -> ir.Value:
        """
        title: Wrap one Arrow tensor handle as an owned Tensor value.
        parameters:
          tensor_handle:
            type: ir.Value
          layout:
            type: TensorLayout
          flags:
            type: int
        returns:
          type: ir.Value
        """
        borrow_view = self.require_runtime_symbol(
            "tensor",
            "irx_arrow_tensor_borrow_buffer_view",
        )
        owner_new = self.require_runtime_symbol(
            "buffer",
            "irx_buffer_owner_external_new",
        )
        release_tensor = self.require_runtime_symbol(
            "tensor",
            "irx_arrow_tensor_release",
        )

        borrowed_slot = self._llvm.ir_builder.alloca(
            self._llvm.BUFFER_VIEW_TYPE,
            name="irx_tensor_borrowed_view",
        )
        self._llvm.ir_builder.call(
            borrow_view,
            [tensor_handle, borrowed_slot],
        )
        borrowed_view = self._llvm.ir_builder.load(
            borrowed_slot,
            name="irx_tensor_borrowed_value",
        )

        owner_slot = self._llvm.ir_builder.alloca(
            self._llvm.BUFFER_OWNER_HANDLE_TYPE,
            name="irx_tensor_owner_slot",
        )
        release_fn = self._llvm.ir_builder.bitcast(
            release_tensor,
            self._llvm.OPAQUE_POINTER_TYPE,
            name="irx_tensor_release_fn",
        )
        self._llvm.ir_builder.call(
            owner_new,
            [tensor_handle, release_fn, owner_slot],
        )
        owner_handle = self._llvm.ir_builder.load(
            owner_slot,
            name="irx_tensor_owner",
        )

        borrowed_offset = self._extract_view_field(
            borrowed_view,
            "offset_bytes",
            name="irx_tensor_borrowed_offset",
        )
        layout_offset = ir.Constant(
            self._llvm.INT64_TYPE,
            layout.offset_bytes,
        )
        final_offset = self._llvm.ir_builder.add(
            borrowed_offset,
            layout_offset,
            name="irx_tensor_offset_bytes",
        )
        return self._tensor_value_from_parts(
            data=self._extract_view_field(
                borrowed_view,
                "data",
                name="irx_tensor_data",
            ),
            owner=owner_handle,
            dtype=self._extract_view_field(
                borrowed_view,
                "dtype",
                name="irx_tensor_dtype",
            ),
            layout=layout,
            flags=flags,
            offset_value=final_offset,
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorLiteral) -> None:
        """
        title: Visit TensorLiteral nodes.
        parameters:
          node:
            type: astx.TensorLiteral
        """
        layout = self._static_tensor_layout(node)
        flags = self._static_tensor_flags(node)
        element_type = self._static_tensor_element_type(node)
        tensor_handle = self._build_arrow_tensor_from_values(
            node.values,
            element_type,
            layout,
        )
        self.result_stack.append(
            self._wrap_arrow_tensor_handle_as_tensor(
                tensor_handle=tensor_handle,
                layout=layout,
                flags=flags,
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorView) -> None:
        """
        title: Visit TensorView nodes.
        parameters:
          node:
            type: astx.TensorView
        """
        base_value = self._require_tensor_value(node.base)
        layout = self._static_tensor_layout(node)
        flags = self._static_tensor_flags(node)

        self.result_stack.append(
            self._tensor_value_from_parts(
                data=self._extract_view_field(
                    base_value,
                    "data",
                    name="irx_tensor_view_data",
                ),
                owner=self._extract_view_field(
                    base_value,
                    "owner",
                    name="irx_tensor_view_owner",
                ),
                dtype=self._extract_view_field(
                    base_value,
                    "dtype",
                    name="irx_tensor_view_dtype",
                ),
                layout=layout,
                flags=flags,
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorIndex) -> None:
        """
        title: Visit TensorIndex nodes.
        parameters:
          node:
            type: astx.TensorIndex
        """
        view = self._require_tensor_value(node.base)
        indices = cast(Any, self)._lower_buffer_index_indices(node.indices)
        element_ptr = cast(Any, self).lower_buffer_element_pointer(
            view,
            indices,
            self._static_tensor_element_type(node.base),
            index_nodes=node.indices,
        )
        result = self._llvm.ir_builder.load(
            element_ptr,
            name="irx_tensor_index_load",
        )
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorStore) -> None:
        """
        title: Visit TensorStore nodes.
        parameters:
          node:
            type: astx.TensorStore
        """
        view = self._require_tensor_value(node.base)
        indices = cast(Any, self)._lower_buffer_index_indices(node.indices)
        element_type = self._static_tensor_element_type(node.base)
        element_ptr = cast(Any, self).lower_buffer_element_pointer(
            view,
            indices,
            element_type,
            index_nodes=node.indices,
        )

        self.visit_child(node.value)
        value = safe_pop(self.result_stack)
        if value is None:
            raise Exception("tensor indexed store requires a value")
        value = self._cast_ast_value(
            value,
            source_type=self._resolved_ast_type(node.value),
            target_type=element_type,
        )
        self._llvm.ir_builder.store(value, element_ptr)
        self.result_stack.append(ir.Constant(self._llvm.INT32_TYPE, 0))

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorNDim) -> None:
        """
        title: Visit TensorNDim nodes.
        parameters:
          node:
            type: astx.TensorNDim
        """
        view = self._require_tensor_value(node.base)
        self.result_stack.append(
            self._extract_view_field(
                view,
                "ndim",
                name="irx_tensor_ndim",
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorShape) -> None:
        """
        title: Visit TensorShape nodes.
        parameters:
          node:
            type: astx.TensorShape
        """
        view = self._require_tensor_value(node.base)
        shape_ptr = self._extract_view_field(
            view,
            "shape",
            name="irx_tensor_shape_ptr",
        )
        axis_ptr = self._llvm.ir_builder.gep(
            shape_ptr,
            [ir.Constant(self._llvm.INT64_TYPE, node.axis)],
            name="irx_tensor_shape_axis_ptr",
        )
        self.result_stack.append(
            self._llvm.ir_builder.load(
                axis_ptr,
                name="irx_tensor_shape_axis",
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorStride) -> None:
        """
        title: Visit TensorStride nodes.
        parameters:
          node:
            type: astx.TensorStride
        """
        view = self._require_tensor_value(node.base)
        stride_ptr = self._extract_view_field(
            view,
            "strides",
            name="irx_tensor_stride_ptr",
        )
        axis_ptr = self._llvm.ir_builder.gep(
            stride_ptr,
            [ir.Constant(self._llvm.INT64_TYPE, node.axis)],
            name="irx_tensor_stride_axis_ptr",
        )
        self.result_stack.append(
            self._llvm.ir_builder.load(
                axis_ptr,
                name="irx_tensor_stride_axis",
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorElementCount) -> None:
        """
        title: Visit TensorElementCount nodes.
        parameters:
          node:
            type: astx.TensorElementCount
        """
        layout = self._static_tensor_layout(node.base)
        self.result_stack.append(
            ir.Constant(
                self._llvm.INT64_TYPE,
                tensor_element_count(layout),
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorByteOffset) -> None:
        """
        title: Visit TensorByteOffset nodes.
        parameters:
          node:
            type: astx.TensorByteOffset
        """
        view = self._require_tensor_value(node.base)
        indices = cast(Any, self)._lower_buffer_index_indices(node.indices)
        offset = cast(Any, self).lower_buffer_byte_offset(
            view,
            indices,
            index_nodes=node.indices,
        )
        self.result_stack.append(offset)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorRetain) -> None:
        """
        title: Visit TensorRetain nodes.
        parameters:
          node:
            type: astx.TensorRetain
        """
        retain = self.require_runtime_symbol(
            "buffer",
            "irx_buffer_view_retain",
        )
        value = self._require_tensor_value(node.base)
        slot = self._llvm.ir_builder.alloca(
            self._llvm.BUFFER_VIEW_TYPE,
            name="irx_tensor_retain_view",
        )
        self._llvm.ir_builder.store(value, slot)
        self.result_stack.append(
            self._llvm.ir_builder.call(
                retain,
                [slot],
                name="irx_tensor_retain_status",
            )
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.TensorRelease) -> None:
        """
        title: Visit TensorRelease nodes.
        parameters:
          node:
            type: astx.TensorRelease
        """
        release = self.require_runtime_symbol(
            "buffer",
            "irx_buffer_view_release",
        )
        value = self._require_tensor_value(node.base)
        slot = self._llvm.ir_builder.alloca(
            self._llvm.BUFFER_VIEW_TYPE,
            name="irx_tensor_release_view",
        )
        self._llvm.ir_builder.store(value, slot)
        self.result_stack.append(
            self._llvm.ir_builder.call(
                release,
                [slot],
                name="irx_tensor_release_status",
            )
        )


__all__ = ["TensorVisitorMixin"]

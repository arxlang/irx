# mypy: disable-error-code=no-redef

"""
title: Low-level buffer/view lowering for llvmliteir.
summary: >-
  Lower the IRx buffer/view substrate as plain structs and explicit runtime
  helper calls without adding a scientific array API.
"""

from __future__ import annotations

from llvmlite import ir

from irx import astx
from irx.analysis.types import is_unsigned_type
from irx.buffer import (
    BUFFER_VIEW_ELEMENT_TYPE_EXTRA,
    BUFFER_VIEW_FIELD_INDICES,
    BUFFER_VIEW_METADATA_EXTRA,
    BufferHandle,
    BufferIndexBoundsPolicy,
    BufferViewMetadata,
)
from irx.builder.core import VisitorCore
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builder.types import is_int_type
from irx.typecheck import typechecked


@typechecked
class BufferVisitorMixin(VisitorMixinBase):
    """
    title: Buffer/view visitor mixin.
    """

    def _buffer_handle_value(
        self,
        handle: BufferHandle,
        target_type: ir.Type,
    ) -> ir.Value:
        """
        title: Lower one static opaque handle.
        parameters:
          handle:
            type: BufferHandle
          target_type:
            type: ir.Type
        returns:
          type: ir.Value
        """
        if handle.is_null:
            return ir.Constant(target_type, None)
        assert handle.address is not None
        token = ir.Constant(self._llvm.INT64_TYPE, handle.address)
        return token.inttoptr(target_type)

    def _i64_array_pointer(
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
            name=f"irx_buffer_{purpose}_{index}",
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
            name=f"irx_buffer_{purpose}_ptr_{index}",
        )

    def _buffer_view_value_from_metadata(
        self,
        metadata: BufferViewMetadata,
    ) -> ir.Value:
        """
        title: Lower static buffer view metadata to a struct value.
        parameters:
          metadata:
            type: BufferViewMetadata
        returns:
          type: ir.Value
        """
        fields: list[ir.Value] = [
            self._buffer_handle_value(
                metadata.data,
                self._llvm.OPAQUE_POINTER_TYPE,
            ),
            self._buffer_handle_value(
                metadata.owner,
                self._llvm.BUFFER_OWNER_HANDLE_TYPE,
            ),
            self._buffer_handle_value(
                metadata.dtype,
                self._llvm.OPAQUE_POINTER_TYPE,
            ),
            ir.Constant(self._llvm.INT32_TYPE, metadata.ndim),
            self._i64_array_pointer(metadata.shape, purpose="shape"),
            self._i64_array_pointer(metadata.strides, purpose="strides"),
            ir.Constant(self._llvm.INT64_TYPE, metadata.offset_bytes),
            ir.Constant(self._llvm.INT32_TYPE, metadata.flags),
        ]

        value: ir.Value = ir.Constant(self._llvm.BUFFER_VIEW_TYPE, None)
        for index, field in enumerate(fields):
            value = self._llvm.ir_builder.insert_value(
                value,
                field,
                index,
                name=f"irx_buffer_view_field_{index}",
            )
        return value

    def _buffer_view_pointer_for_call(
        self,
        view: astx.AST,
        *,
        name: str,
    ) -> ir.Value:
        """
        title: Lower a buffer view expression to a temporary call pointer.
        parameters:
          view:
            type: astx.AST
          name:
            type: str
        returns:
          type: ir.Value
        """
        self.visit_child(view)
        value = safe_pop(self.result_stack)
        if value is None:
            raise Exception("buffer helper expected a view value")
        if value.type != self._llvm.BUFFER_VIEW_TYPE:
            raise Exception("buffer helper requires a BufferViewType value")

        slot = self._llvm.ir_builder.alloca(
            self._llvm.BUFFER_VIEW_TYPE,
            name=name,
        )
        self._llvm.ir_builder.store(value, slot)
        return slot

    def _extract_buffer_view_field(
        self,
        view: ir.Value,
        field_name: str,
        *,
        name: str,
    ) -> ir.Value:
        """
        title: Extract one canonical buffer view descriptor field.
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

    def _extract_buffer_view_data(self, view: ir.Value) -> ir.Value:
        """
        title: Extract a buffer view data pointer.
        parameters:
          view:
            type: ir.Value
        returns:
          type: ir.Value
        """
        return self._extract_buffer_view_field(
            view,
            "data",
            name="irx_buffer_view_data",
        )

    def _extract_buffer_view_shape(self, view: ir.Value) -> ir.Value:
        """
        title: Extract a buffer view shape pointer.
        parameters:
          view:
            type: ir.Value
        returns:
          type: ir.Value
        """
        return self._extract_buffer_view_field(
            view,
            "shape",
            name="irx_buffer_view_shape",
        )

    def _extract_buffer_view_strides(self, view: ir.Value) -> ir.Value:
        """
        title: Extract a buffer view strides pointer.
        parameters:
          view:
            type: ir.Value
        returns:
          type: ir.Value
        """
        return self._extract_buffer_view_field(
            view,
            "strides",
            name="irx_buffer_view_strides",
        )

    def _extract_buffer_view_offset_bytes(self, view: ir.Value) -> ir.Value:
        """
        title: Extract a buffer view byte offset.
        parameters:
          view:
            type: ir.Value
        returns:
          type: ir.Value
        """
        return self._extract_buffer_view_field(
            view,
            "offset_bytes",
            name="irx_buffer_view_offset_bytes",
        )

    def _buffer_index_element_type(
        self,
        node: astx.AST,
    ) -> astx.DataType:
        """
        title: Return the semantic element type for indexed access lowering.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.DataType
        """
        semantic = getattr(node, "semantic", None)
        extras = getattr(semantic, "extras", {})
        element_type = extras.get(BUFFER_VIEW_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type
        resolved_type = self._resolved_ast_type(node)
        if isinstance(resolved_type, astx.DataType):
            return resolved_type
        raise Exception(
            "buffer view indexing requires a semantic element type"
        )

    def _normalize_buffer_index_value(
        self,
        value: ir.Value,
        index_node: astx.AST,
    ) -> ir.Value:
        """
        title: Normalize one lowered index to descriptor stride arithmetic.
        parameters:
          value:
            type: ir.Value
          index_node:
            type: astx.AST
        returns:
          type: ir.Value
        """
        if not is_int_type(value.type):
            raise Exception(
                "buffer view index lowering requires integer values"
            )
        if value.type == self._llvm.INT64_TYPE:
            return value
        if value.type.width < self._llvm.INT64_TYPE.width:
            if is_unsigned_type(self._resolved_ast_type(index_node)):
                return self._llvm.ir_builder.zext(
                    value,
                    self._llvm.INT64_TYPE,
                    name="irx_buffer_index_zext",
                )
            return self._llvm.ir_builder.sext(
                value,
                self._llvm.INT64_TYPE,
                name="irx_buffer_index_sext",
            )
        return self._llvm.ir_builder.trunc(
            value,
            self._llvm.INT64_TYPE,
            name="irx_buffer_index_trunc",
        )

    def lower_buffer_byte_offset(
        self,
        view: ir.Value,
        indices: list[ir.Value],
        *,
        index_nodes: list[astx.AST],
        bounds_policy: BufferIndexBoundsPolicy = (
            BufferIndexBoundsPolicy.DEFAULT
        ),
    ) -> ir.Value:
        """
        title: Lower one buffer view indexed access to a byte offset.
        parameters:
          view:
            type: ir.Value
          indices:
            type: list[ir.Value]
          index_nodes:
            type: list[astx.AST]
          bounds_policy:
            type: BufferIndexBoundsPolicy
        returns:
          type: ir.Value
        """
        _ = bounds_policy
        if view.type != self._llvm.BUFFER_VIEW_TYPE:
            raise Exception("buffer view indexing requires a BufferViewType")
        if len(indices) != len(index_nodes):
            raise Exception("buffer view index lowering arity mismatch")

        strides = self._extract_buffer_view_strides(view)
        total_offset = self._extract_buffer_view_offset_bytes(view)

        for axis, (index, index_node) in enumerate(
            zip(indices, index_nodes, strict=True)
        ):
            index64 = self._normalize_buffer_index_value(index, index_node)
            stride_ptr = self._llvm.ir_builder.gep(
                strides,
                [ir.Constant(self._llvm.INT64_TYPE, axis)],
                name=f"irx_buffer_index_stride_ptr_{axis}",
            )
            stride = self._llvm.ir_builder.load(
                stride_ptr,
                name=f"irx_buffer_index_stride_{axis}",
            )
            scaled_index = self._llvm.ir_builder.mul(
                index64,
                stride,
                name=f"irx_buffer_index_scaled_{axis}",
            )
            total_offset = self._llvm.ir_builder.add(
                total_offset,
                scaled_index,
                name=f"irx_buffer_index_offset_{axis}",
            )

        return total_offset

    def lower_buffer_element_pointer(
        self,
        view: ir.Value,
        indices: list[ir.Value],
        element_type: astx.DataType,
        *,
        index_nodes: list[astx.AST],
        bounds_policy: BufferIndexBoundsPolicy = (
            BufferIndexBoundsPolicy.DEFAULT
        ),
    ) -> ir.Value:
        """
        title: Lower a buffer view indexed access to a typed element pointer.
        parameters:
          view:
            type: ir.Value
          indices:
            type: list[ir.Value]
          element_type:
            type: astx.DataType
          index_nodes:
            type: list[astx.AST]
          bounds_policy:
            type: BufferIndexBoundsPolicy
        returns:
          type: ir.Value
        """
        element_llvm_type = self._llvm_type_for_ast_type(element_type)
        if element_llvm_type is None:
            raise Exception(
                "buffer view indexing has unsupported element type"
            )

        data = self._extract_buffer_view_data(view)
        total_offset = self.lower_buffer_byte_offset(
            view,
            indices,
            index_nodes=index_nodes,
            bounds_policy=bounds_policy,
        )
        byte_ptr = self._llvm.ir_builder.gep(
            data,
            [total_offset],
            name="irx_buffer_index_byte_ptr",
        )
        element_ptr_type = element_llvm_type.as_pointer()
        if byte_ptr.type == element_ptr_type:
            return byte_ptr
        return self._llvm.ir_builder.bitcast(
            byte_ptr,
            element_ptr_type,
            name="irx_buffer_index_element_ptr",
        )

    def _lower_buffer_index_indices(
        self,
        indices: list[astx.AST],
    ) -> list[ir.Value]:
        """
        title: Lower buffer view index expressions.
        parameters:
          indices:
            type: list[astx.AST]
        returns:
          type: list[ir.Value]
        """
        lowered_indices: list[ir.Value] = []
        for index in indices:
            self.visit_child(index)
            value = safe_pop(self.result_stack)
            if value is None or not is_int_type(value.type):
                raise Exception(
                    "buffer view index lowering requires integer values"
                )
            lowered_indices.append(value)
        return lowered_indices

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.BufferViewDescriptor) -> None:
        """
        title: Visit BufferViewDescriptor nodes.
        parameters:
          node:
            type: astx.BufferViewDescriptor
        """
        metadata = getattr(
            getattr(node, "semantic", None),
            "extras",
            {},
        ).get(BUFFER_VIEW_METADATA_EXTRA, node.metadata)
        self.result_stack.append(
            self._buffer_view_value_from_metadata(metadata)
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.BufferViewIndex) -> None:
        """
        title: Visit BufferViewIndex nodes.
        parameters:
          node:
            type: astx.BufferViewIndex
        """
        self.visit_child(node.base)
        view = safe_pop(self.result_stack)
        if view is None or view.type != self._llvm.BUFFER_VIEW_TYPE:
            raise Exception(
                "buffer view indexed read requires a BufferViewType value"
            )
        indices = self._lower_buffer_index_indices(node.indices)
        element_ptr = self.lower_buffer_element_pointer(
            view,
            indices,
            self._buffer_index_element_type(node),
            index_nodes=node.indices,
        )
        result = self._llvm.ir_builder.load(
            element_ptr,
            name="irx_buffer_index_load",
        )
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.BufferViewStore) -> None:
        """
        title: Visit BufferViewStore nodes.
        parameters:
          node:
            type: astx.BufferViewStore
        """
        self.visit_child(node.base)
        view = safe_pop(self.result_stack)
        if view is None or view.type != self._llvm.BUFFER_VIEW_TYPE:
            raise Exception(
                "buffer view indexed store requires a BufferViewType value"
            )
        indices = self._lower_buffer_index_indices(node.indices)
        element_type = self._buffer_index_element_type(node)
        element_ptr = self.lower_buffer_element_pointer(
            view,
            indices,
            element_type,
            index_nodes=node.indices,
        )

        self.visit_child(node.value)
        value = safe_pop(self.result_stack)
        if value is None:
            raise Exception("buffer view indexed store requires a value")
        value = self._cast_ast_value(
            value,
            source_type=self._resolved_ast_type(node.value),
            target_type=element_type,
        )
        self._llvm.ir_builder.store(value, element_ptr)
        self.result_stack.append(ir.Constant(self._llvm.INT32_TYPE, 0))

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.BufferViewWrite) -> None:
        """
        title: Visit BufferViewWrite nodes.
        parameters:
          node:
            type: astx.BufferViewWrite
        """
        self.visit_child(node.view)
        view = safe_pop(self.result_stack)
        if view is None or view.type != self._llvm.BUFFER_VIEW_TYPE:
            raise Exception(
                "buffer view write requires a BufferViewType value"
            )

        self.visit_child(node.value)
        value = safe_pop(self.result_stack)
        if value is None or not is_int_type(value.type):
            raise Exception("buffer view write requires an integer byte value")
        if value.type.width != self._llvm.INT8_TYPE.width:
            raise Exception("buffer view write requires an 8-bit value")

        data = self._extract_buffer_view_data(view)
        offset = self._extract_buffer_view_offset_bytes(view)
        byte_offset = ir.Constant(self._llvm.INT64_TYPE, node.byte_offset)
        total_offset = self._llvm.ir_builder.add(
            offset,
            byte_offset,
            name="irx_buffer_write_total_offset",
        )
        write_ptr = self._llvm.ir_builder.gep(
            data,
            [total_offset],
            name="irx_buffer_write_ptr",
        )
        self._llvm.ir_builder.store(value, write_ptr)
        self.result_stack.append(ir.Constant(self._llvm.INT32_TYPE, 0))

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.BufferViewRetain) -> None:
        """
        title: Visit BufferViewRetain nodes.
        parameters:
          node:
            type: astx.BufferViewRetain
        """
        retain = self.require_runtime_symbol(
            "buffer",
            "irx_buffer_view_retain",
        )
        view_ptr = self._buffer_view_pointer_for_call(
            node.view,
            name="irx_buffer_retain_view",
        )
        result = self._llvm.ir_builder.call(
            retain,
            [view_ptr],
            name="irx_buffer_retain_status",
        )
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.BufferViewRelease) -> None:
        """
        title: Visit BufferViewRelease nodes.
        parameters:
          node:
            type: astx.BufferViewRelease
        """
        release = self.require_runtime_symbol(
            "buffer",
            "irx_buffer_view_release",
        )
        view_ptr = self._buffer_view_pointer_for_call(
            node.view,
            name="irx_buffer_release_view",
        )
        result = self._llvm.ir_builder.call(
            release,
            [view_ptr],
            name="irx_buffer_release_status",
        )
        self.result_stack.append(result)

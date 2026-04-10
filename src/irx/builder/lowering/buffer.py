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
from irx.buffer import (
    BUFFER_VIEW_FIELD_INDICES,
    BUFFER_VIEW_METADATA_EXTRA,
    BufferHandle,
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

        data = self._llvm.ir_builder.extract_value(
            view,
            BUFFER_VIEW_FIELD_INDICES["data"],
            name="irx_buffer_write_data",
        )
        offset = self._llvm.ir_builder.extract_value(
            view,
            BUFFER_VIEW_FIELD_INDICES["offset_bytes"],
            name="irx_buffer_write_offset",
        )
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

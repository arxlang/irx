# mypy: disable-error-code=no-redef

"""
title: Array visitor mixins for llvmliteir.
"""

from irx import astx
from irx.builder.core import VisitorCore
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builder.types import is_int_type
from irx.typecheck import typechecked


@typechecked
class ArrayVisitorMixin(VisitorMixinBase):
    @VisitorCore.visit.dispatch  # type: ignore[attr-defined,untyped-decorator]
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


__all__ = ["ArrayVisitorMixin"]

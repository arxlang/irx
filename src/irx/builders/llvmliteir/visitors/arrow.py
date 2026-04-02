# mypy: disable-error-code=no-redef

"""
title: Arrow visitor mixins for llvmliteir.
"""

from irx import astx
from irx.builders.base import BuilderVisitor
from irx.builders.llvmliteir.protocols import VisitorMixinBase
from irx.builders.llvmliteir.runtime import safe_pop
from irx.builders.llvmliteir.types import is_int_type


class ArrowVisitorMixin(VisitorMixinBase):
    @BuilderVisitor.visit.dispatch  # type: ignore[attr-defined,untyped-decorator]
    def visit(self, node: astx.ArrowInt32ArrayLength) -> None:
        builder_new = self.require_runtime_symbol(
            "arrow", "irx_arrow_array_builder_int32_new"
        )
        append_int32 = self.require_runtime_symbol(
            "arrow", "irx_arrow_array_builder_append_int32"
        )
        finish_builder = self.require_runtime_symbol(
            "arrow", "irx_arrow_array_builder_finish"
        )
        array_length = self.require_runtime_symbol(
            "arrow", "irx_arrow_array_length"
        )
        release_array = self.require_runtime_symbol(
            "arrow", "irx_arrow_array_release"
        )

        builder_slot = self._llvm.ir_builder.alloca(
            self._llvm.ARROW_ARRAY_BUILDER_HANDLE_TYPE,
            name="arrow_builder_slot",
        )
        self._llvm.ir_builder.call(builder_new, [builder_slot])
        builder_handle = self._llvm.ir_builder.load(
            builder_slot, "arrow_builder"
        )

        for item in node.values:
            self.visit_child(item)
            value = safe_pop(self.result_stack)
            if value is None:
                raise Exception("Arrow helper expected an integer value")
            if not is_int_type(value.type):
                raise Exception(
                    "Arrow helper supports only integer expressions"
                )

            if value.type.width < self._llvm.INT32_TYPE.width:
                value = self._llvm.ir_builder.sext(
                    value, self._llvm.INT32_TYPE, "arrow_i32_promote"
                )
            elif value.type.width > self._llvm.INT32_TYPE.width:
                value = self._llvm.ir_builder.trunc(
                    value, self._llvm.INT32_TYPE, "arrow_i32_trunc"
                )

            self._llvm.ir_builder.call(append_int32, [builder_handle, value])

        array_slot = self._llvm.ir_builder.alloca(
            self._llvm.ARROW_ARRAY_HANDLE_TYPE,
            name="arrow_array_slot",
        )
        self._llvm.ir_builder.call(
            finish_builder, [builder_handle, array_slot]
        )
        array_handle = self._llvm.ir_builder.load(array_slot, "arrow_array")
        length_i64 = self._llvm.ir_builder.call(
            array_length, [array_handle], "arrow_length"
        )
        self._llvm.ir_builder.call(release_array, [array_handle])

        length_i32 = self._llvm.ir_builder.trunc(
            length_i64, self._llvm.INT32_TYPE, "arrow_length_i32"
        )
        self.result_stack.append(length_i32)

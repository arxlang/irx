# mypy: disable-error-code=no-redef

"""
title: System/runtime visitor mixins for llvmliteir.
"""

from llvmlite import ir

from irx import astx
from irx.builders.base import BuilderVisitor
from irx.builders.llvmliteir.protocols import VisitorMixinBase
from irx.builders.llvmliteir.runtime import safe_pop
from irx.builders.llvmliteir.types import is_fp_type, is_int_type


class SystemVisitorMixin(VisitorMixinBase):
    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.Cast) -> None:
        self.visit_child(node.value)
        value = safe_pop(self.result_stack)
        if value is None:
            raise Exception("Invalid value in Cast")

        target_type_str = node.target_type.__class__.__name__.lower()
        target_type = self._llvm.get_data_type(target_type_str)

        if value.type == target_type:
            self.result_stack.append(value)
            return

        result: ir.Value
        if is_int_type(value.type) and is_int_type(target_type):
            if value.type.width < target_type.width:
                result = self._llvm.ir_builder.sext(
                    value, target_type, "cast_int_up"
                )
            else:
                result = self._llvm.ir_builder.trunc(
                    value, target_type, "cast_int_down"
                )
        elif is_int_type(value.type) and is_fp_type(target_type):
            result = self._llvm.ir_builder.sitofp(
                value, target_type, "cast_int_to_fp"
            )
        elif is_fp_type(value.type) and is_int_type(target_type):
            result = self._llvm.ir_builder.fptosi(
                value, target_type, "cast_fp_to_int"
            )
        elif isinstance(value.type, ir.FloatType) and isinstance(
            target_type, ir.HalfType
        ):
            result = self._llvm.ir_builder.fptrunc(
                value, target_type, "cast_fp_to_half"
            )
        elif isinstance(value.type, ir.HalfType) and isinstance(
            target_type, ir.FloatType
        ):
            result = self._llvm.ir_builder.fpext(
                value, target_type, "cast_half_to_fp"
            )
        elif isinstance(value.type, ir.FloatType) and isinstance(
            target_type, ir.FloatType
        ):
            if value.type.width < target_type.width:
                result = self._llvm.ir_builder.fpext(
                    value, target_type, "cast_fp_up"
                )
            else:
                result = self._llvm.ir_builder.fptrunc(
                    value, target_type, "cast_fp_down"
                )
        elif target_type in (
            self._llvm.ASCII_STRING_TYPE,
            self._llvm.UTF8_STRING_TYPE,
        ):
            if is_int_type(value.type):
                arg, fmt_str = self._normalize_int_for_printf(value)
                fmt_gv = self._get_or_create_format_global(fmt_str)
                ptr = self._snprintf_heap(fmt_gv, [arg])
                self.result_stack.append(ptr)
                return

            if isinstance(
                value.type, (ir.FloatType, ir.DoubleType, ir.HalfType)
            ):
                if isinstance(value.type, (ir.FloatType, ir.HalfType)):
                    value_prom = self._llvm.ir_builder.fpext(
                        value, self._llvm.DOUBLE_TYPE, "to_double"
                    )
                else:
                    value_prom = value
                fmt_gv = self._get_or_create_format_global("%.6f")
                ptr = self._snprintf_heap(fmt_gv, [value_prom])
                self.result_stack.append(ptr)
                return
            raise Exception(
                f"Unsupported cast from {value.type} to {target_type}"
            )
        else:
            raise Exception(
                f"Unsupported cast from {value.type} to {target_type}"
            )

        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.PrintExpr) -> None:
        self.visit_child(node.message)
        message_value = safe_pop(self.result_stack)
        if message_value is None:
            raise Exception("Invalid message in PrintExpr")

        message_type = message_value.type
        ptr: ir.Value
        if (
            isinstance(message_type, ir.PointerType)
            and message_type.pointee == self._llvm.INT8_TYPE
        ):
            ptr = message_value
        elif is_int_type(message_type):
            int_arg, int_fmt = self._normalize_int_for_printf(message_value)
            int_fmt_gv = self._get_or_create_format_global(int_fmt)
            ptr = self._snprintf_heap(int_fmt_gv, [int_arg])
        elif isinstance(
            message_type, (ir.HalfType, ir.FloatType, ir.DoubleType)
        ):
            if isinstance(message_type, (ir.HalfType, ir.FloatType)):
                float_arg = self._llvm.ir_builder.fpext(
                    message_value, self._llvm.DOUBLE_TYPE, "print_to_double"
                )
            else:
                float_arg = message_value
            float_fmt_gv = self._get_or_create_format_global("%.6f")
            ptr = self._snprintf_heap(float_fmt_gv, [float_arg])
        else:
            raise Exception(
                f"Unsupported message type in PrintExpr: {message_type}"
            )

        puts_fn = self.require_runtime_symbol("libc", "puts")
        self._llvm.ir_builder.call(puts_fn, [ptr])
        self.result_stack.append(ir.Constant(self._llvm.INT32_TYPE, 0))

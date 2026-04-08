# mypy: disable-error-code=no-redef

"""
title: System/runtime visitor mixins for llvmliteir.
"""

from llvmlite import ir

from irx import astx
from irx.analysis.types import is_boolean_type, is_unsigned_type
from irx.builder.core import VisitorCore
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builder.types import is_int_type
from irx.typecheck import typechecked


@typechecked
class SystemVisitorMixin(VisitorMixinBase):
    @VisitorCore.visit.dispatch
    def visit(self, node: astx.Cast) -> None:
        """
        title: Visit Cast nodes.
        parameters:
          node:
            type: astx.Cast
        """
        self.visit_child(node.value)
        value = safe_pop(self.result_stack)
        if value is None:
            raise Exception("Invalid value in Cast")

        source_type = self._resolved_ast_type(node.value)
        target_type = node.target_type
        target_llvm_type = self._llvm_type_for_ast_type(target_type)
        if target_llvm_type in (
            self._llvm.ASCII_STRING_TYPE,
            self._llvm.UTF8_STRING_TYPE,
        ):
            if (
                isinstance(value.type, ir.PointerType)
                and value.type.pointee == self._llvm.INT8_TYPE
            ):
                self.result_stack.append(value)
                return
            if is_int_type(value.type):
                arg, fmt_str = self._normalize_int_for_printf(
                    value,
                    unsigned=is_unsigned_type(source_type)
                    or is_boolean_type(source_type),
                )
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
                f"Unsupported cast from {value.type} to {target_llvm_type}"
            )
        result = self._cast_ast_value(
            value,
            source_type=source_type,
            target_type=target_type,
        )
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.PrintExpr) -> None:
        """
        title: Visit PrintExpr nodes.
        parameters:
          node:
            type: astx.PrintExpr
        """
        self.visit_child(node.message)
        message_value = safe_pop(self.result_stack)
        if message_value is None:
            raise Exception("Invalid message in PrintExpr")

        message_source_type = self._resolved_ast_type(node.message)
        message_type = message_value.type
        ptr: ir.Value
        if (
            isinstance(message_type, ir.PointerType)
            and message_type.pointee == self._llvm.INT8_TYPE
        ):
            ptr = message_value
        elif is_int_type(message_type):
            int_arg, int_fmt = self._normalize_int_for_printf(
                message_value,
                unsigned=is_unsigned_type(message_source_type)
                or is_boolean_type(message_source_type),
            )
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

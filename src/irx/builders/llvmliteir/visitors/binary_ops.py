# mypy: disable-error-code=no-redef

"""
title: Binary-operator visitor mixins for llvmliteir.
"""

import astx

from llvmlite import ir

from irx.builders.base import BuilderVisitor
from irx.builders.llvmliteir.core import (
    _semantic_flag,
    _semantic_fma_rhs,
    _semantic_symbol_key,
    _uses_unsigned_semantics,
)
from irx.builders.llvmliteir.protocols import VisitorMixinBase
from irx.builders.llvmliteir.runtime import safe_pop
from irx.builders.llvmliteir.types import is_fp_type
from irx.builders.llvmliteir.vector import emit_add, emit_int_div, is_vector


class BinaryOpVisitorMixin(VisitorMixinBase):
    @BuilderVisitor.visit.dispatch  # type: ignore[attr-defined,untyped-decorator]
    def visit(self, node: astx.BinaryOp) -> None:
        if node.op_code == "=":
            var_lhs = node.lhs
            if not isinstance(var_lhs, astx.Identifier):
                raise Exception("destination of '=' must be a variable")

            lhs_name = var_lhs.name
            lhs_key = _semantic_symbol_key(var_lhs, lhs_name)
            if lhs_key in self.const_vars:
                raise Exception(
                    f"Cannot assign to '{lhs_name}': declared as constant"
                )

            self.visit_child(node.rhs)
            llvm_rhs = safe_pop(self.result_stack)
            if llvm_rhs is None:
                raise Exception("codegen: Invalid rhs expression.")

            llvm_lhs = self.named_values.get(lhs_key)
            if not llvm_lhs:
                raise Exception("codegen: Invalid lhs variable name")

            self._llvm.ir_builder.store(llvm_rhs, llvm_lhs)
            self.result_stack.append(llvm_rhs)
            return

        self.visit_child(node.lhs)
        llvm_lhs = safe_pop(self.result_stack)
        self.visit_child(node.rhs)
        llvm_rhs = safe_pop(self.result_stack)

        if self._try_set_binary_op(llvm_lhs, llvm_rhs, node.op_code):
            return

        if llvm_lhs is None or llvm_rhs is None:
            raise Exception("codegen: Invalid lhs/rhs")

        unsigned = _uses_unsigned_semantics(node)
        if self._is_numeric_value(llvm_lhs) and self._is_numeric_value(
            llvm_rhs
        ):
            llvm_lhs, llvm_rhs = self._unify_numeric_operands(
                llvm_lhs, llvm_rhs, unsigned=unsigned
            )

        if is_vector(llvm_lhs) and is_vector(llvm_rhs):
            is_float_vec = is_fp_type(llvm_lhs.type.element)
            op = node.op_code
            set_fast = is_float_vec and _semantic_flag(node, "fast_math")
            if op == "*" and is_float_vec and _semantic_flag(node, "fma"):
                fma_rhs_node = _semantic_fma_rhs(node)
                if fma_rhs_node is None:
                    raise Exception("FMA requires a third operand (fma_rhs)")
                self.visit_child(fma_rhs_node)
                llvm_fma_rhs = safe_pop(self.result_stack)
                if llvm_fma_rhs is None:
                    raise Exception("FMA requires a valid third operand")
                if llvm_fma_rhs.type != llvm_lhs.type:
                    raise Exception(
                        f"FMA operand type mismatch: "
                        f"{llvm_lhs.type} vs {llvm_fma_rhs.type}"
                    )
                prev_fast_math = self._fast_math_enabled
                if set_fast:
                    self.set_fast_math(True)
                try:
                    result = self._emit_fma(llvm_lhs, llvm_rhs, llvm_fma_rhs)
                finally:
                    self.set_fast_math(prev_fast_math)
                self.result_stack.append(result)
                return

            prev_fast_math = self._fast_math_enabled
            if set_fast:
                self.set_fast_math(True)
            try:
                if op == "+":
                    if is_float_vec:
                        result = self._llvm.ir_builder.fadd(
                            llvm_lhs, llvm_rhs, name="vfaddtmp"
                        )
                        self._apply_fast_math(result)
                    else:
                        result = self._llvm.ir_builder.add(
                            llvm_lhs, llvm_rhs, name="vaddtmp"
                        )
                elif op == "-":
                    if is_float_vec:
                        result = self._llvm.ir_builder.fsub(
                            llvm_lhs, llvm_rhs, name="vfsubtmp"
                        )
                        self._apply_fast_math(result)
                    else:
                        result = self._llvm.ir_builder.sub(
                            llvm_lhs, llvm_rhs, name="vsubtmp"
                        )
                elif op == "*":
                    if is_float_vec:
                        result = self._llvm.ir_builder.fmul(
                            llvm_lhs, llvm_rhs, name="vfmultmp"
                        )
                        self._apply_fast_math(result)
                    else:
                        result = self._llvm.ir_builder.mul(
                            llvm_lhs, llvm_rhs, name="vmultmp"
                        )
                elif op == "/":
                    if is_float_vec:
                        result = self._llvm.ir_builder.fdiv(
                            llvm_lhs, llvm_rhs, name="vfdivtmp"
                        )
                        self._apply_fast_math(result)
                    else:
                        result = emit_int_div(
                            self._llvm.ir_builder, llvm_lhs, llvm_rhs, unsigned
                        )
                else:
                    raise Exception(f"Vector binop {op} not implemented.")
            finally:
                self.set_fast_math(prev_fast_math)

            self.result_stack.append(result)
            return

        if node.op_code in ("&&", "and"):
            result = self._llvm.ir_builder.and_(llvm_lhs, llvm_rhs, "andtmp")
            self.result_stack.append(result)
            return

        if node.op_code in ("||", "or"):
            result = self._llvm.ir_builder.or_(llvm_lhs, llvm_rhs, "ortmp")
            self.result_stack.append(result)
            return

        if node.op_code == "+":
            if (
                isinstance(llvm_lhs.type, ir.PointerType)
                and isinstance(llvm_rhs.type, ir.PointerType)
                and llvm_lhs.type.pointee == self._llvm.INT8_TYPE
                and llvm_rhs.type.pointee == self._llvm.INT8_TYPE
            ):
                result = self._handle_string_concatenation(llvm_lhs, llvm_rhs)
            else:
                result = emit_add(
                    self._llvm.ir_builder, llvm_lhs, llvm_rhs, "addtmp"
                )
            self.result_stack.append(result)
            return

        if node.op_code == "-":
            if is_fp_type(llvm_lhs.type):
                result = self._llvm.ir_builder.fsub(
                    llvm_lhs, llvm_rhs, "subtmp"
                )
                self._apply_fast_math(result)
            else:
                result = self._llvm.ir_builder.sub(
                    llvm_lhs, llvm_rhs, "subtmp"
                )
            self.result_stack.append(result)
            return

        if node.op_code == "*":
            if is_fp_type(llvm_lhs.type):
                result = self._llvm.ir_builder.fmul(
                    llvm_lhs, llvm_rhs, "multmp"
                )
                self._apply_fast_math(result)
            else:
                result = self._llvm.ir_builder.mul(
                    llvm_lhs, llvm_rhs, "multmp"
                )
            self.result_stack.append(result)
            return

        if node.op_code == "<":
            if is_fp_type(llvm_lhs.type):
                result = self._llvm.ir_builder.fcmp_ordered(
                    "<", llvm_lhs, llvm_rhs, "lttmp"
                )
            elif unsigned:
                result = self._llvm.ir_builder.icmp_unsigned(
                    "<", llvm_lhs, llvm_rhs, "lttmp"
                )
            else:
                result = self._llvm.ir_builder.icmp_signed(
                    "<", llvm_lhs, llvm_rhs, "lttmp"
                )
            self.result_stack.append(result)
            return

        if node.op_code == ">":
            if is_fp_type(llvm_lhs.type):
                result = self._llvm.ir_builder.fcmp_ordered(
                    ">", llvm_lhs, llvm_rhs, "gttmp"
                )
            elif unsigned:
                result = self._llvm.ir_builder.icmp_unsigned(
                    ">", llvm_lhs, llvm_rhs, "gttmp"
                )
            else:
                result = self._llvm.ir_builder.icmp_signed(
                    ">", llvm_lhs, llvm_rhs, "gttmp"
                )
            self.result_stack.append(result)
            return

        if node.op_code == "<=":
            if is_fp_type(llvm_lhs.type):
                result = self._llvm.ir_builder.fcmp_ordered(
                    "<=", llvm_lhs, llvm_rhs, "letmp"
                )
            elif unsigned:
                result = self._llvm.ir_builder.icmp_unsigned(
                    "<=", llvm_lhs, llvm_rhs, "letmp"
                )
            else:
                result = self._llvm.ir_builder.icmp_signed(
                    "<=", llvm_lhs, llvm_rhs, "letmp"
                )
            self.result_stack.append(result)
            return

        if node.op_code == ">=":
            if is_fp_type(llvm_lhs.type):
                result = self._llvm.ir_builder.fcmp_ordered(
                    ">=", llvm_lhs, llvm_rhs, "getmp"
                )
            elif unsigned:
                result = self._llvm.ir_builder.icmp_unsigned(
                    ">=", llvm_lhs, llvm_rhs, "getmp"
                )
            else:
                result = self._llvm.ir_builder.icmp_signed(
                    ">=", llvm_lhs, llvm_rhs, "getmp"
                )
            self.result_stack.append(result)
            return

        if node.op_code == "/":
            if is_fp_type(llvm_lhs.type):
                result = self._llvm.ir_builder.fdiv(
                    llvm_lhs, llvm_rhs, "divtmp"
                )
                self._apply_fast_math(result)
            elif unsigned:
                result = self._llvm.ir_builder.udiv(
                    llvm_lhs, llvm_rhs, "divtmp"
                )
            else:
                result = self._llvm.ir_builder.sdiv(
                    llvm_lhs, llvm_rhs, "divtmp"
                )
            self.result_stack.append(result)
            return

        if node.op_code == "==":
            if (
                isinstance(llvm_lhs.type, ir.PointerType)
                and isinstance(llvm_rhs.type, ir.PointerType)
                and llvm_lhs.type.pointee == self._llvm.INT8_TYPE
                and llvm_rhs.type.pointee == self._llvm.INT8_TYPE
            ):
                cmp_result = self._handle_string_comparison(
                    llvm_lhs, llvm_rhs, "=="
                )
            elif is_fp_type(llvm_lhs.type):
                cmp_result = self._llvm.ir_builder.fcmp_ordered(
                    "==", llvm_lhs, llvm_rhs, "eqtmp"
                )
            elif unsigned:
                cmp_result = self._llvm.ir_builder.icmp_unsigned(
                    "==", llvm_lhs, llvm_rhs, "eqtmp"
                )
            else:
                cmp_result = self._llvm.ir_builder.icmp_signed(
                    "==", llvm_lhs, llvm_rhs, "eqtmp"
                )
            self.result_stack.append(cmp_result)
            return

        if node.op_code == "!=":
            if (
                isinstance(llvm_lhs.type, ir.PointerType)
                and isinstance(llvm_rhs.type, ir.PointerType)
                and llvm_lhs.type.pointee == self._llvm.INT8_TYPE
                and llvm_rhs.type.pointee == self._llvm.INT8_TYPE
            ):
                cmp_result = self._handle_string_comparison(
                    llvm_lhs, llvm_rhs, "!="
                )
            elif is_fp_type(llvm_lhs.type):
                cmp_result = self._llvm.ir_builder.fcmp_ordered(
                    "!=", llvm_lhs, llvm_rhs, "netmp"
                )
            elif unsigned:
                cmp_result = self._llvm.ir_builder.icmp_unsigned(
                    "!=", llvm_lhs, llvm_rhs, "netmp"
                )
            else:
                cmp_result = self._llvm.ir_builder.icmp_signed(
                    "!=", llvm_lhs, llvm_rhs, "netmp"
                )
            self.result_stack.append(cmp_result)
            return

        if node.op_code == "%":
            if is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                result = self._llvm.ir_builder.frem(
                    llvm_lhs, llvm_rhs, "fremtmp"
                )
            elif unsigned:
                result = self._llvm.ir_builder.urem(
                    llvm_lhs, llvm_rhs, "uremtmp"
                )
            else:
                result = self._llvm.ir_builder.srem(
                    llvm_lhs, llvm_rhs, "sremtmp"
                )
            self.result_stack.append(result)
            return

        raise Exception(f"Binary op {node.op_code} not implemented yet.")

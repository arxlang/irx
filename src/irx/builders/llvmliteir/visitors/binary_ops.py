# mypy: disable-error-code=no-redef

"""
title: Binary-operator visitor mixins for llvmliteir.
"""

from __future__ import annotations

import astx

from llvmlite import ir

from irx.analysis.resolved_nodes import (
    SPECIALIZED_BINARY_OP_EXTRA,
    AddBinOp,
    AssignmentBinOp,
    BitAndBinOp,
    BitOrBinOp,
    BitXorBinOp,
    DivBinOp,
    EqBinOp,
    GeBinOp,
    GtBinOp,
    LeBinOp,
    LogicalAndBinOp,
    LogicalOrBinOp,
    LtBinOp,
    ModBinOp,
    MulBinOp,
    NeBinOp,
    SubBinOp,
    specialize_binary_op,
)
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
    def _resolved_binary_variant(self, node: astx.BinaryOp) -> astx.BinaryOp:
        semantic = getattr(node, "semantic", None)
        extras = getattr(semantic, "extras", None)
        if isinstance(extras, dict):
            specialized = extras.get(SPECIALIZED_BINARY_OP_EXTRA)
            if isinstance(specialized, astx.BinaryOp):
                return specialized
        return specialize_binary_op(node)

    def _load_binary_operands(
        self,
        node: astx.BinaryOp,
        *,
        unify_numeric: bool = True,
    ) -> tuple[ir.Value, ir.Value, bool]:
        self.visit_child(node.lhs)
        llvm_lhs = safe_pop(self.result_stack)
        self.visit_child(node.rhs)
        llvm_rhs = safe_pop(self.result_stack)

        if llvm_lhs is None or llvm_rhs is None:
            raise Exception("codegen: Invalid lhs/rhs")

        unsigned = _uses_unsigned_semantics(node)
        if (
            unify_numeric
            and self._is_numeric_value(llvm_lhs)
            and self._is_numeric_value(llvm_rhs)
        ):
            llvm_lhs, llvm_rhs = self._unify_numeric_operands(
                llvm_lhs,
                llvm_rhs,
                unsigned=unsigned,
            )

        return llvm_lhs, llvm_rhs, unsigned

    def _emit_vector_add(
        self,
        node: AddBinOp,
        llvm_lhs: ir.Value,
        llvm_rhs: ir.Value,
    ) -> ir.Value | None:
        if not (is_vector(llvm_lhs) and is_vector(llvm_rhs)):
            return None

        is_float_vec = is_fp_type(llvm_lhs.type.element)
        prev_fast_math = self._fast_math_enabled
        if is_float_vec and _semantic_flag(node, "fast_math"):
            self.set_fast_math(True)
        try:
            if is_float_vec:
                result = self._llvm.ir_builder.fadd(
                    llvm_lhs, llvm_rhs, name="vfaddtmp"
                )
                self._apply_fast_math(result)
            else:
                result = self._llvm.ir_builder.add(
                    llvm_lhs, llvm_rhs, name="vaddtmp"
                )
        finally:
            self.set_fast_math(prev_fast_math)
        return result

    def _emit_vector_sub(
        self,
        node: SubBinOp,
        llvm_lhs: ir.Value,
        llvm_rhs: ir.Value,
    ) -> ir.Value | None:
        if not (is_vector(llvm_lhs) and is_vector(llvm_rhs)):
            return None

        is_float_vec = is_fp_type(llvm_lhs.type.element)
        prev_fast_math = self._fast_math_enabled
        if is_float_vec and _semantic_flag(node, "fast_math"):
            self.set_fast_math(True)
        try:
            if is_float_vec:
                result = self._llvm.ir_builder.fsub(
                    llvm_lhs, llvm_rhs, name="vfsubtmp"
                )
                self._apply_fast_math(result)
            else:
                result = self._llvm.ir_builder.sub(
                    llvm_lhs, llvm_rhs, name="vsubtmp"
                )
        finally:
            self.set_fast_math(prev_fast_math)
        return result

    def _emit_vector_mul(
        self,
        node: MulBinOp,
        llvm_lhs: ir.Value,
        llvm_rhs: ir.Value,
    ) -> ir.Value | None:
        if not (is_vector(llvm_lhs) and is_vector(llvm_rhs)):
            return None

        is_float_vec = is_fp_type(llvm_lhs.type.element)
        set_fast = is_float_vec and _semantic_flag(node, "fast_math")
        if _semantic_flag(node, "fma") and is_float_vec:
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
                return self._emit_fma(llvm_lhs, llvm_rhs, llvm_fma_rhs)
            finally:
                self.set_fast_math(prev_fast_math)

        prev_fast_math = self._fast_math_enabled
        if set_fast:
            self.set_fast_math(True)
        try:
            if is_float_vec:
                result = self._llvm.ir_builder.fmul(
                    llvm_lhs, llvm_rhs, name="vfmultmp"
                )
                self._apply_fast_math(result)
            else:
                result = self._llvm.ir_builder.mul(
                    llvm_lhs, llvm_rhs, name="vmultmp"
                )
        finally:
            self.set_fast_math(prev_fast_math)
        return result

    def _emit_vector_div(
        self,
        node: DivBinOp,
        llvm_lhs: ir.Value,
        llvm_rhs: ir.Value,
        *,
        unsigned: bool,
    ) -> ir.Value | None:
        if not (is_vector(llvm_lhs) and is_vector(llvm_rhs)):
            return None

        is_float_vec = is_fp_type(llvm_lhs.type.element)
        prev_fast_math = self._fast_math_enabled
        if is_float_vec and _semantic_flag(node, "fast_math"):
            self.set_fast_math(True)
        try:
            if is_float_vec:
                result = self._llvm.ir_builder.fdiv(
                    llvm_lhs, llvm_rhs, name="vfdivtmp"
                )
                self._apply_fast_math(result)
            else:
                result = emit_int_div(
                    self._llvm.ir_builder, llvm_lhs, llvm_rhs, unsigned
                )
        finally:
            self.set_fast_math(prev_fast_math)
        return result

    def _emit_ordered_compare(
        self,
        op_code: str,
        llvm_lhs: ir.Value,
        llvm_rhs: ir.Value,
        *,
        unsigned: bool,
        name: str,
    ) -> ir.Value:
        if is_fp_type(llvm_lhs.type):
            return self._llvm.ir_builder.fcmp_ordered(
                op_code,
                llvm_lhs,
                llvm_rhs,
                name,
            )
        if unsigned:
            return self._llvm.ir_builder.icmp_unsigned(
                op_code,
                llvm_lhs,
                llvm_rhs,
                name,
            )
        return self._llvm.ir_builder.icmp_signed(
            op_code,
            llvm_lhs,
            llvm_rhs,
            name,
        )

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.BinaryOp) -> None:
        specialized = self._resolved_binary_variant(node)
        if specialized is node:
            raise Exception(f"Binary op {node.op_code} not implemented yet.")
        self.visit_child(specialized)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: AssignmentBinOp) -> None:
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

    @BuilderVisitor.visit.dispatch
    def visit(self, node: AddBinOp) -> None:
        llvm_lhs, llvm_rhs, _unsigned = self._load_binary_operands(node)

        vector_result = self._emit_vector_add(node, llvm_lhs, llvm_rhs)
        if vector_result is not None:
            self.result_stack.append(vector_result)
            return

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

    @BuilderVisitor.visit.dispatch
    def visit(self, node: SubBinOp) -> None:
        llvm_lhs, llvm_rhs, _unsigned = self._load_binary_operands(node)

        if self._try_set_binary_op(llvm_lhs, llvm_rhs, node.op_code):
            return

        vector_result = self._emit_vector_sub(node, llvm_lhs, llvm_rhs)
        if vector_result is not None:
            self.result_stack.append(vector_result)
            return

        if is_fp_type(llvm_lhs.type):
            result = self._llvm.ir_builder.fsub(llvm_lhs, llvm_rhs, "subtmp")
            self._apply_fast_math(result)
        else:
            result = self._llvm.ir_builder.sub(llvm_lhs, llvm_rhs, "subtmp")
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: MulBinOp) -> None:
        llvm_lhs, llvm_rhs, _unsigned = self._load_binary_operands(node)

        vector_result = self._emit_vector_mul(node, llvm_lhs, llvm_rhs)
        if vector_result is not None:
            self.result_stack.append(vector_result)
            return

        if is_fp_type(llvm_lhs.type):
            result = self._llvm.ir_builder.fmul(llvm_lhs, llvm_rhs, "multmp")
            self._apply_fast_math(result)
        else:
            result = self._llvm.ir_builder.mul(llvm_lhs, llvm_rhs, "multmp")
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: DivBinOp) -> None:
        llvm_lhs, llvm_rhs, unsigned = self._load_binary_operands(node)

        vector_result = self._emit_vector_div(
            node,
            llvm_lhs,
            llvm_rhs,
            unsigned=unsigned,
        )
        if vector_result is not None:
            self.result_stack.append(vector_result)
            return

        if is_fp_type(llvm_lhs.type):
            result = self._llvm.ir_builder.fdiv(llvm_lhs, llvm_rhs, "divtmp")
            self._apply_fast_math(result)
        elif unsigned:
            result = self._llvm.ir_builder.udiv(llvm_lhs, llvm_rhs, "divtmp")
        else:
            result = self._llvm.ir_builder.sdiv(llvm_lhs, llvm_rhs, "divtmp")
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: ModBinOp) -> None:
        llvm_lhs, llvm_rhs, unsigned = self._load_binary_operands(node)

        if is_vector(llvm_lhs) and is_vector(llvm_rhs):
            raise Exception(f"Vector binop {node.op_code} not implemented.")

        if is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
            result = self._llvm.ir_builder.frem(llvm_lhs, llvm_rhs, "fremtmp")
        elif unsigned:
            result = self._llvm.ir_builder.urem(llvm_lhs, llvm_rhs, "uremtmp")
        else:
            result = self._llvm.ir_builder.srem(llvm_lhs, llvm_rhs, "sremtmp")
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: LogicalAndBinOp) -> None:
        llvm_lhs, llvm_rhs, _unsigned = self._load_binary_operands(node)
        result = self._llvm.ir_builder.and_(llvm_lhs, llvm_rhs, "andtmp")
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: LogicalOrBinOp) -> None:
        llvm_lhs, llvm_rhs, _unsigned = self._load_binary_operands(node)
        result = self._llvm.ir_builder.or_(llvm_lhs, llvm_rhs, "ortmp")
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: LtBinOp) -> None:
        llvm_lhs, llvm_rhs, unsigned = self._load_binary_operands(node)
        if is_vector(llvm_lhs) and is_vector(llvm_rhs):
            raise Exception(f"Vector binop {node.op_code} not implemented.")
        result = self._emit_ordered_compare(
            "<",
            llvm_lhs,
            llvm_rhs,
            unsigned=unsigned,
            name="lttmp",
        )
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: GtBinOp) -> None:
        llvm_lhs, llvm_rhs, unsigned = self._load_binary_operands(node)
        if is_vector(llvm_lhs) and is_vector(llvm_rhs):
            raise Exception(f"Vector binop {node.op_code} not implemented.")
        result = self._emit_ordered_compare(
            ">",
            llvm_lhs,
            llvm_rhs,
            unsigned=unsigned,
            name="gttmp",
        )
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: LeBinOp) -> None:
        llvm_lhs, llvm_rhs, unsigned = self._load_binary_operands(node)
        if is_vector(llvm_lhs) and is_vector(llvm_rhs):
            raise Exception(f"Vector binop {node.op_code} not implemented.")
        result = self._emit_ordered_compare(
            "<=",
            llvm_lhs,
            llvm_rhs,
            unsigned=unsigned,
            name="letmp",
        )
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: GeBinOp) -> None:
        llvm_lhs, llvm_rhs, unsigned = self._load_binary_operands(node)
        if is_vector(llvm_lhs) and is_vector(llvm_rhs):
            raise Exception(f"Vector binop {node.op_code} not implemented.")
        result = self._emit_ordered_compare(
            ">=",
            llvm_lhs,
            llvm_rhs,
            unsigned=unsigned,
            name="getmp",
        )
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: EqBinOp) -> None:
        llvm_lhs, llvm_rhs, unsigned = self._load_binary_operands(node)

        if is_vector(llvm_lhs) and is_vector(llvm_rhs):
            raise Exception(f"Vector binop {node.op_code} not implemented.")

        if (
            isinstance(llvm_lhs.type, ir.PointerType)
            and isinstance(llvm_rhs.type, ir.PointerType)
            and llvm_lhs.type.pointee == self._llvm.INT8_TYPE
            and llvm_rhs.type.pointee == self._llvm.INT8_TYPE
        ):
            result = self._handle_string_comparison(llvm_lhs, llvm_rhs, "==")
        elif is_fp_type(llvm_lhs.type):
            result = self._llvm.ir_builder.fcmp_ordered(
                "==",
                llvm_lhs,
                llvm_rhs,
                "eqtmp",
            )
        elif unsigned:
            result = self._llvm.ir_builder.icmp_unsigned(
                "==",
                llvm_lhs,
                llvm_rhs,
                "eqtmp",
            )
        else:
            result = self._llvm.ir_builder.icmp_signed(
                "==",
                llvm_lhs,
                llvm_rhs,
                "eqtmp",
            )
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: NeBinOp) -> None:
        llvm_lhs, llvm_rhs, unsigned = self._load_binary_operands(node)

        if is_vector(llvm_lhs) and is_vector(llvm_rhs):
            raise Exception(f"Vector binop {node.op_code} not implemented.")

        if (
            isinstance(llvm_lhs.type, ir.PointerType)
            and isinstance(llvm_rhs.type, ir.PointerType)
            and llvm_lhs.type.pointee == self._llvm.INT8_TYPE
            and llvm_rhs.type.pointee == self._llvm.INT8_TYPE
        ):
            result = self._handle_string_comparison(llvm_lhs, llvm_rhs, "!=")
        elif is_fp_type(llvm_lhs.type):
            result = self._llvm.ir_builder.fcmp_ordered(
                "!=",
                llvm_lhs,
                llvm_rhs,
                "netmp",
            )
        elif unsigned:
            result = self._llvm.ir_builder.icmp_unsigned(
                "!=",
                llvm_lhs,
                llvm_rhs,
                "netmp",
            )
        else:
            result = self._llvm.ir_builder.icmp_signed(
                "!=",
                llvm_lhs,
                llvm_rhs,
                "netmp",
            )
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: BitOrBinOp) -> None:
        llvm_lhs, llvm_rhs, _unsigned = self._load_binary_operands(
            node,
            unify_numeric=False,
        )
        if self._try_set_binary_op(llvm_lhs, llvm_rhs, node.op_code):
            return
        raise Exception(f"Binary op {node.op_code} not implemented yet.")

    @BuilderVisitor.visit.dispatch
    def visit(self, node: BitAndBinOp) -> None:
        llvm_lhs, llvm_rhs, _unsigned = self._load_binary_operands(
            node,
            unify_numeric=False,
        )
        if self._try_set_binary_op(llvm_lhs, llvm_rhs, node.op_code):
            return
        raise Exception(f"Binary op {node.op_code} not implemented yet.")

    @BuilderVisitor.visit.dispatch
    def visit(self, node: BitXorBinOp) -> None:
        llvm_lhs, llvm_rhs, _unsigned = self._load_binary_operands(
            node,
            unify_numeric=False,
        )
        if self._try_set_binary_op(llvm_lhs, llvm_rhs, node.op_code):
            return
        raise Exception(f"Binary op {node.op_code} not implemented yet.")

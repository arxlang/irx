"""LLVM-IR builder."""

from __future__ import annotations

import ctypes
import os
import tempfile

from datetime import datetime
from typing import Any, Callable, Optional, cast

import astx
import xh

from llvmlite import binding as llvm
from llvmlite import ir
from plum import dispatch
from public import public

from irx import system
from irx.builders.base import Builder, BuilderVisitor
from irx.tools.typing import typechecked


@typechecked
def safe_pop(lst: list[ir.Value | ir.Function]) -> ir.Value | ir.Function | None:
    try:
        return lst.pop()
    except IndexError:
        return None


@typechecked
class VariablesLLVM:
    """Store all LLVM variables used during code generation."""

    FLOAT_TYPE: ir.Type
    FLOAT16_TYPE: ir.Type
    DOUBLE_TYPE: ir.Type
    INT8_TYPE: ir.Type
    INT16_TYPE: ir.Type
    INT32_TYPE: ir.Type
    INT64_TYPE: ir.Type
    BOOLEAN_TYPE: ir.Type
    VOID_TYPE: ir.Type

    STRING_PTR_TYPE: ir.Type  # i8*

    TIMESTAMP_TYPE: ir.Type
    SIZE_T_TYPE: ir.Type
    POINTER_BITS: int

    context: ir.Context
    module: ir.Module
    ir_builder: ir.IRBuilder

    def get_data_type(self, type_name: str) -> ir.Type:
        if type_name == "float32":
            return self.FLOAT_TYPE
        elif type_name == "float16":
            return self.FLOAT16_TYPE
        elif type_name == "double":
            return self.DOUBLE_TYPE
        elif type_name == "boolean":
            return self.BOOLEAN_TYPE
        elif type_name == "int8":
            return self.INT8_TYPE
        elif type_name == "int16":
            return self.INT16_TYPE
        elif type_name == "int32":
            return self.INT32_TYPE
        elif type_name == "int64":
            return self.INT64_TYPE
        elif type_name == "char":
            return self.INT8_TYPE
        elif type_name in ("string", "stringascii", "utf8string"):
            return self.STRING_PTR_TYPE
        elif type_name == "nonetype":
            return self.VOID_TYPE

        raise Exception(f"[EE]: Type name {type_name} not valid.")


@typechecked
class LLVMLiteIRVisitor(BuilderVisitor):
    """LLVM-IR Translator."""

    named_values: dict[str, Any] = {}
    _llvm: VariablesLLVM
    function_protos: dict[str, astx.FunctionPrototype]
    result_stack: list[ir.Value | ir.Function | None] = []

    def __init__(self) -> None:
        super().__init__()

        # named_values as instance variable so it isn't shared across instances
        self.named_values: dict[str, Any] = {}
        self.function_protos: dict[str, astx.FunctionPrototype] = {}
        self.result_stack: list[ir.Value | ir.Function] = []
        self._fast_math_enabled = False

        self.initialize()

        self.target = llvm.Target.from_default_triple()
        self.target_machine = self.target.create_target_machine(
            codemodel="small"
        )

        self._add_builtins()

    def _init_native_size_types(self) -> None:
        self._llvm.POINTER_BITS = ctypes.sizeof(ctypes.c_void_p) * 8
        self._llvm.SIZE_T_TYPE = ir.IntType(ctypes.sizeof(ctypes.c_size_t) * 8)

    def initialize(self) -> None:
        self._llvm = VariablesLLVM()
        self._llvm.module = ir.Module("Arx")

        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

        self._llvm.ir_builder = ir.IRBuilder()

        self._llvm.FLOAT_TYPE = ir.FloatType()
        self._llvm.FLOAT16_TYPE = ir.HalfType()
        self._llvm.DOUBLE_TYPE = ir.DoubleType()
        self._llvm.BOOLEAN_TYPE = ir.IntType(1)
        self._llvm.INT8_TYPE = ir.IntType(8)
        self._llvm.INT16_TYPE = ir.IntType(16)
        self._llvm.INT32_TYPE = ir.IntType(32)
        self._llvm.INT64_TYPE = ir.IntType(64)
        self._llvm.VOID_TYPE = ir.VoidType()

        # ✅ SINGLE STRING REPRESENTATION
        self._llvm.STRING_PTR_TYPE = ir.IntType(8).as_pointer()

        self._llvm.TIMESTAMP_TYPE = ir.LiteralStructType(
            [self._llvm.INT32_TYPE] * 7
        )

        self._llvm.SIZE_T_TYPE = ir.IntType(64)

    # ------------------------------------------------------------
    # STRING HELPERS (ALL i8*)
    # ------------------------------------------------------------

    def _create_strlen_inline(self) -> ir.Function:
        name = "strlen_inline"
        if name in self._llvm.module.globals:
            return self._llvm.module.get_global(name)

        fn = ir.Function(
            self._llvm.module,
            ir.FunctionType(
                self._llvm.INT32_TYPE,
                [self._llvm.STRING_PTR_TYPE],
            ),
            name=name,
        )

        entry = fn.append_basic_block("entry")
        loop = fn.append_basic_block("loop")
        end = fn.append_basic_block("end")

        b = ir.IRBuilder(entry)
        idx = b.alloca(self._llvm.INT32_TYPE)
        b.store(ir.Constant(self._llvm.INT32_TYPE, 0), idx)
        b.branch(loop)

        b.position_at_start(loop)
        i = b.load(idx)
        ch = b.load(b.gep(fn.args[0], [i], inbounds=True))
        is_null = b.icmp_signed("==", ch, ir.Constant(self._llvm.INT8_TYPE, 0))
        b.store(b.add(i, ir.Constant(self._llvm.INT32_TYPE, 1)), idx)
        b.cbranch(is_null, end, loop)

        b.position_at_start(end)
        b.ret(b.load(idx))
        return fn

    def _create_strcmp_inline(self) -> ir.Function:
        name = "strcmp_inline"
        if name in self._llvm.module.globals:
            return self._llvm.module.get_global(name)

        fn_ty = ir.FunctionType(ty, [ty, ty, ty])
        fn = ir.Function(self._llvm.module, fn_ty, name)
        fn.linkage = "external"
        return fn

    def _emit_fma(
        self, lhs: ir.Value, rhs: ir.Value, addend: ir.Value
    ) -> ir.Value:
        """Emit a fused multiply-add, using intrinsic fallback if needed."""
        builder = self._llvm.ir_builder
        if hasattr(builder, "fma"):
            return builder.fma(lhs, rhs, addend, name="vfma")

        fma_fn = self._get_fma_function(lhs.type)
        inst = builder.call(fma_fn, [lhs, rhs, addend], name="vfma")
        self._apply_fast_math(inst)
        return inst

    def set_fast_math(self, enabled: bool) -> None:
        """Enable/disable fast-math flags for subsequent FP instructions."""
        self._fast_math_enabled = enabled

    def _apply_fast_math(self, inst: ir.Instruction) -> None:
        """Attach fast-math flags when enabled and applicable."""
        if not self._fast_math_enabled:
            return
        ty = inst.type
        if isinstance(ty, ir.VectorType):
            if not is_fp_type(ty.element):
                return
        elif not is_fp_type(ty):
            return

        flags = getattr(inst, "flags", None)
        if flags is None:
            return

        if "fast" in flags:
            return

        try:
            flags.append("fast")
        except (AttributeError, TypeError):
            return

    @dispatch.abstract
    def visit(self, node: astx.AST) -> None:
        """Translate an ASTx expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.UnaryOp) -> None:
        """Translate an ASTx UnaryOp expression."""
        if node.op_code == "++":
            self.visit(node.operand)
            operand_val = safe_pop(self.result_stack)

            one = ir.Constant(operand_val.type, 1)

            # Perform the increment operation
            result = self._llvm.ir_builder.add(operand_val, one, "inctmp")

            # If operand is a variable, store the new value back
            if isinstance(node.operand, astx.Identifier):
                var_addr = self.named_values.get(node.operand.name)
                if var_addr:
                    self._llvm.ir_builder.store(result, var_addr)

            self.result_stack.append(result)
            return

        elif node.op_code == "--":
            self.visit(node.operand)
            operand_val = safe_pop(self.result_stack)
            one = ir.Constant(operand_val.type, 1)
            result = self._llvm.ir_builder.sub(operand_val, one, "dectmp")

            if isinstance(node.operand, astx.Identifier):
                var_addr = self.named_values.get(node.operand.name)
                if var_addr:
                    self._llvm.ir_builder.store(result, var_addr)

            self.result_stack.append(result)
            return

        elif node.op_code == "!":
            self.visit(node.operand)
            val = safe_pop(self.result_stack)
            result = self._llvm.ir_builder.xor(
                val, ir.Constant(val.type, 1), "nottmp"
            )

            if isinstance(node.operand, astx.Identifier):
                addr = self.named_values.get(node.operand.name)
                if addr:
                    self._llvm.ir_builder.store(result, addr)

            self.result_stack.append(result)
            return

        raise Exception(f"Unary operator {node.op_code} not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.BinaryOp) -> None:
        """Translate binary operation expression."""
        if node.op_code == "=":
            # Special case '=' because we don't want to emit the lhs as an
            # expression.
            # Assignment requires the lhs to be an identifier.
            # This assumes we're building without RTTI because LLVM builds
            # that way by default.
            # If you build LLVM with RTTI, this can be changed to a
            # dynamic_cast for automatic error checking.
            var_lhs = node.lhs

            if not isinstance(var_lhs, astx.VariableExprAST):
                raise Exception("destination of '=' must be a variable")

            # Codegen the rhs.
            self.visit(node.rhs)
            llvm_rhs = safe_pop(self.result_stack)

            if not llvm_rhs:
                raise Exception("codegen: Invalid rhs expression.")

            llvm_lhs = self.named_values.get(var_lhs.get_name())

            if not llvm_lhs:
                raise Exception("codegen: Invalid lhs variable name")

            self._llvm.ir_builder.store(llvm_rhs, llvm_lhs)
            result = llvm_rhs
            self.result_stack.append(result)
            return

        self.visit(node.lhs)
        llvm_lhs = safe_pop(self.result_stack)

        self.visit(node.rhs)
        llvm_rhs = safe_pop(self.result_stack)

        if not llvm_lhs or not llvm_rhs:
            raise Exception("codegen: Invalid lhs/rhs")

        # Scalar-vector promotion: one vector + matching scalar -> splat scalar
        lhs_is_vec = is_vector(llvm_lhs)
        rhs_is_vec = is_vector(llvm_rhs)
        if lhs_is_vec and not rhs_is_vec:
            elem_ty = llvm_lhs.type.element
            if llvm_rhs.type == elem_ty:
                llvm_rhs = splat_scalar(
                    self._llvm.ir_builder, llvm_rhs, llvm_lhs.type
                )
            elif is_fp_type(elem_ty) and is_fp_type(llvm_rhs.type):
                if isinstance(elem_ty, FloatType) and isinstance(
                    llvm_rhs.type, DoubleType
                ):
                    llvm_rhs = self._llvm.ir_builder.fptrunc(
                        llvm_rhs, elem_ty, "vec_promote_scalar"
                    )
                    llvm_rhs = splat_scalar(
                        self._llvm.ir_builder, llvm_rhs, llvm_lhs.type
                    )
                elif isinstance(elem_ty, DoubleType) and isinstance(
                    llvm_rhs.type, FloatType
                ):
                    llvm_rhs = self._llvm.ir_builder.fpext(
                        llvm_rhs, elem_ty, "vec_promote_scalar"
                    )
                    llvm_rhs = splat_scalar(
                        self._llvm.ir_builder, llvm_rhs, llvm_lhs.type
                    )
        elif rhs_is_vec and not lhs_is_vec:
            elem_ty = llvm_rhs.type.element
            if llvm_lhs.type == elem_ty:
                llvm_lhs = splat_scalar(
                    self._llvm.ir_builder, llvm_lhs, llvm_rhs.type
                )
            elif is_fp_type(elem_ty) and is_fp_type(llvm_lhs.type):
                if isinstance(elem_ty, FloatType) and isinstance(
                    llvm_lhs.type, DoubleType
                ):
                    llvm_lhs = self._llvm.ir_builder.fptrunc(
                        llvm_lhs, elem_ty, "vec_promote_scalar"
                    )
                    llvm_lhs = splat_scalar(
                        self._llvm.ir_builder, llvm_lhs, llvm_rhs.type
                    )
                elif isinstance(elem_ty, DoubleType) and isinstance(
                    llvm_lhs.type, FloatType
                ):
                    llvm_lhs = self._llvm.ir_builder.fpext(
                        llvm_lhs, elem_ty, "vec_promote_scalar"
                    )
                    llvm_lhs = splat_scalar(
                        self._llvm.ir_builder, llvm_lhs, llvm_rhs.type
                    )

        # If both operands are LLVM vectors, handle as vector ops
        if is_vector(llvm_lhs) and is_vector(llvm_rhs):
            if llvm_lhs.type.count != llvm_rhs.type.count:
                raise Exception(
                    f"Vector size mismatch: {llvm_lhs.type} vs {llvm_rhs.type}"
                )
            if llvm_lhs.type.element != llvm_rhs.type.element:
                raise Exception(
                    f"Vector element type mismatch: "
                    f"{llvm_lhs.type.element} vs {llvm_rhs.type.element}"
                )
            is_float_vec = is_fp_type(llvm_lhs.type.element)
            op = node.op_code
            set_fast = is_float_vec and getattr(node, "fast_math", False)
            if op == "*" and is_float_vec and getattr(node, "fma", False):
                if not hasattr(node, "fma_rhs"):
                    raise Exception("FMA requires a third operand (fma_rhs)")
                self.visit(node.fma_rhs)
                llvm_fma_rhs = safe_pop(self.result_stack)
                if llvm_fma_rhs.type != llvm_lhs.type:
                    raise Exception(
                        f"FMA operand type mismatch: "
                        f"{llvm_lhs.type} vs {llvm_fma_rhs.type}"
                    )
                if set_fast:
                    self.set_fast_math(True)
                try:
                    result = self._emit_fma(llvm_lhs, llvm_rhs, llvm_fma_rhs)
                finally:
                    if set_fast:
                        self.set_fast_math(False)
                self.result_stack.append(result)
                return
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
                        unsigned = getattr(node, "unsigned", None)
                        if unsigned is None:
                            raise Exception(
                                "Cannot infer integer division signedness "
                                "for vector op"
                            )
                        result = emit_int_div(
                            self._llvm.ir_builder, llvm_lhs, llvm_rhs, unsigned
                        )
                else:
                    raise Exception(f"Vector binop {op} not implemented.")
            finally:
                if set_fast:
                    self.set_fast_math(False)
            self.result_stack.append(result)
            return

        # Scalar Fallback: Original scalar promotion logic
        llvm_lhs, llvm_rhs = self.promote_operands(llvm_lhs, llvm_rhs)

        if node.op_code in ("&&", "and"):
            result = self._llvm.ir_builder.and_(llvm_lhs, llvm_rhs, "andtmp")
            self.result_stack.append(result)
            return
        elif node.op_code in ("||", "or"):
            result = self._llvm.ir_builder.or_(llvm_lhs, llvm_rhs, "ortmp")
            self.result_stack.append(result)
            return

        if node.op_code == "+":
            # note: it should be according the datatype,
            #       e.g. for float it should be fadd
            if (
                isinstance(llvm_lhs.type, ir.PointerType)
                and isinstance(llvm_rhs.type, ir.PointerType)
                and llvm_lhs.type.pointee == self._llvm.INT8_TYPE
                and llvm_rhs.type.pointee == self._llvm.INT8_TYPE
            ):
                result = self._handle_string_concatenation(llvm_lhs, llvm_rhs)
                self.result_stack.append(result)
                return

            elif is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                result = self._llvm.ir_builder.fadd(
                    llvm_lhs, llvm_rhs, "addtmp"
                )
                self._apply_fast_math(result)
            else:
                # there's more conditions to be handled
                result = self._llvm.ir_builder.add(
                    llvm_lhs, llvm_rhs, "addtmp"
                )
            self.result_stack.append(result)
            return
        elif node.op_code == "-":
            # note: it should be according the datatype,
            if is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                result = self._llvm.ir_builder.fsub(
                    llvm_lhs, llvm_rhs, "subtmp"
                )
                self._apply_fast_math(result)
            else:
                # note: be careful you should handle this as  INT32
                result = self._llvm.ir_builder.sub(
                    llvm_lhs, llvm_rhs, "subtmp"
                )
            self.result_stack.append(result)
            return
        elif node.op_code == "*":
            # note: it should be according the datatype,
            #       e.g. for float it should be fmul
            if is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                result = self._llvm.ir_builder.fmul(
                    llvm_lhs, llvm_rhs, "multmp"
                )
                self._apply_fast_math(result)
            else:
                # note: be careful you should handle this as INT32
                result = self._llvm.ir_builder.mul(
                    llvm_lhs, llvm_rhs, "multmp"
                )
            self.result_stack.append(result)
            return
        elif node.op_code == "<":
            # note: it should be according the datatype,
            #       e.g. for float it should be fcmp
            if is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                result = self._llvm.ir_builder.fcmp_ordered(
                    "<", llvm_lhs, llvm_rhs, "lttmp"
                )
            else:
                # handle it depend on datatype
                result = self._llvm.ir_builder.icmp_signed(
                    "<", llvm_lhs, llvm_rhs, "lttmp"
                )
            self.result_stack.append(result)
            return
        elif node.op_code == ">":
            # note: it should be according the datatype,
            #       e.g. for float it should be fcmp
            if is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                result = self._llvm.ir_builder.fcmp_ordered(
                    ">", llvm_lhs, llvm_rhs, "gttmp"
                )
            else:
                # be careful we havn't  handled all the conditions
                result = self._llvm.ir_builder.icmp_signed(
                    ">", llvm_lhs, llvm_rhs, "gttmp"
                )
            self.result_stack.append(result)
            return
        elif node.op_code == "<=":
            if is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                result = self._llvm.ir_builder.fcmp_ordered(
                    "<=", llvm_lhs, llvm_rhs, "letmp"
                )
            else:
                result = self._llvm.ir_builder.icmp_signed(
                    "<=", llvm_lhs, llvm_rhs, "letmp"
                )
            self.result_stack.append(result)
            return
        elif node.op_code == ">=":
            if is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                result = self._llvm.ir_builder.fcmp_ordered(
                    ">=", llvm_lhs, llvm_rhs, "getmp"
                )
            else:
                result = self._llvm.ir_builder.icmp_signed(
                    ">=", llvm_lhs, llvm_rhs, "getmp"
                )
            self.result_stack.append(result)
            return
        elif node.op_code == "/":
            # Check the datatype to decide between floating-point and integer
            # division
            if is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                # Floating-point division
                result = self._llvm.ir_builder.fdiv(
                    llvm_lhs, llvm_rhs, "divtmp"
                )
                self._apply_fast_math(result)
            else:
                # Assuming the division is signed by default. Use `udiv` for
                # unsigned division.
                result = self._llvm.ir_builder.sdiv(
                    llvm_lhs, llvm_rhs, "divtmp"
                )
            self.result_stack.append(result)
            return

        elif node.op_code == "==":
            # Handle string comparison for equality
            if (
                isinstance(llvm_lhs.type, ir.PointerType)
                and isinstance(llvm_rhs.type, ir.PointerType)
                and llvm_lhs.type.pointee == self._llvm.INT8_TYPE
                and llvm_rhs.type.pointee == self._llvm.INT8_TYPE
            ):
                # String comparison
                cmp_result = self._handle_string_comparison(
                    llvm_lhs, llvm_rhs, "=="
                )
            elif is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                cmp_result = self._llvm.ir_builder.fcmp_ordered(
                    "==", llvm_lhs, llvm_rhs, "eqtmp"
                )
            else:
                cmp_result = self._llvm.ir_builder.icmp_signed(
                    "==", llvm_lhs, llvm_rhs, "eqtmp"
                )
            self.result_stack.append(cmp_result)
            return

        elif node.op_code == "!=":
            # Handle string comparison for inequality
            if (
                isinstance(llvm_lhs.type, ir.PointerType)
                and isinstance(llvm_rhs.type, ir.PointerType)
                and llvm_lhs.type.pointee == self._llvm.INT8_TYPE
                and llvm_rhs.type.pointee == self._llvm.INT8_TYPE
            ):
                # String comparison
                cmp_result = self._handle_string_comparison(
                    llvm_lhs, llvm_rhs, "!="
                )
            elif is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                cmp_result = self._llvm.ir_builder.fcmp_ordered(
                    "!=", llvm_lhs, llvm_rhs, "netmp"
                )
            else:
                cmp_result = self._llvm.ir_builder.icmp_signed(
                    "!=", llvm_lhs, llvm_rhs, "netmp"
                )
            self.result_stack.append(cmp_result)
            return

        raise Exception(f"Binary op {node.op_code} not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, block: astx.Block) -> None:
        """Translate ASTx Block to LLVM-IR."""
        result = None
        for node in block.nodes:
            self.visit(node)
            try:
                result = self.result_stack.pop()
            except IndexError:
                # some nodes doesn't add anything in the stack
                pass
        if result is not None:
            self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.IfStmt) -> None:
        """Translate IF statement."""
        self.visit(node.condition)
        cond_v = self.result_stack.pop()
        if not cond_v:
            raise Exception("codegen: Invalid condition expression.")

        if isinstance(cond_v.type, (ir.FloatType, ir.DoubleType)):
            cmp_instruction = self._llvm.ir_builder.fcmp_ordered
            zero_val = ir.Constant(cond_v.type, 0.0)
        else:
            cmp_instruction = self._llvm.ir_builder.icmp_signed
            zero_val = ir.Constant(cond_v.type, 0)

        cond_v = cmp_instruction(
            "!=",
            cond_v,
            zero_val,
        )

        # Create blocks for the then and else cases.
        then_bb = self._llvm.ir_builder.function.append_basic_block(
            "bb_if_then"
        )
        else_bb = self._llvm.ir_builder.function.append_basic_block(
            "bb_if_else"
        )
        merge_bb = self._llvm.ir_builder.function.append_basic_block(
            "bb_if_end"
        )

        self._llvm.ir_builder.cbranch(cond_v, then_bb, else_bb)

        # Emit then value.
        self._llvm.ir_builder.position_at_start(then_bb)
        self.visit(node.then)
        then_v = self.result_stack.pop()
        if not then_v:
            raise Exception("codegen: `Then` expression is invalid.")

        self._llvm.ir_builder.branch(merge_bb)

        # Update reference to final block of 'then'
        then_bb = self._llvm.ir_builder.block

        # Emit else block.
        self._llvm.ir_builder.position_at_start(else_bb)
        else_v = None
        if node.else_ is not None:
            self.visit(node.else_)
            else_v = self.result_stack.pop()
        else:
            else_v = ir.Constant(self._llvm.INT32_TYPE, 0)

        # Update reference to final block of 'else'
        else_bb = self._llvm.ir_builder.block
        self._llvm.ir_builder.branch(merge_bb)

        # Emit merge block and PHI node
        self._llvm.ir_builder.position_at_start(merge_bb)
        phi = self._llvm.ir_builder.phi(self._llvm.INT32_TYPE, "iftmp")
        phi.add_incoming(then_v, then_bb)
        phi.add_incoming(else_v, else_bb)

        self.result_stack.append(phi)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.WhileStmt) -> None:
        """Translate ASTx While Loop to LLVM-IR."""
        # Create blocks for the condition check, the loop body,
        # and the block after the loop.
        cond_bb = self._llvm.ir_builder.function.append_basic_block(
            "whilecond"
        )
        body_bb = self._llvm.ir_builder.function.append_basic_block(
            "whilebody"
        )
        after_bb = self._llvm.ir_builder.function.append_basic_block(
            "afterwhile"
        )

        # Branch to the condition check block.
        self._llvm.ir_builder.branch(cond_bb)

        # Start inserting into the condition check block.
        self._llvm.ir_builder.position_at_start(cond_bb)

        # Emit the condition.
        self.visit(expr.condition)
        cond_val = self.result_stack.pop()
        if not cond_val:
            raise Exception("codegen: Invalid condition expression.")

        # Convert condition to a bool by comparing non-equal to 0.
        if isinstance(cond_val.type, (ir.FloatType, ir.DoubleType)):
            cmp_instruction = self._llvm.ir_builder.fcmp_ordered
            zero_val = ir.Constant(cond_val.type, 0.0)
        else:
            cmp_instruction = self._llvm.ir_builder.icmp_signed
            zero_val = ir.Constant(cond_val.type, 0)

        cond_val = cmp_instruction(
            "!=",
            cond_val,
            zero_val,
            "whilecond",
        )

        # Conditional branch based on the condition.
        self._llvm.ir_builder.cbranch(cond_val, body_bb, after_bb)

        # Start inserting into the loop body block.
        self._llvm.ir_builder.position_at_start(body_bb)

        # Emit the body of the loop.
        self.visit(expr.body)
        body_val = self.result_stack.pop()

        if not body_val:
            return

        # Branch back to the condition check.
        self._llvm.ir_builder.branch(cond_bb)

        # Start inserting into the block after the loop.
        self._llvm.ir_builder.position_at_start(after_bb)

        # While loop always returns 0.
        result = ir.Constant(self._llvm.INT32_TYPE, 0)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.VariableAssignment) -> None:
        """Translate variable assignment expression."""
        # Get the name of the variable to assign to
        var_name = expr.name

        # Codegen the value expression on the right-hand side
        self.visit(expr.value)
        llvm_value = safe_pop(self.result_stack)

        if not llvm_value:
            raise Exception("codegen: Invalid value in VariableAssignment.")

        # Look up the variable in the named values
        llvm_var = self.named_values.get(var_name)

        if not llvm_var:
            raise Exception(
                f"Identifier '{var_name}' not found in the named values."
            )

        # Store the value in the variable
        self._llvm.ir_builder.store(llvm_value, llvm_var)

        # Optionally, you can push the result onto the result stack if needed
        self.result_stack.append(llvm_value)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.ForCountLoopStmt) -> None:
        """Translate ASTx For Range Loop to LLVM-IR."""
        saved_block = self._llvm.ir_builder.block
        var_addr = self.create_entry_block_alloca(
            "for_count_loop", node.initializer.type_.__class__.__name__.lower()
        )
        self._llvm.ir_builder.position_at_end(saved_block)

        # Emit the start code first, without 'variable' in scope.
        self.visit(node.initializer)
        initializer_val = self.result_stack.pop()
        if not initializer_val:
            raise Exception("codegen: Invalid start argument.")

        # Store the value into the alloca.
        self._llvm.ir_builder.store(initializer_val, var_addr)

        loop_header_bb = self._llvm.ir_builder.function.append_basic_block(
            "loop.header"
        )
        self._llvm.ir_builder.branch(loop_header_bb)

        # Start insertion in loop header
        self._llvm.ir_builder.position_at_start(loop_header_bb)

        # Save old value if variable shadows an existing one
        old_val = self.named_values.get(node.initializer.name)
        self.named_values[node.initializer.name] = var_addr

        # Emit condition check (e.g., i < 10)
        self.visit(node.condition)
        cond_val = self.result_stack.pop()

        # Create blocks for loop body and after loop
        loop_body_bb = self._llvm.ir_builder.function.append_basic_block(
            "loop.body"
        )
        after_loop_bb = self._llvm.ir_builder.function.append_basic_block(
            "after.loop"
        )

        # Branch based on condition
        self._llvm.ir_builder.cbranch(cond_val, loop_body_bb, after_loop_bb)

        # Emit loop body
        self._llvm.ir_builder.position_at_start(loop_body_bb)
        self.visit(node.body)
        _body_val = self.result_stack.pop()

        # Emit update expression
        self.visit(node.update)
        update_val = self.result_stack.pop()

        # Store updated value
        self._llvm.ir_builder.store(update_val, var_addr)

        # Branch back to loop header
        self._llvm.ir_builder.branch(loop_header_bb)

        # Move to after-loop block
        self._llvm.ir_builder.position_at_start(after_loop_bb)

        # Restore the unshadowed variable.
        if old_val:
            self.named_values[node.initializer.name] = old_val
        else:
            self.named_values.pop(node.initializer.name, None)

        result = ir.Constant(
            self._llvm.get_data_type(
                node.initializer.type_.__class__.__name__.lower()
            ),
            name=name,
        )

        entry = fn.append_basic_block("entry")
        loop = fn.append_basic_block("loop")
        eq = fn.append_basic_block("eq")
        ne = fn.append_basic_block("ne")

        b = ir.IRBuilder(entry)
        idx = b.alloca(self._llvm.INT32_TYPE)
        b.store(ir.Constant(self._llvm.INT32_TYPE, 0), idx)
        b.branch(loop)

        b.position_at_start(loop)
        i = b.load(idx)
        c1 = b.load(b.gep(fn.args[0], [i], inbounds=True))
        c2 = b.load(b.gep(fn.args[1], [i], inbounds=True))
        same = b.icmp_signed("==", c1, c2)
        is_null = b.icmp_signed("==", c1, ir.Constant(self._llvm.INT8_TYPE, 0))
        b.cbranch(b.and_(same, is_null), eq, ne)

        b.position_at_start(ne)
        b.ret(ir.Constant(self._llvm.BOOLEAN_TYPE, 0))

        b.position_at_start(eq)
        b.ret(ir.Constant(self._llvm.BOOLEAN_TYPE, 1))

        return fn

    # ------------------------------------------------------------
    # STRING LITERALS
    # ------------------------------------------------------------

    @dispatch
    def visit(self, expr: astx.LiteralUTF8String) -> None:
        data = expr.value.encode("utf-8") + b"\0"
        arr_ty = ir.ArrayType(self._llvm.INT8_TYPE, len(data))
        gv = ir.GlobalVariable(
            self._llvm.module,
            arr_ty,
            name=f"str_{abs(hash(expr.value))}",
        )
        gv.linkage = "internal"
        gv.global_constant = True
        gv.initializer = ir.Constant(arr_ty, data)

        ptr = self._llvm.ir_builder.gep(
            gv,
            [ir.Constant(self._llvm.INT32_TYPE, 0),
             ir.Constant(self._llvm.INT32_TYPE, 0)],
            inbounds=True,
        )
        self.result_stack.append(ptr)

    @dispatch
    def visit(self, expr: astx.LiteralString) -> None:
        self.visit(astx.LiteralUTF8String(value=expr.value))


@public
class LLVMLiteIR(Builder):
    """LLVM-IR transpiler and compiler."""

    def __init__(self) -> None:
        super().__init__()
        self.translator = LLVMLiteIRVisitor()

    def build(self, node: astx.AST, output_file: str) -> None:
        self.translator = LLVMLiteIRVisitor()
        ir_text = self.translator.translate(node)

        mod = llvm.parse_assembly(ir_text)
        obj = self.translator.target_machine.emit_object(mod)

        with tempfile.NamedTemporaryFile(delete=True) as f:
            obj_path = f.name + ".o"

        with open(obj_path, "wb") as f:
            f.write(obj)

        xh.clang(obj_path, "-o", output_file)
        os.chmod(output_file, 0o755)

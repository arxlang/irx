"""
title: LLVM-IR builder.
"""

from __future__ import annotations

import ctypes
import os
import tempfile

from datetime import datetime
from datetime import time as _time
from pathlib import Path
from typing import Any, Optional, cast

import astx

from llvmlite import binding as llvm
from llvmlite import ir
from llvmlite.ir import DoubleType, FloatType, HalfType, VectorType

try:  # FP128 may not exist depending on llvmlite build
    from llvmlite.ir import FP128Type
except ImportError:  # pragma: no cover - optional
    FP128Type = None

from plum import dispatch
from public import public

from irx import arrow as irx_arrow
from irx import system
from irx.builders.base import Builder, BuilderVisitor
from irx.runtime.linking import link_executable
from irx.runtime.registry import (
    RuntimeFeatureState,
    get_default_runtime_feature_registry,
)
from irx.tools.typing import typechecked


def is_fp_type(t: "ir.Type") -> bool:
    """
    title: Return True if t is any floating-point LLVM type.
    parameters:
      t:
        type: ir.Type
    returns:
      type: bool
    """
    fp_types = [HalfType, FloatType, DoubleType]
    if FP128Type is not None:
        fp_types.append(FP128Type)
    return isinstance(t, tuple(fp_types))


def is_int_type(t: "ir.Type") -> bool:
    """
    title: Return True if t is any scalar integer LLVM type.
    parameters:
      t:
        type: ir.Type
    returns:
      type: bool
    """
    return isinstance(t, ir.IntType)


def is_vector(v: "ir.Value") -> bool:
    """
    title: Return True if v is an LLVM vector value.
    parameters:
      v:
        type: ir.Value
    returns:
      type: bool
    """
    return isinstance(getattr(v, "type", None), VectorType)


def _is_unsigned_node(node: "astx.AST") -> bool:
    """
    title: Return True if the AST node carries an unsigned integer type.
    parameters:
      node:
        type: astx.AST
    returns:
      type: bool
    """
    type_ = getattr(node, "type_", None)
    return isinstance(type_, astx.UnsignedInteger)


def emit_int_div(
    ir_builder: "ir.IRBuilder",
    lhs: "ir.Value",
    rhs: "ir.Value",
    unsigned: bool,
) -> "ir.Instruction":
    """
    title: Emit signed or unsigned vector integer division.
    parameters:
      ir_builder:
        type: ir.IRBuilder
      lhs:
        type: ir.Value
      rhs:
        type: ir.Value
      unsigned:
        type: bool
    returns:
      type: ir.Instruction
    """
    return (
        ir_builder.udiv(lhs, rhs, name="vdivtmp")
        if unsigned
        else ir_builder.sdiv(lhs, rhs, name="vdivtmp")
    )


def emit_add(
    ir_builder: "ir.IRBuilder",
    lhs: "ir.Value",
    rhs: "ir.Value",
    name: str = "addtmp",
) -> "ir.Instruction":
    """
    title: Emit float or integer addition based on operand type.
    parameters:
      ir_builder:
        type: ir.IRBuilder
      lhs:
        type: ir.Value
      rhs:
        type: ir.Value
      name:
        type: str
    returns:
      type: ir.Instruction
    """
    if is_fp_type(lhs.type):
        return ir_builder.fadd(lhs, rhs, name=name)
    return ir_builder.add(lhs, rhs, name=name)


def splat_scalar(
    ir_builder: "ir.IRBuilder", scalar: "ir.Value", vec_type: "ir.VectorType"
) -> "ir.Value":
    """
    title: Broadcast a scalar to all lanes of a vector.
    parameters:
      ir_builder:
        type: ir.IRBuilder
      scalar:
        type: ir.Value
      vec_type:
        type: ir.VectorType
    returns:
      type: ir.Value
    """
    zero_i32 = ir.Constant(ir.IntType(32), 0)
    undef_vec = ir.Constant(vec_type, ir.Undefined)
    v0 = ir_builder.insert_element(undef_vec, scalar, zero_i32)
    mask_ty = ir.VectorType(ir.IntType(32), vec_type.count)
    mask = ir.Constant(mask_ty, [0] * vec_type.count)
    return ir_builder.shuffle_vector(v0, undef_vec, mask)


@typechecked
def safe_pop(
    lst: list[ir.Value | ir.Function],
) -> Optional[ir.Value | ir.Function]:
    """
    title: Implement a safe pop operation for lists.
    parameters:
      lst:
        type: list[ir.Value | ir.Function]
    returns:
      type: Optional[ir.Value | ir.Function]
    """
    try:
        return lst.pop()
    except IndexError:
        return None


@typechecked
class VariablesLLVM:
    """
    title: Store all the LLVM variables used for code generation.
    attributes:
      FLOAT_TYPE:
        type: ir.types.Type
      FLOAT16_TYPE:
        type: ir.types.Type
      DOUBLE_TYPE:
        type: ir.types.Type
      INT8_TYPE:
        type: ir.types.Type
      INT64_TYPE:
        type: ir.types.Type
      INT16_TYPE:
        type: ir.types.Type
      INT32_TYPE:
        type: ir.types.Type
      VOID_TYPE:
        type: ir.types.Type
      BOOLEAN_TYPE:
        type: ir.types.Type
      UINT8_TYPE:
        type: ir.types.Type
      UINT16_TYPE:
        type: ir.types.Type
      UINT32_TYPE:
        type: ir.types.Type
      UINT64_TYPE:
        type: ir.types.Type
      UINT128_TYPE:
        type: ir.types.Type
      STRING_TYPE:
        type: ir.types.Type
      ASCII_STRING_TYPE:
        type: ir.types.Type
      UTF8_STRING_TYPE:
        type: ir.types.Type
      TIME_TYPE:
        type: ir.types.Type
      TIMESTAMP_TYPE:
        type: ir.types.Type
      DATETIME_TYPE:
        type: ir.types.Type
      SIZE_T_TYPE:
        type: ir.types.Type
      POINTER_BITS:
        type: int
      OPAQUE_POINTER_TYPE:
        type: ir.types.Type
      ARROW_ARRAY_BUILDER_HANDLE_TYPE:
        type: ir.types.Type
      ARROW_ARRAY_HANDLE_TYPE:
        type: ir.types.Type
      context:
        type: ir.context.Context
      module:
        type: ir.module.Module
      ir_builder:
        type: ir.builder.IRBuilder
    """

    FLOAT_TYPE: ir.types.Type
    FLOAT16_TYPE: ir.types.Type
    DOUBLE_TYPE: ir.types.Type
    INT8_TYPE: ir.types.Type
    INT64_TYPE: ir.types.Type
    INT16_TYPE: ir.types.Type
    INT32_TYPE: ir.types.Type
    VOID_TYPE: ir.types.Type
    BOOLEAN_TYPE: ir.types.Type
    UINT8_TYPE: ir.types.Type
    UINT16_TYPE: ir.types.Type
    UINT32_TYPE: ir.types.Type
    UINT64_TYPE: ir.types.Type
    UINT128_TYPE: ir.types.Type
    STRING_TYPE: ir.types.Type
    ASCII_STRING_TYPE: ir.types.Type
    UTF8_STRING_TYPE: ir.types.Type
    TIME_TYPE: ir.types.Type
    TIMESTAMP_TYPE: ir.types.Type
    DATETIME_TYPE: ir.types.Type
    SIZE_T_TYPE: ir.types.Type
    POINTER_BITS: int
    OPAQUE_POINTER_TYPE: ir.types.Type
    ARROW_ARRAY_BUILDER_HANDLE_TYPE: ir.types.Type
    ARROW_ARRAY_HANDLE_TYPE: ir.types.Type

    context: ir.context.Context
    module: ir.module.Module

    ir_builder: ir.builder.IRBuilder

    def get_data_type(self, type_name: str) -> ir.types.Type:
        """
        title: Get the LLVM data type for the given type name.
        parameters:
          type_name:
            type: str
            description: The name of the type.
        returns:
          type: ir.types.Type
          description: The LLVM data type.
        """
        if type_name == "float32":
            return self.FLOAT_TYPE
        elif type_name == "float16":
            return self.FLOAT16_TYPE
        elif type_name in ("double", "float64"):
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
            return self.ASCII_STRING_TYPE
        elif type_name == "utf8string":
            return self.UTF8_STRING_TYPE
        elif type_name == "uint8":
            return self.UINT8_TYPE
        elif type_name == "uint16":
            return self.UINT16_TYPE
        elif type_name == "uint32":
            return self.UINT32_TYPE
        elif type_name == "uint64":
            return self.UINT64_TYPE
        elif type_name == "uint128":
            return self.UINT128_TYPE
        elif type_name == "nonetype":
            return self.VOID_TYPE

        raise Exception(f"[EE]: Type name {type_name} not valid.")


@typechecked
class LLVMLiteIRVisitor(BuilderVisitor):
    """
    title: LLVM-IR Translator.
    attributes:
      named_values:
        type: dict[str, Any]
      _llvm:
        type: VariablesLLVM
      function_protos:
        type: dict[str, astx.FunctionPrototype]
      result_stack:
        type: list[ir.Value | ir.Function]
      runtime_features:
        type: RuntimeFeatureState
      const_vars:
        type: set[str]
      _fast_math_enabled:
        type: bool
      target:
        type: llvm.TargetRef
      target_machine:
        type: llvm.TargetMachine
    """

    # AllocaInst
    named_values: dict[str, Any] = {}

    _llvm: VariablesLLVM

    function_protos: dict[str, astx.FunctionPrototype]
    result_stack: list[ir.Value | ir.Function] = []
    runtime_features: RuntimeFeatureState

    def __init__(
        self,
        active_runtime_features: Optional[set[str]] = None,
    ) -> None:
        """
        title: Initialize LLVMTranslator object.
        parameters:
          active_runtime_features:
            type: Optional[set[str]]
        """
        super().__init__()

        # named_values as instance variable so it isn't shared across instances
        self.named_values: dict[str, Any] = {}
        self.const_vars: set[str] = set()
        self.function_protos: dict[str, astx.FunctionPrototype] = {}
        self.result_stack: list[ir.Value | ir.Function] = []
        self._fast_math_enabled: bool = False

        self.initialize()

        self.target: llvm.TargetRef = llvm.Target.from_default_triple()
        try:
            self.target_machine: llvm.TargetMachine = (
                self.target.create_target_machine(
                    codemodel="small",
                    reloc="pic",
                )
            )
        except TypeError:
            # Older llvmlite versions may not expose reloc in Python bindings.
            self.target_machine = self.target.create_target_machine(
                codemodel="small"
            )

        self._llvm.module.triple = self.target_machine.triple
        self._llvm.module.data_layout = str(self.target_machine.target_data)

        if self._llvm.SIZE_T_TYPE is None:
            self._llvm.SIZE_T_TYPE = self._get_size_t_type_from_triple()

        self._add_builtins()
        self.runtime_features = RuntimeFeatureState(
            owner=self,
            registry=get_default_runtime_feature_registry(),
            active_features=active_runtime_features,
        )

    def translate(self, node: astx.AST) -> str:
        """
        title: Translate an ASTx expression to string.
        parameters:
          node:
            type: astx.AST
        returns:
          type: str
        """
        self.visit(node)
        return str(self._llvm.module)

    def activate_runtime_feature(self, feature_name: str) -> None:
        """
        title: Activate a runtime feature for the current module.
        parameters:
          feature_name:
            type: str
        """
        self.runtime_features.activate(feature_name)

    def require_runtime_symbol(
        self, feature_name: str, symbol_name: str
    ) -> ir.Function:
        """
        title: Declare an external symbol owned by a runtime feature.
        parameters:
          feature_name:
            type: str
          symbol_name:
            type: str
        returns:
          type: ir.Function
        """
        return self.runtime_features.require_symbol(feature_name, symbol_name)

    def _init_native_size_types(self) -> None:
        """
        title: Initialize pointer/size_t types from host.
        """
        self._llvm.POINTER_BITS = ctypes.sizeof(ctypes.c_void_p) * 8
        self._llvm.SIZE_T_TYPE = None

    def _get_size_t_type_from_triple(self) -> ir.IntType:
        """
        title: Determine size_t type from target triple using LLVM API.
        returns:
          type: ir.IntType
        """
        triple = self.target_machine.triple.lower()

        if any(
            arch in triple
            for arch in [
                "x86_64",
                "amd64",
                "aarch64",
                "arm64",
                "ppc64",
                "mips64",
            ]
        ):
            return ir.IntType(64)
        elif any(arch in triple for arch in ["i386", "i686", "arm", "mips"]):
            if "64" in triple:
                return ir.IntType(64)
            return ir.IntType(32)

        return ir.IntType(ctypes.sizeof(ctypes.c_size_t) * 8)

    def initialize(self) -> None:
        """
        title: Initialize self.
        """
        self._llvm = VariablesLLVM()
        self._llvm.module = ir.module.Module("Arx")
        # Initialize native-sized types (size_t, pointer width)
        self._init_native_size_types()

        # Initialize LLVM targets
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        llvm.initialize_native_target()
        llvm.initialize_native_asmparser()
        llvm.initialize_native_asmprinter()

        # Create a new builder for the module.
        self._llvm.ir_builder = ir.IRBuilder()

        # Data Types
        self._llvm.FLOAT_TYPE = ir.FloatType()
        self._llvm.FLOAT16_TYPE = ir.HalfType()
        self._llvm.DOUBLE_TYPE = ir.DoubleType()
        self._llvm.BOOLEAN_TYPE = ir.IntType(1)
        self._llvm.INT8_TYPE = ir.IntType(8)
        self._llvm.INT16_TYPE = ir.IntType(16)
        self._llvm.INT32_TYPE = ir.IntType(32)
        self._llvm.INT64_TYPE = ir.IntType(64)
        self._llvm.UINT8_TYPE = ir.IntType(8)
        self._llvm.UINT16_TYPE = ir.IntType(16)
        self._llvm.UINT32_TYPE = ir.IntType(32)
        self._llvm.UINT64_TYPE = ir.IntType(64)
        self._llvm.UINT128_TYPE = ir.IntType(128)
        self._llvm.VOID_TYPE = ir.VoidType()
        self._llvm.ASCII_STRING_TYPE = ir.IntType(8).as_pointer()
        self._llvm.UTF8_STRING_TYPE = self._llvm.ASCII_STRING_TYPE
        self._llvm.OPAQUE_POINTER_TYPE = self._llvm.INT8_TYPE.as_pointer()
        self._llvm.ARROW_ARRAY_BUILDER_HANDLE_TYPE = (
            self._llvm.OPAQUE_POINTER_TYPE
        )
        self._llvm.ARROW_ARRAY_HANDLE_TYPE = self._llvm.OPAQUE_POINTER_TYPE
        # Composite types
        self._llvm.TIME_TYPE = ir.LiteralStructType(
            [
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
            ]
        )
        self._llvm.TIMESTAMP_TYPE = ir.LiteralStructType(
            [
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
            ]
        )
        self._llvm.DATETIME_TYPE = ir.LiteralStructType(
            [
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
            ]
        )
        # SIZE_T_TYPE already initialized based on host; do not override with a
        # fixed width here to avoid mismatches on non-64-bit targets.

    def _add_builtins(self) -> None:
        # The C++ tutorial adds putchard() simply by defining it in the host
        # C++ code, which is then accessible to the JIT. It doesn't work as
        # simply for us; but luckily it's very easy to define new "C level"
        # functions for our JITed code to use - just emit them as LLVM IR.
        # This is what this method does.

        # Add the declaration of putchar
        putchar_ty = ir.FunctionType(
            self._llvm.INT32_TYPE, [self._llvm.INT32_TYPE]
        )
        putchar = ir.Function(self._llvm.module, putchar_ty, "putchar")

        # Add putchard
        putchard_ty = ir.FunctionType(
            self._llvm.INT32_TYPE, [self._llvm.INT32_TYPE]
        )
        putchard = ir.Function(self._llvm.module, putchard_ty, "putchard")

        ir_builder = ir.IRBuilder(putchard.append_basic_block("entry"))

        ival = ir_builder.fptoui(
            putchard.args[0], self._llvm.INT32_TYPE, "intcast"
        )

        ir_builder.call(putchar, [ival])
        ir_builder.ret(ir.Constant(self._llvm.INT32_TYPE, 0))

    def get_function(self, name: str) -> Optional[ir.Function]:
        """
        title: Put the function defined by the given name to result stack.
        parameters:
          name:
            type: str
            description: Function name.
        returns:
          type: Optional[ir.Function]
        """
        if name in self._llvm.module.globals:
            return self._llvm.module.get_global(name)

        if name in self.function_protos:
            self.visit(self.function_protos[name])
            return cast(ir.Function, self.result_stack.pop())

        return None

    def create_entry_block_alloca(
        self, var_name: str, type_name: str
    ) -> Any:  # llvm.AllocaInst
        """
        title: Create an alloca instruction in the entry block of the function.
        summary: This is used for mutable variables, etc.
        parameters:
          var_name:
            type: str
            description: The variable name.
          type_name:
            type: str
            description: The type name.
        returns:
          type: Any
          description: An llvm allocation instance.
        """
        current_block = self._llvm.ir_builder.block
        self._llvm.ir_builder.position_at_start(
            self._llvm.ir_builder.function.entry_basic_block
        )
        alloca = self._llvm.ir_builder.alloca(
            self._llvm.get_data_type(type_name), None, var_name
        )
        if current_block is not None:
            self._llvm.ir_builder.position_at_end(current_block)
        return alloca

    def fp_rank(self, t: ir.Type) -> int:
        """
        title: Rank floating-point types half, float, double.
        parameters:
          t:
            type: ir.Type
        returns:
          type: int
        """
        if isinstance(t, ir.HalfType):
            return 1
        if isinstance(t, ir.FloatType):
            return 2
        if isinstance(t, ir.DoubleType):
            return 3
        return 0

    def promote_operands(
        self, lhs: ir.Value, rhs: ir.Value, unsigned: bool = False
    ) -> tuple[ir.Value, ir.Value]:
        """
        title: Promote two LLVM IR numeric operands to a common type.
        parameters:
          lhs:
            type: ir.Value
            description: The left-hand operand.
          rhs:
            type: ir.Value
            description: The right-hand operand.
          unsigned:
            type: bool
        returns:
          type: tuple[ir.Value, ir.Value]
          description: A tuple containing the promoted operands.
        """
        if lhs.type == rhs.type:
            return lhs, rhs

        # perform sign/zero extension (for integer operands)
        if is_int_type(lhs.type) and is_int_type(rhs.type):
            if unsigned:
                if lhs.type.width < rhs.type.width:
                    lhs = self._llvm.ir_builder.zext(
                        lhs, rhs.type, "promote_lhs"
                    )
                elif lhs.type.width > rhs.type.width:
                    rhs = self._llvm.ir_builder.zext(
                        rhs, lhs.type, "promote_rhs"
                    )
            elif lhs.type.width < rhs.type.width:
                lhs = self._llvm.ir_builder.sext(lhs, rhs.type, "promote_lhs")
            elif lhs.type.width > rhs.type.width:
                rhs = self._llvm.ir_builder.sext(rhs, lhs.type, "promote_rhs")
            return lhs, rhs

        lhs_fp_rank = self.fp_rank(lhs.type)
        rhs_fp_rank = self.fp_rank(rhs.type)

        if lhs_fp_rank > 0 and rhs_fp_rank > 0:
            # make both the wider FP
            if lhs_fp_rank < rhs_fp_rank:
                lhs = self._llvm.ir_builder.fpext(lhs, rhs.type, "promote_lhs")
            elif lhs_fp_rank > rhs_fp_rank:
                rhs = self._llvm.ir_builder.fpext(rhs, lhs.type, "promote_rhs")
            return lhs, rhs

        # If one is int and other is FP, convert int -> FP
        if is_int_type(lhs.type) and rhs_fp_rank > 0:
            target_fp = rhs.type
            if unsigned:
                lhs_fp = self._llvm.ir_builder.uitofp(
                    lhs, target_fp, "uint_to_fp"
                )
            else:
                lhs_fp = self._llvm.ir_builder.sitofp(
                    lhs, target_fp, "int_to_fp"
                )
            return lhs_fp, rhs

        if is_int_type(rhs.type) and lhs_fp_rank > 0:
            target_fp = lhs.type
            if unsigned:
                rhs_fp = self._llvm.ir_builder.uitofp(
                    rhs, target_fp, "uint_to_fp"
                )
            else:
                rhs_fp = self._llvm.ir_builder.sitofp(
                    rhs, target_fp, "int_to_fp"
                )
            return lhs, rhs_fp

        return lhs, rhs

    def _get_fma_function(self, ty: ir.Type) -> ir.Function:
        """
        title: Return (and cache) the llvm.fma.* intrinsic for a type.
        parameters:
          ty:
            type: ir.Type
        returns:
          type: ir.Function
        """
        if isinstance(ty, ir.VectorType):
            elem_ty = ty.element
            count = ty.count
        else:
            elem_ty = ty
            count = None

        if isinstance(elem_ty, FloatType):
            suffix = "f32"
        elif isinstance(elem_ty, DoubleType):
            suffix = "f64"
        elif isinstance(elem_ty, HalfType):
            suffix = "f16"
        else:
            raise Exception("FMA supports only floating-point operands")

        if count is not None:
            suffix = f"v{count}{suffix}"

        name = f"llvm.fma.{suffix}"
        if name in self._llvm.module.globals:
            return self._llvm.module.get_global(name)

        fn_ty = ir.FunctionType(ty, [ty, ty, ty])
        fn = ir.Function(self._llvm.module, fn_ty, name)
        fn.linkage = "external"
        return fn

    def _emit_fma(
        self, lhs: ir.Value, rhs: ir.Value, addend: ir.Value
    ) -> ir.Value:
        """
        title: Emit a fused multiply-add, using intrinsic fallback if needed.
        parameters:
          lhs:
            type: ir.Value
          rhs:
            type: ir.Value
          addend:
            type: ir.Value
        returns:
          type: ir.Value
        """
        builder = self._llvm.ir_builder
        if hasattr(builder, "fma"):
            inst = builder.fma(lhs, rhs, addend, name="vfma")
            self._apply_fast_math(inst)
            return inst

        fma_fn = self._get_fma_function(lhs.type)
        inst = builder.call(fma_fn, [lhs, rhs, addend], name="vfma")
        self._apply_fast_math(inst)
        return inst

    def set_fast_math(self, enabled: bool) -> None:
        """
        title: Enable/disable fast-math flags for subsequent FP instructions.
        parameters:
          enabled:
            type: bool
        """
        self._fast_math_enabled = enabled

    def _apply_fast_math(self, inst: ir.Instruction) -> None:
        """
        title: Attach fast-math flags when enabled and applicable.
        parameters:
          inst:
            type: ir.Instruction
        """
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
        """
        title: Translate an ASTx expression.
        parameters:
          node:
            type: astx.AST
        """
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.UnaryOp) -> None:
        """
        title: Translate an ASTx UnaryOp expression.
        parameters:
          node:
            type: astx.UnaryOp
        """
        if node.op_code == "++":
            self.visit(node.operand)
            operand_val = safe_pop(self.result_stack)

            one = ir.Constant(operand_val.type, 1)

            # Perform the increment operation
            result = self._llvm.ir_builder.add(operand_val, one, "inctmp")

            # If operand is a variable, store the new value back
            if isinstance(node.operand, astx.Identifier):
                if node.operand.name in self.const_vars:
                    raise Exception(
                        f"Cannot mutate '{node.operand.name}':"
                        "declared as constant"
                    )
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
                if node.operand.name in self.const_vars:
                    raise Exception(
                        f"Cannot mutate '{node.operand.name}':"
                        "declared as constant"
                    )
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
                if node.operand.name in self.const_vars:
                    raise Exception(
                        f"Cannot mutate '{node.operand.name}':"
                        "declared as constant"
                    )
                addr = self.named_values.get(node.operand.name)
                if addr:
                    self._llvm.ir_builder.store(result, addr)

            self.result_stack.append(result)
            return

        raise Exception(f"Unary operator {node.op_code} not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.BinaryOp) -> None:
        """
        title: Translate binary operation expression.
        parameters:
          node:
            type: astx.BinaryOp
        """
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

            lhs_name = var_lhs.get_name()
            if lhs_name in self.const_vars:
                raise Exception(
                    f"Cannot assign to '{lhs_name}': declared as constant"
                )
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
                _prev_fast_math = self._fast_math_enabled
                if set_fast:
                    self.set_fast_math(True)
                try:
                    result = self._emit_fma(llvm_lhs, llvm_rhs, llvm_fma_rhs)
                finally:
                    self.set_fast_math(_prev_fast_math)
                self.result_stack.append(result)
                return
            _prev_fast_math = self._fast_math_enabled
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
                        unsigned = getattr(node, "unsigned", False)
                        if unsigned is None:
                            # Fallback to signed division (sdiv) by default
                            unsigned = False

                        result = emit_int_div(
                            self._llvm.ir_builder, llvm_lhs, llvm_rhs, unsigned
                        )
                else:
                    raise Exception(f"Vector binop {op} not implemented.")
            finally:
                self.set_fast_math(_prev_fast_math)
            self.result_stack.append(result)
            return

        # Scalar Fallback: Original scalar promotion logic
        llvm_lhs, llvm_rhs = self.promote_operands(
            llvm_lhs,
            llvm_rhs,
            unsigned=_is_unsigned_node(node),
        )

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

            else:
                result = emit_add(
                    self._llvm.ir_builder, llvm_lhs, llvm_rhs, "addtmp"
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
            # handle it depend on datatype
            elif _is_unsigned_node(node):
                result = self._llvm.ir_builder.icmp_unsigned(
                    "<", llvm_lhs, llvm_rhs, "lttmp"
                )
            else:
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
            # be careful we havn't  handled all the conditions
            elif _is_unsigned_node(node):
                result = self._llvm.ir_builder.icmp_unsigned(
                    ">", llvm_lhs, llvm_rhs, "gttmp"
                )
            else:
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
            elif _is_unsigned_node(node):
                result = self._llvm.ir_builder.icmp_unsigned(
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
            elif _is_unsigned_node(node):
                result = self._llvm.ir_builder.icmp_unsigned(
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
            elif _is_unsigned_node(node):
                result = self._llvm.ir_builder.udiv(
                    llvm_lhs, llvm_rhs, "divtmp"
                )
            else:
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
            elif _is_unsigned_node(node):
                cmp_result = self._llvm.ir_builder.icmp_unsigned(
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
            elif _is_unsigned_node(node):
                cmp_result = self._llvm.ir_builder.icmp_unsigned(
                    "!=", llvm_lhs, llvm_rhs, "netmp"
                )
            else:
                cmp_result = self._llvm.ir_builder.icmp_signed(
                    "!=", llvm_lhs, llvm_rhs, "netmp"
                )
            self.result_stack.append(cmp_result)
            return

        elif node.op_code == "%":
            if is_fp_type(llvm_lhs.type) or is_fp_type(llvm_rhs.type):
                result = self._llvm.ir_builder.frem(
                    llvm_lhs, llvm_rhs, "fremtmp"
                )
            elif _is_unsigned_node(node):
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

    @dispatch  # type: ignore[no-redef]
    def visit(self, block: astx.Block) -> None:
        """
        title: Translate ASTx Block to LLVM-IR.
        parameters:
          block:
            type: astx.Block
        """
        result: Optional[ir.Value | ir.Function] = None
        for node in block.nodes:
            if self._llvm.ir_builder.block.terminator is not None:
                break

            stack_size_before = len(self.result_stack)
            self.visit(node)
            if len(self.result_stack) > stack_size_before:
                result = self.result_stack.pop()

            if self._llvm.ir_builder.block.terminator is not None:
                result = None
                break

        if result is not None:
            self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.IfStmt) -> None:
        """
        title: Translate IF statement.
        parameters:
          node:
            type: astx.IfStmt
        """
        self.visit(node.condition)
        cond_v = safe_pop(self.result_stack)
        if not cond_v:
            raise Exception("codegen: Invalid condition expression.")

        if is_fp_type(cond_v.type):
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

        # Emit then branch.
        self._llvm.ir_builder.position_at_start(then_bb)
        then_stack_size = len(self.result_stack)
        self.visit(node.then)
        then_terminated = self._llvm.ir_builder.block.terminator is not None
        then_v: Optional[ir.Value | ir.Function] = None
        if len(self.result_stack) > then_stack_size:
            then_v = self.result_stack.pop()
        if not then_terminated:
            self._llvm.ir_builder.branch(merge_bb)
            then_bb = self._llvm.ir_builder.block

        # Emit else block.
        self._llvm.ir_builder.position_at_start(else_bb)
        else_stack_size = len(self.result_stack)
        if node.else_ is not None:
            self.visit(node.else_)
        else_terminated = self._llvm.ir_builder.block.terminator is not None
        else_v: Optional[ir.Value | ir.Function] = None
        if len(self.result_stack) > else_stack_size:
            else_v = self.result_stack.pop()
        if not else_terminated:
            self._llvm.ir_builder.branch(merge_bb)
            else_bb = self._llvm.ir_builder.block

        then_reaches_merge = not then_terminated
        else_reaches_merge = not else_terminated
        if not then_reaches_merge and not else_reaches_merge:
            self._llvm.ir_builder.position_at_start(merge_bb)
            self._llvm.ir_builder.unreachable()
            return

        # Emit merge block and PHI node when both branches produce compatible
        # values and can reach the merge block.
        self._llvm.ir_builder.position_at_start(merge_bb)
        if (
            then_reaches_merge
            and else_reaches_merge
            and then_v is not None
            and else_v is not None
            and then_v.type == else_v.type
        ):
            phi = self._llvm.ir_builder.phi(then_v.type, "iftmp")
            phi.add_incoming(then_v, then_bb)
            phi.add_incoming(else_v, else_bb)
            self.result_stack.append(phi)
            return

        if (
            then_reaches_merge
            and not else_reaches_merge
            and then_v is not None
        ):
            self.result_stack.append(then_v)
            return

        if (
            else_reaches_merge
            and not then_reaches_merge
            and else_v is not None
        ):
            self.result_stack.append(else_v)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.WhileStmt) -> None:
        """
        title: Translate ASTx While Loop to LLVM-IR.
        parameters:
          expr:
            type: astx.WhileStmt
        """
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
        self._llvm.ir_builder.position_at_end(cond_bb)

        # Emit the condition.
        self.visit(expr.condition)
        cond_val = self.result_stack.pop()
        if not cond_val:
            raise Exception("codegen: Invalid condition expression.")

        # Convert condition to a bool by comparing non-equal to 0.
        if is_fp_type(cond_val.type):
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

        #  use position_at_end for body block
        self._llvm.ir_builder.position_at_end(body_bb)

        # Emit the body of the loop.
        self.visit(expr.body)
        body_val = self.result_stack.pop()

        if not body_val:
            return

        # Don't rely on result_stack for control flow.
        # Only branch back if the block isn't already terminated
        if not self._llvm.ir_builder.block.is_terminated:
            self._llvm.ir_builder.branch(cond_bb)

        # use position_at_end for after block
        self._llvm.ir_builder.position_at_end(after_bb)

        # While loop always returns 0.
        result = ir.Constant(self._llvm.INT32_TYPE, 0)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.VariableAssignment) -> None:
        """
        title: Translate variable assignment expression.
        parameters:
          expr:
            type: astx.VariableAssignment
        """
        # Get the name of the variable to assign to
        var_name = expr.name

        if var_name in self.const_vars:
            raise Exception(
                f"Cannot assign to '{var_name}': declared as constant"
            )
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
        """
        title: Translate ASTx For Range Loop to LLVM-IR.
        parameters:
          node:
            type: astx.ForCountLoopStmt
        """
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
            0,
        )
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.ForRangeLoopStmt) -> None:
        """
        title: Translate ASTx For Range Loop to LLVM-IR.
        parameters:
          node:
            type: astx.ForRangeLoopStmt
        """
        saved_block = self._llvm.ir_builder.block

        var_addr = self.create_entry_block_alloca(
            "for_count_loop",
            node.variable.type_.__class__.__name__.lower(),
        )

        self._llvm.ir_builder.position_at_end(saved_block)

        # initialize start value
        self.visit(node.start)
        start_val = self.result_stack.pop()
        if not start_val:
            raise Exception("codegen: Invalid start argument.")

        self._llvm.ir_builder.store(start_val, var_addr)

        # create blocks
        func = self._llvm.ir_builder.function

        header_bb = func.append_basic_block("for.header")
        body_bb = func.append_basic_block("for.body")
        after_bb = func.append_basic_block("for.after")

        # jump to header
        self._llvm.ir_builder.branch(header_bb)

        # LOOP HEADER  (condition checked before body)
        self._llvm.ir_builder.position_at_start(header_bb)

        cur_var = self._llvm.ir_builder.load(var_addr, node.variable.name)

        self.visit(node.end)
        end_val = self.result_stack.pop()
        if not end_val:
            raise Exception("codegen: Invalid end argument.")

        if node.step:
            self.visit(node.step)
            step_val = self.result_stack.pop()
            if not step_val:
                raise Exception("codegen: Invalid step argument.")
        else:
            step_val = ir.Constant(
                self._llvm.get_data_type(
                    node.variable.type_.__class__.__name__.lower()
                ),
                1,
            )

        # comparison
        if is_fp_type(cur_var.type):
            cmp_instruction = self._llvm.ir_builder.fcmp_ordered
            cmp_op = (
                "<"
                if isinstance(step_val, ir.Constant) and step_val.constant > 0
                else ">"
            )
        else:
            cmp_instruction = self._llvm.ir_builder.icmp_signed
            cmp_op = (
                "<"
                if isinstance(step_val, ir.Constant) and step_val.constant > 0
                else ">"
            )

        loop_cond = cmp_instruction(
            cmp_op,
            cur_var,
            end_val,
            "loopcond",
        )

        # condition decides entry into body
        self._llvm.ir_builder.cbranch(loop_cond, body_bb, after_bb)

        # LOOP BODY
        self._llvm.ir_builder.position_at_start(body_bb)

        old_val = self.named_values.get(node.variable.name)
        self.named_values[node.variable.name] = var_addr

        self.visit(node.body)
        _ = self.result_stack.pop()

        # increment
        cur_var = self._llvm.ir_builder.load(var_addr, node.variable.name)
        next_var = emit_add(
            self._llvm.ir_builder, cur_var, step_val, "nextvar"
        )
        self._llvm.ir_builder.store(next_var, var_addr)

        self._llvm.ir_builder.branch(header_bb)

        # AFTER LOOP
        self._llvm.ir_builder.position_at_start(after_bb)

        if old_val:
            self.named_values[node.variable.name] = old_val
        else:
            self.named_values.pop(node.variable.name, None)

        result = ir.Constant(
            self._llvm.get_data_type(
                node.variable.type_.__class__.__name__.lower()
            ),
            0,
        )

        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.Module) -> None:
        """
        title: Translate ASTx Module to LLVM-IR.
        parameters:
          node:
            type: astx.Module
        """
        for mod_node in node.nodes:
            self.visit(mod_node)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralInt32) -> None:
        """
        title: Translate ASTx LiteralInt32 to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralInt32
        """
        result = ir.Constant(self._llvm.INT32_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralFloat32) -> None:
        """
        title: Translate ASTx LiteralFloat32 to LLVM-IR.
        parameters:
          expr:
            type: astx.LiteralFloat32
        """
        result = ir.Constant(self._llvm.FLOAT_TYPE, expr.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralFloat64) -> None:
        """
        title: Translate ASTx LiteralFloat64 to LLVM-IR.
        parameters:
          expr:
            type: astx.LiteralFloat64
        """
        result = ir.Constant(self._llvm.DOUBLE_TYPE, expr.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralFloat16) -> None:
        """
        title: Translate ASTx LiteralFloat16 to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralFloat16
        """
        result = ir.Constant(self._llvm.FLOAT16_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralNone) -> None:
        """
        title: Translate ASTx LiteralNone to LLVM-IR.
        parameters:
          expr:
            type: astx.LiteralNone
        """
        self.result_stack.append(None)  # No IR emitted for void

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralBoolean) -> None:
        """
        title: Translate ASTx LiteralBoolean to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralBoolean
        """
        result = ir.Constant(self._llvm.BOOLEAN_TYPE, int(node.value))
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralInt64) -> None:
        """
        title: Translate ASTx LiteralInt64 to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralInt64
        """
        result = ir.Constant(self._llvm.INT64_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralInt8) -> None:
        """
        title: Translate ASTx LiteralInt8 to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralInt8
        """
        result = ir.Constant(self._llvm.INT8_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralUInt8) -> None:
        """
        title: Translate ASTx LiteralUInt8 to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralUInt8
        """
        result = ir.Constant(self._llvm.UINT8_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralUInt16) -> None:
        """
        title: Translate ASTx LiteralUInt16 to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralUInt16
        """
        result = ir.Constant(self._llvm.UINT16_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralUInt32) -> None:
        """
        title: Translate ASTx LiteralUInt32 to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralUInt32
        """
        result = ir.Constant(self._llvm.UINT32_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralUInt64) -> None:
        """
        title: Translate ASTx LiteralUInt64 to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralUInt64
        """
        result = ir.Constant(self._llvm.UINT64_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralUInt128) -> None:
        """
        title: Translate ASTx LiteralUInt128 to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralUInt128
        """
        result = ir.Constant(self._llvm.UINT128_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralUTF8Char) -> None:
        """
        title: Handle ASCII string literals.
        parameters:
          expr:
            type: astx.LiteralUTF8Char
        """
        string_value = expr.value
        utf8_bytes = string_value.encode("utf-8")
        string_length = len(utf8_bytes)

        # Create a global constant for the string data
        string_data_type = ir.ArrayType(
            self._llvm.INT8_TYPE, string_length + 1
        )
        string_data = ir.GlobalVariable(
            self._llvm.module, string_data_type, name=f"str_ascii_{id(expr)}"
        )
        string_data.linkage = "internal"
        string_data.global_constant = True
        string_data.initializer = ir.Constant(
            string_data_type, bytearray(string_value + "\0", "ascii")
        )

        ptr = self._llvm.ir_builder.gep(
            string_data,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)],
            inbounds=True,
        )

        self.result_stack.append(ptr)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralUTF8String) -> None:
        """
        title: Handle UTF-8 string literals.
        parameters:
          expr:
            type: astx.LiteralUTF8String
        """
        string_value = expr.value
        utf8_bytes = string_value.encode("utf-8")
        string_length = len(utf8_bytes)

        # Create a global constant for the string data
        string_data_type = ir.ArrayType(
            self._llvm.INT8_TYPE, string_length + 1
        )
        unique_name = f"str_utf8_{abs(hash(string_value))}_{id(expr)}"
        string_data = ir.GlobalVariable(
            self._llvm.module, string_data_type, name=unique_name
        )
        string_data.linkage = "internal"
        string_data.global_constant = True
        string_data.initializer = ir.Constant(
            string_data_type, bytearray(utf8_bytes + b"\0")
        )

        # Get pointer to the string data (i8*)
        data_ptr = self._llvm.ir_builder.gep(
            string_data,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)],
            inbounds=True,
        )

        self.result_stack.append(data_ptr)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralString) -> None:
        """
        title: Handle generic string literals - defaults to UTF-8.
        parameters:
          expr:
            type: astx.LiteralString
        """
        utf8_literal = astx.LiteralUTF8String(value=expr.value)
        self.visit(utf8_literal)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralTime) -> None:
        """
        title: Lower a LiteralTime to LLVM IR.
        summary: >-
          Representation: { i32 hour, i32 minute, i32 second } emitted as a
          constant struct. Accepted formats are HH:MM and HH:MM:SS.
        parameters:
          node:
            type: astx.LiteralTime
        """
        s = node.value.strip()

        parts = s.split(":")
        if len(parts) not in (2, 3):
            raise Exception(
                f"LiteralTime: invalid time format '{node.value}'. "
                "Expected 'HH:MM' or 'HH:MM:SS'."
            )

        # Parse hour, minute
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except Exception as exc:
            raise Exception(
                f"LiteralTime: invalid hour/minute in '{node.value}'."
            ) from exc

        # Parse second (optional)
        if len(parts) == 3:  # noqa: PLR2004
            sec_part = parts[2]
            if "." in sec_part:
                raise Exception(
                    "LiteralTime: fractional seconds "
                    f"not supported in '{node.value}'."
                )
            try:
                second = int(sec_part)
            except Exception as exc:
                raise Exception(
                    f"LiteralTime: invalid seconds in '{node.value}'."
                ) from exc
        else:
            second = 0

        # Range checks
        MAX_HOUR = 23
        MAX_MINUTE = 59
        MAX_SECOND = 59
        if not (0 <= hour <= MAX_HOUR):
            raise Exception(
                f"LiteralTime: hour out of range in '{node.value}'."
            )
        if not (0 <= minute <= MAX_MINUTE):
            raise Exception(
                f"LiteralTime: minute out of range in '{node.value}'."
            )
        if not (0 <= second <= MAX_SECOND):
            raise Exception(
                f"LiteralTime: second out of range in '{node.value}'."
            )

        # Build constant struct { i32, i32, i32 }
        i32 = self._llvm.INT32_TYPE
        const_time = ir.Constant(
            self._llvm.TIME_TYPE,
            [
                ir.Constant(i32, hour),
                ir.Constant(i32, minute),
                ir.Constant(i32, second),
            ],
        )
        self.result_stack.append(const_time)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralTimestamp) -> None:
        """
        title: Lower a LiteralTimestamp to a constant struct.
        summary: >-
          Layout is { i32 year, i32 month, i32 day, i32 hour, i32 minute, i32
          second, i32 nanos }. Accepted formats (no timezone) are YYYY-MM-
          DDTHH:MM:SS[.fffffffff] and YYYY-MM-DD HH:MM:SS[.fffffffff].
        parameters:
          node:
            type: astx.LiteralTimestamp
        """
        s = node.value.strip()

        # Split date and time by 'T' or space.
        if "T" in s:
            date_part, time_part = s.split("T", 1)
        elif " " in s:
            date_part, time_part = s.split(" ", 1)
        else:
            raise Exception(
                "LiteralTimestamp: invalid format '"
                f"{node.value}'. Expected 'YYYY-MM-DDTHH:MM:SS"
                "[.fffffffff]' (or space instead of 'T')."
            )

        # Reject timezone suffixes for now.
        if time_part.endswith("Z") or "+" in time_part or "-" in time_part[2:]:
            raise Exception(
                "LiteralTimestamp: timezone offsets not supported in '"
                f"{node.value}'."
            )

        # Parse and validate date: YYYY-MM-DD
        try:
            y_str, m_str, d_str = date_part.split("-")
            year = int(y_str)
            month = int(m_str)
            day = int(d_str)
            # Validate real calendar date (handles month/day/leap years)
            datetime(year, month, day)
        except ValueError as exc:
            raise Exception(
                "LiteralTimestamp: invalid date in '"
                f"{node.value}'. Expected valid 'YYYY-MM-DD'."
            ) from exc
        except Exception as exc:
            raise Exception(
                "LiteralTimestamp: invalid date part in '"
                f"{node.value}'. Expected 'YYYY-MM-DD'."
            ) from exc

        # Parse time: HH:MM:SS(.fffffffff)?
        # Named bounds to avoid magic numbers
        NS_DIGITS = 9
        MAX_HOUR = 23
        MAX_MINUTE = 59
        MAX_SECOND = 59

        frac_ns = 0
        try:
            if "." in time_part:
                hms, frac = time_part.split(".", 1)
                if not frac.isdigit():
                    raise ValueError("fractional seconds must be digits")
                if len(frac) > NS_DIGITS:
                    frac = frac[:NS_DIGITS]
                frac_ns = int(frac.ljust(NS_DIGITS, "0"))
            else:
                hms = time_part

            h_str, m_str, s_str = hms.split(":")
            hour = int(h_str)
            minute = int(m_str)
            second = int(s_str)
        except Exception as exc:
            raise Exception(
                "LiteralTimestamp: invalid time part in '"
                f"{node.value}'. Expected 'HH:MM:SS'"
                " (optionally with '.fffffffff')."
            ) from exc

        if not (0 <= hour <= MAX_HOUR):
            raise Exception(
                f"LiteralTimestamp: hour out of range in '{node.value}'."
            )
        if not (0 <= minute <= MAX_MINUTE):
            raise Exception(
                f"LiteralTimestamp: minute out of range in '{node.value}'."
            )
        if not (0 <= second <= MAX_SECOND):
            raise Exception(
                f"LiteralTimestamp: second out of range in '{node.value}'."
            )

        i32 = self._llvm.INT32_TYPE
        const_ts = ir.Constant(
            self._llvm.TIMESTAMP_TYPE,
            [
                ir.Constant(i32, year),
                ir.Constant(i32, month),
                ir.Constant(i32, day),
                ir.Constant(i32, hour),
                ir.Constant(i32, minute),
                ir.Constant(i32, second),
                ir.Constant(i32, frac_ns),
            ],
        )
        self.result_stack.append(const_ts)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralDateTime) -> None:
        """
        title: Lower a LiteralDateTime to a constant struct.
        summary: >-
          Layout is { i32 year, i32 month, i32 day, i32 hour, i32 minute, i32
          second }. Accepted formats (no timezone, no fractional seconds) are
          YYYY-MM-DDTHH:MM and YYYY-MM-DDTHH:MM:SS (space may be used instead
          of T).
        parameters:
          node:
            type: astx.LiteralDateTime
        """
        s = node.value.strip()

        # Split date and time by 'T' or space.
        if "T" in s:
            date_part, time_part = s.split("T", 1)
        elif " " in s:
            date_part, time_part = s.split(" ", 1)
        else:
            raise ValueError(
                f"LiteralDateTime: invalid format '{node.value}'. "
                "Expected 'YYYY-MM-DDTHH:MM[:SS]' (or space instead of 'T')."
            )

        # Disallow fractional seconds and timezone suffixes here.
        if "." in time_part:
            raise ValueError(
                f"LiteralDateTime: fractional seconds not supported in "
                f"'{node.value}'. Use LiteralTimestamp instead."
            )
        if time_part.endswith("Z") or "+" in time_part or "-" in time_part[2:]:
            raise ValueError(
                f"LiteralDateTime: timezone offsets not supported in "
                f"'{node.value}'. Use LiteralTimestamp for timezones."
            )

        # Parse date: YYYY-MM-DD
        try:
            y_str, m_str, d_str = date_part.split("-")
            year = int(y_str)
            month = int(m_str)
            day = int(d_str)
        except Exception as exc:
            raise ValueError(
                f"LiteralDateTime: invalid date part in '{node.value}'. "
                "Expected 'YYYY-MM-DD'."
            ) from exc

        # Validate i32 range for year
        INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
        if not (INT32_MIN <= year <= INT32_MAX):
            raise ValueError(
                f"LiteralDateTime: year out of 32-bit range in '{node.value}'."
            )

        # Parse time: HH:MM[:SS]
        HOUR_MINUTE_ONLY = 2
        HOUR_MINUTE_SECOND = 3
        try:
            parts = time_part.split(":")
            if len(parts) not in (HOUR_MINUTE_ONLY, HOUR_MINUTE_SECOND):
                raise ValueError("time must be HH:MM or HH:MM:SS")
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) == HOUR_MINUTE_SECOND else 0
        except Exception as exc:
            raise ValueError(
                f"LiteralDateTime: invalid time part in '{node.value}'. "
                "Expected 'HH:MM' or 'HH:MM:SS'."
            ) from exc

        # Named bounds for time validation
        MAX_HOUR = 23
        MAX_MINUTE_SECOND = 59
        if not (0 <= hour <= MAX_HOUR):
            raise ValueError(
                f"LiteralDateTime: hour out of range in '{node.value}'."
            )
        if not (0 <= minute <= MAX_MINUTE_SECOND):
            raise ValueError(
                f"LiteralDateTime: minute out of range in '{node.value}'."
            )
        if not (0 <= second <= MAX_MINUTE_SECOND):
            raise ValueError(
                f"LiteralDateTime: second out of range in '{node.value}'."
            )

        # Validate calendar date and time (handles month/day/leap years)
        try:
            datetime(year, month, day)
            _time(hour, minute, second)
        except ValueError as exc:
            raise ValueError(
                f"LiteralDateTime: invalid calendar date/time in "
                f"'{node.value}'."
            ) from exc

        # Build constant using shared DATETIME_TYPE
        i32 = self._llvm.INT32_TYPE
        const_dt = ir.Constant(
            self._llvm.DATETIME_TYPE,
            [
                ir.Constant(i32, year),
                ir.Constant(i32, month),
                ir.Constant(i32, day),
                ir.Constant(i32, hour),
                ir.Constant(i32, minute),
                ir.Constant(i32, second),
            ],
        )

        self.result_stack.append(const_dt)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralList) -> None:
        """
        title: Lower a LiteralList to LLVM IR (minimal support).
        summary: >-
          Supported cases are empty list (constant [0 x i32]) and homogeneous
          integer constant lists (constant [N x iX]). Otherwise raises to keep
          behavior explicit and aligned with current test-suite conventions.
        parameters:
          node:
            type: astx.LiteralList
        """
        # Lower each element and collect the LLVM values
        llvm_elems: list[ir.Value] = []
        for elem in node.elements:
            self.visit(elem)
            v = self.result_stack.pop()
            if v is None:
                raise Exception("LiteralList: invalid element lowering.")
            llvm_elems.append(v)

        n = len(llvm_elems)
        # Empty list => [0 x i32] constant
        # TODO: Infer element type from declared list type when available.
        # Currently uses i32 as placeholder; update when non-int lists
        # are supported.
        if n == 0:
            empty_ty = ir.ArrayType(self._llvm.INT32_TYPE, 0)
            self.result_stack.append(ir.Constant(empty_ty, []))
            return

        # Homogeneous integer constant lists => constant array
        first_ty = llvm_elems[0].type
        is_ints = all(is_int_type(v.type) for v in llvm_elems)
        homogeneous = all(v.type == first_ty for v in llvm_elems)
        all_constants = all(isinstance(v, ir.Constant) for v in llvm_elems)
        if is_ints and homogeneous and all_constants:
            arr_ty = ir.ArrayType(first_ty, n)
            const_arr = ir.Constant(arr_ty, llvm_elems)
            self.result_stack.append(const_arr)
            return

        raise TypeError(
            "LiteralList: only empty or homogeneous integer constants "
            "are supported"
        )

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralSet) -> None:
        """
        title: Lower a LiteralSet to LLVM IR.
        parameters:
          node:
            type: astx.LiteralSet
        """

        # Sort elements deterministically for stable IR output
        def _sort_key(lit: astx.Literal) -> tuple[str, Any]:
            tname = type(lit).__name__
            val = getattr(lit, "value", None)
            return (
                tname,
                val if isinstance(val, (int, float, str)) else repr(lit),
            )

        elems_sorted = sorted(node.elements, key=_sort_key)

        # Lower each element and collect the LLVM values
        llvm_elems: list[ir.Value] = []
        for elem in elems_sorted:
            self.visit(elem)
            v = self.result_stack.pop()
            if v is None:
                raise Exception("LiteralSet: invalid element lowering.")
            llvm_elems.append(v)

        n = len(llvm_elems)

        # Empty set
        if n == 0:
            empty_ty = ir.ArrayType(self._llvm.INT32_TYPE, 0)
            self.result_stack.append(ir.Constant(empty_ty, []))
            return

        first_ty = llvm_elems[0].type
        is_ints = all(isinstance(v.type, ir.IntType) for v in llvm_elems)
        homogeneous = all(v.type == first_ty for v in llvm_elems)
        all_constants = all(isinstance(v, ir.Constant) for v in llvm_elems)

        # Homogeneous integer constants → constant array
        if is_ints and homogeneous and all_constants:
            arr_ty = ir.ArrayType(first_ty, n)
            const_arr = ir.Constant(arr_ty, llvm_elems)
            self.result_stack.append(const_arr)
            return

        # Mixed-width integer constants → promote to widest type
        if is_ints and all_constants and not homogeneous:
            widest = max(v.type.width for v in llvm_elems)
            elem_ty = ir.IntType(widest)
            arr_ty = ir.ArrayType(elem_ty, n)

            builder = self._llvm.ir_builder

            # If outside function context (tests), fallback to constant array
            if builder.block is None:
                promoted_vals: list[ir.Constant] = []
                for v in llvm_elems:
                    if v.type.width != widest:
                        promoted_vals.append(ir.Constant(elem_ty, v.constant))
                    else:
                        promoted_vals.append(v)

                const_arr = ir.Constant(arr_ty, promoted_vals)
                self.result_stack.append(const_arr)
                return

            # Runtime lowering using alloca + store
            entry_bb = builder.function.entry_basic_block
            current_bb = builder.block

            builder.position_at_start(entry_bb)
            alloca = builder.alloca(arr_ty, name="set.lit")
            builder.position_at_end(current_bb)

            i32 = self._llvm.INT32_TYPE

            for i, v in enumerate(llvm_elems):
                cast_val = v
                if cast_val.type != elem_ty:
                    cast_val = builder.sext(
                        cast_val, elem_ty, name=f"set_sext{i}"
                    )

                ptr = builder.gep(
                    alloca,
                    [ir.Constant(i32, 0), ir.Constant(i32, i)],
                    inbounds=True,
                )

                builder.store(cast_val, ptr)

            self.result_stack.append(alloca)
            return

        raise TypeError(
            "LiteralSet: only integer constants are currently supported "
            "(homogeneous or mixed-width)"
        )

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralTuple) -> None:
        """
        title: LiteralTuple lowering
        parameters:
          node:
            type: astx.LiteralTuple
        """

        # Lower each element and collect LLVM values
        llvm_elems: list[ir.Value] = []
        for elem in node.elements:
            self.visit(elem)
            v = self.result_stack.pop()
            if v is None:
                raise Exception("LiteralTuple: invalid element lowering.")
            llvm_elems.append(v)

        n = len(llvm_elems)

        # Empty tuple -> constant empty struct {}
        if n == 0:
            struct_ty = ir.LiteralStructType([])
            self.result_stack.append(ir.Constant(struct_ty, []))
            return

        first_ty = llvm_elems[0].type
        homogeneous = all(v.type == first_ty for v in llvm_elems)
        all_constants = all(isinstance(v, ir.Constant) for v in llvm_elems)

        if homogeneous and all_constants:
            struct_ty = ir.LiteralStructType([first_ty] * n)
            const_struct = ir.Constant(struct_ty, llvm_elems)
            self.result_stack.append(const_struct)
            return

        raise TypeError(
            "LiteralTuple: only empty or homogeneous constant tuples "
            "are supported"
        )

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralDict) -> None:
        """
        title: LiteralDict lowering
        parameters:
          node:
            type: astx.LiteralDict
        """
        # 1) Collect lowered key/value LLVM values
        llvm_pairs: list[tuple[ir.Value, ir.Value]] = []

        for key_node, value_node in node.elements.items():
            # Lower key
            self.visit(key_node)
            key_val = self.result_stack.pop()
            if key_val is None:
                raise Exception("LiteralDict: failed to lower key.")

            # Lower value
            self.visit(value_node)
            val_val = self.result_stack.pop()
            if val_val is None:
                raise Exception("LiteralDict: failed to lower value.")

            llvm_pairs.append((key_val, val_val))

        n = len(llvm_pairs)

        # 2) Empty dict -> constant empty array with placeholder struct type
        if n == 0:
            # Placeholder element types (same philosophy as LiteralList)
            pair_ty = ir.LiteralStructType(
                [self._llvm.INT32_TYPE, self._llvm.INT32_TYPE]
            )
            arr_ty = ir.ArrayType(pair_ty, 0)
            self.result_stack.append(ir.Constant(arr_ty, []))
            return

        # 3) Check constant fast-path
        all_constants = all(
            isinstance(k, ir.Constant) and isinstance(v, ir.Constant)
            for k, v in llvm_pairs
        )

        if all_constants:
            # Infer struct type from first pair
            first_key_ty = llvm_pairs[0][0].type
            first_val_ty = llvm_pairs[0][1].type

            pair_ty = ir.LiteralStructType([first_key_ty, first_val_ty])
            arr_ty = ir.ArrayType(pair_ty, n)

            struct_consts: list[ir.Constant] = []

            for key_val, val_val in llvm_pairs:
                # Ensure homogeneous types
                if (
                    key_val.type != first_key_ty
                    or val_val.type != first_val_ty
                ):
                    raise TypeError(
                        "LiteralDict: heterogeneous constant key/value types "
                        "are not yet supported"
                    )

                struct_consts.append(ir.Constant(pair_ty, [key_val, val_val]))

            const_arr = ir.Constant(arr_ty, struct_consts)
            self.result_stack.append(const_arr)
            return

        # 4) Non-constant path not yet supported
        raise TypeError(
            "LiteralDict: only empty or all-constant dictionaries "
            "are supported in this version"
        )

    def _create_string_concat_function(self) -> ir.Function:
        """
        title: Create a string concatenation function.
        returns:
          type: ir.Function
        """
        func_name = "string_concat"
        if func_name in self._llvm.module.globals:
            return self._llvm.module.get_global(func_name)

        func_type = ir.FunctionType(
            self._llvm.ASCII_STRING_TYPE,
            [self._llvm.ASCII_STRING_TYPE, self._llvm.ASCII_STRING_TYPE],
        )
        func = ir.Function(self._llvm.module, func_type, func_name)

        func.linkage = "external"
        return func

    def _create_string_length_function(self) -> ir.Function:
        """
        title: Create a string length function.
        returns:
          type: ir.Function
        """
        func_name = "string_length"
        if func_name in self._llvm.module.globals:
            return self._llvm.module.get_global(func_name)

        # Function signature: string_length(char* str) -> i32
        func_type = ir.FunctionType(
            self._llvm.INT32_TYPE, [self._llvm.ASCII_STRING_TYPE]
        )
        func = ir.Function(self._llvm.module, func_type, func_name)
        block = func.append_basic_block("entry")
        builder = ir.IRBuilder(block)
        strlen_func = self._create_strlen_inline()
        result = builder.call(strlen_func, [func.args[0]], "len")
        builder.ret(result)
        return func

    def _create_string_equals_function(self) -> ir.Function:
        """
        title: Create a string equality comparison function.
        returns:
          type: ir.Function
        """
        func_name = "string_equals"
        if func_name in self._llvm.module.globals:
            return self._llvm.module.get_global(func_name)

        # Function signature: string_equals(char* str1, char* str2) -> i1
        func_type = ir.FunctionType(
            self._llvm.BOOLEAN_TYPE,
            [self._llvm.ASCII_STRING_TYPE, self._llvm.ASCII_STRING_TYPE],
        )
        func = ir.Function(self._llvm.module, func_type, func_name)

        block = func.append_basic_block("entry")
        builder = ir.IRBuilder(block)
        strcmp_func = self._create_strcmp_inline()
        result = builder.call(strcmp_func, [func.args[0], func.args[1]], "cmp")
        builder.ret(result)
        return func

    def _create_string_substring_function(self) -> ir.Function:
        """
        title: Create a string substring function.
        returns:
          type: ir.Function
        """
        func_name = "string_substring"
        if func_name in self._llvm.module.globals:
            return self._llvm.module.get_global(func_name)

        # string_substring(char* str, i32 start, i32 length) -> char*
        func_type = ir.FunctionType(
            self._llvm.ASCII_STRING_TYPE,
            [
                self._llvm.ASCII_STRING_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT32_TYPE,
            ],
        )
        func = ir.Function(self._llvm.module, func_type, func_name)
        func.linkage = "external"
        return func

    def _handle_string_concatenation(
        self, lhs: ir.Value, rhs: ir.Value
    ) -> ir.Value:
        """
        title: Handle string concatenation operation using inline function.
        parameters:
          lhs:
            type: ir.Value
          rhs:
            type: ir.Value
        returns:
          type: ir.Value
        """
        strcat_func = self._create_strcat_inline()
        return self._llvm.ir_builder.call(
            strcat_func, [lhs, rhs], "str_concat"
        )

    def _create_strcat_inline(self) -> ir.Function:
        """
        title: Create an inline string concatenation function in LLVM IR.
        returns:
          type: ir.Function
        """
        func_name = "strcat_inline"
        if func_name in self._llvm.module.globals:
            return self._llvm.module.get_global(func_name)

        func_type = ir.FunctionType(
            self._llvm.INT8_TYPE.as_pointer(),
            [
                self._llvm.INT8_TYPE.as_pointer(),
                self._llvm.INT8_TYPE.as_pointer(),
            ],
        )
        func = ir.Function(self._llvm.module, func_type, func_name)

        entry = func.append_basic_block("entry")
        builder = ir.IRBuilder(entry)

        strlen_func = self._create_string_length_function()
        len1 = builder.call(strlen_func, [func.args[0]], "len1")
        len2 = builder.call(strlen_func, [func.args[1]], "len2")

        # Total length = len1 + len2 + 1 (for null terminator)
        total_len = builder.add(len1, len2, "total_len")
        total_len = builder.add(
            total_len,
            ir.Constant(self._llvm.INT32_TYPE, 1),
            "total_len_with_null",
        )

        # Allocate on heap to avoid use-after-return
        malloc = self._create_malloc_decl()
        total_len_szt = builder.zext(total_len, self._llvm.SIZE_T_TYPE)
        result_ptr = builder.call(malloc, [total_len_szt], "result")

        self._generate_strcpy(builder, result_ptr, func.args[0])

        result_end = builder.gep(result_ptr, [len1], inbounds=True)
        self._generate_strcpy(builder, result_end, func.args[1])

        builder.ret(result_ptr)
        return func

    def _generate_strcpy(
        self, builder: ir.IRBuilder, dest: ir.Value, src: ir.Value
    ) -> None:
        """
        title: Generate inline string copy code.
        parameters:
          builder:
            type: ir.IRBuilder
          dest:
            type: ir.Value
          src:
            type: ir.Value
        """
        loop_bb = builder.function.append_basic_block("strcpy_loop")
        end_bb = builder.function.append_basic_block("strcpy_end")

        index = builder.alloca(self._llvm.INT32_TYPE, name="strcpy_index")
        builder.store(ir.Constant(self._llvm.INT32_TYPE, 0), index)
        builder.branch(loop_bb)

        builder.position_at_start(loop_bb)
        idx_val = builder.load(index, "idx_val")

        src_char_ptr = builder.gep(src, [idx_val], inbounds=True)
        char_val = builder.load(src_char_ptr, "char_val")

        dest_char_ptr = builder.gep(dest, [idx_val], inbounds=True)
        builder.store(char_val, dest_char_ptr)

        is_null = builder.icmp_signed(
            "==", char_val, ir.Constant(self._llvm.INT8_TYPE, 0)
        )

        next_idx = builder.add(idx_val, ir.Constant(self._llvm.INT32_TYPE, 1))
        builder.store(next_idx, index)

        builder.cbranch(is_null, end_bb, loop_bb)

        builder.position_at_start(end_bb)

    def _create_strcmp_inline(self) -> ir.Function:
        """
        title: Create an inline strcmp function in LLVM IR.
        returns:
          type: ir.Function
        """
        func_name = "strcmp_inline"
        if func_name in self._llvm.module.globals:
            return self._llvm.module.get_global(func_name)

        func_type = ir.FunctionType(
            self._llvm.BOOLEAN_TYPE,
            [
                self._llvm.INT8_TYPE.as_pointer(),
                self._llvm.INT8_TYPE.as_pointer(),
            ],
        )
        func = ir.Function(self._llvm.module, func_type, func_name)

        entry = func.append_basic_block("entry")
        loop = func.append_basic_block("loop")
        not_equal = func.append_basic_block("not_equal")
        equal = func.append_basic_block("equal")

        builder = ir.IRBuilder(entry)

        index = builder.alloca(self._llvm.INT32_TYPE, name="index")

        builder.store(ir.Constant(self._llvm.INT32_TYPE, 0), index)
        builder.branch(loop)

        builder.position_at_start(loop)
        idx_val = builder.load(index, "idx_val")

        char1_ptr = builder.gep(func.args[0], [idx_val], inbounds=True)
        char2_ptr = builder.gep(func.args[1], [idx_val], inbounds=True)

        char1 = builder.load(char1_ptr, "char1")
        char2 = builder.load(char2_ptr, "char2")

        chars_equal = builder.icmp_signed("==", char1, char2)

        char1_null = builder.icmp_signed(
            "==", char1, ir.Constant(self._llvm.INT8_TYPE, 0)
        )

        builder.cbranch(
            chars_equal,
            builder.function.append_basic_block("check_null"),
            not_equal,
        )

        check_null_bb = builder.function.basic_blocks[-1]
        builder.position_at_start(check_null_bb)
        builder.cbranch(
            char1_null,
            equal,
            builder.function.append_basic_block("continue_loop"),
        )

        continue_bb = builder.function.basic_blocks[-1]
        builder.position_at_start(continue_bb)
        next_idx = builder.add(idx_val, ir.Constant(self._llvm.INT32_TYPE, 1))
        builder.store(next_idx, index)
        builder.branch(loop)

        builder.position_at_start(not_equal)
        builder.ret(ir.Constant(self._llvm.BOOLEAN_TYPE, 0))

        builder.position_at_start(equal)
        builder.ret(ir.Constant(self._llvm.BOOLEAN_TYPE, 1))

        return func

    def _create_strlen_inline(self) -> ir.Function:
        """
        title: Create an inline strlen function in LLVM IR.
        returns:
          type: ir.Function
        """
        func_name = "strlen_inline"
        if func_name in self._llvm.module.globals:
            return self._llvm.module.get_global(func_name)

        # Function signature: strlen_inline(char* str) -> i32
        func_type = ir.FunctionType(
            self._llvm.INT32_TYPE, [self._llvm.INT8_TYPE.as_pointer()]
        )
        func = ir.Function(self._llvm.module, func_type, func_name)

        entry = func.append_basic_block("entry")
        loop = func.append_basic_block("loop")
        end = func.append_basic_block("end")

        builder = ir.IRBuilder(entry)

        counter = builder.alloca(self._llvm.INT32_TYPE, name="counter")
        builder.store(ir.Constant(self._llvm.INT32_TYPE, 0), counter)

        builder.branch(loop)

        builder.position_at_start(loop)
        count_val = builder.load(counter, "count_val")

        char_ptr = builder.gep(func.args[0], [count_val], inbounds=True)
        char_val = builder.load(char_ptr, "char_val")

        is_null = builder.icmp_signed(
            "==", char_val, ir.Constant(self._llvm.INT8_TYPE, 0)
        )

        next_count = builder.add(
            count_val, ir.Constant(self._llvm.INT32_TYPE, 1)
        )
        builder.store(next_count, counter)

        builder.cbranch(is_null, end, loop)

        builder.position_at_start(end)
        builder.ret(count_val)

        return func

    def _handle_string_comparison(
        self, lhs: ir.Value, rhs: ir.Value, op: str
    ) -> ir.Value:
        """
        title: Handle string comparison operations using inline functions.
        parameters:
          lhs:
            type: ir.Value
          rhs:
            type: ir.Value
          op:
            type: str
        returns:
          type: ir.Value
        """
        if op == "==":
            equals_func = self._create_string_equals_function()
            return self._llvm.ir_builder.call(
                equals_func, [lhs, rhs], "str_equals"
            )
        elif op == "!=":
            equals_func = self._create_string_equals_function()
            equals_result = self._llvm.ir_builder.call(
                equals_func, [lhs, rhs], "str_equals"
            )
            return self._llvm.ir_builder.xor(
                equals_result,
                ir.Constant(self._llvm.BOOLEAN_TYPE, 1),
                "str_not_equals",
            )
        else:
            raise Exception(f"String comparison operator {op} not implemented")

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.FunctionCall) -> None:
        """
        title: Translate Function FunctionCall.
        parameters:
          node:
            type: astx.FunctionCall
        """
        # callee_f = self.get_function(node.fn)
        fn_name = node.fn

        callee_f = self.get_function(fn_name)
        if not callee_f:
            raise Exception("Unknown function referenced")

        if len(callee_f.args) != len(node.args):
            raise Exception("codegen: Incorrect # arguments passed.")

        llvm_args = []
        for arg in node.args:
            self.visit(arg)
            llvm_arg = self.result_stack.pop()
            if not llvm_arg:
                raise Exception("codegen: Invalid callee argument.")
            llvm_args.append(llvm_arg)

        result = self._llvm.ir_builder.call(callee_f, llvm_args, "calltmp")
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.FunctionDef) -> None:
        """
        title: Translate ASTx Function to LLVM-IR.
        parameters:
          node:
            type: astx.FunctionDef
        """
        proto = node.prototype
        self.function_protos[proto.name] = proto
        fn = self.get_function(proto.name)

        if not fn:
            raise Exception("Invalid function.")

        # Create a new basic block to start insertion into.
        basic_block = fn.append_basic_block("entry")
        self._llvm.ir_builder = ir.IRBuilder(basic_block)

        for idx, llvm_arg in enumerate(fn.args):
            arg_ast = proto.args.nodes[idx]
            type_str = arg_ast.type_.__class__.__name__.lower()
            arg_type = self._llvm.get_data_type(type_str)

            # Create an alloca for this variable.
            alloca = self._llvm.ir_builder.alloca(arg_type, name=llvm_arg.name)

            # Store the initial value into the alloca.
            self._llvm.ir_builder.store(llvm_arg, alloca)

            # Add arguments to variable symbol table.
            self.named_values[llvm_arg.name] = alloca

        self.visit(node.body)
        self.result_stack.append(fn)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.FunctionPrototype) -> None:
        """
        title: Translate ASTx Function Prototype to LLVM-IR.
        parameters:
          node:
            type: astx.FunctionPrototype
        """
        args_type = []
        for arg in node.args.nodes:
            type_str = arg.type_.__class__.__name__.lower()
            args_type.append(self._llvm.get_data_type(type_str))
        # note: it should be dynamic
        return_type = self._llvm.get_data_type(
            node.return_type.__class__.__name__.lower()
        )
        fn_type = ir.FunctionType(return_type, args_type, False)

        fn = ir.Function(self._llvm.module, fn_type, node.name)

        # Set names for all arguments.
        for idx, llvm_arg in enumerate(fn.args):
            llvm_arg.name = node.args.nodes[idx].name

        self.result_stack.append(fn)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.FunctionReturn) -> None:
        """
        title: Translate ASTx FunctionReturn to LLVM-IR.
        parameters:
          node:
            type: astx.FunctionReturn
        """
        if node.value is not None:
            self.visit(node.value)
            try:
                retval = self.result_stack.pop()
            except IndexError:
                retval = None
        else:
            retval = None

        if retval is not None:
            fn_return_type = (
                self._llvm.ir_builder.function.function_type.return_type
            )
            if is_int_type(fn_return_type) and fn_return_type.width == 1:
                # Force cast retval to i1 if not already
                if is_int_type(retval.type) and retval.type.width != 1:
                    retval = self._llvm.ir_builder.trunc(retval, ir.IntType(1))
            self._llvm.ir_builder.ret(retval)
            return
        self._llvm.ir_builder.ret_void()

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.InlineVariableDeclaration) -> None:
        """
        title: Translate an ASTx InlineVariableDeclaration expression.
        parameters:
          node:
            type: astx.InlineVariableDeclaration
        """
        if self.named_values.get(node.name):
            raise Exception(f"Identifier already declared: {node.name}")

        type_str = node.type_.__class__.__name__.lower()

        # Emit the initializer
        if node.value is not None:
            self.visit(node.value)
            init_val = self.result_stack.pop()
            if init_val is None:
                raise Exception("Initializer code generation failed.")
        # Default zero value based on type
        elif "float" in type_str:
            init_val = ir.Constant(self._llvm.get_data_type(type_str), 0.0)
        else:
            init_val = ir.Constant(self._llvm.get_data_type(type_str), 0)

        if type_str == "string":
            alloca = self.create_entry_block_alloca(node.name, "stringascii")
        else:
            alloca = self.create_entry_block_alloca(node.name, type_str)

        self._llvm.ir_builder.store(init_val, alloca)
        if node.mutability == astx.MutabilityKind.constant:
            self.const_vars.add(node.name)
        self.named_values[node.name] = alloca

        self.result_stack.append(init_val)

    def _normalize_int_for_printf(self, v: ir.Value) -> tuple[ir.Value, str]:
        """
        title: Promote/truncate integer to match printf format.
        parameters:
          v:
            type: ir.Value
        returns:
          type: tuple[ir.Value, str]
        """
        INT64_WIDTH = 64
        if not is_int_type(v.type):
            raise Exception("Expected integer value")
        w = v.type.width
        if w < INT64_WIDTH:
            # i1 uses zero-extension to print as 1/0, not -1/0
            if w == 1:
                arg = self._llvm.ir_builder.zext(v, self._llvm.INT64_TYPE)
            else:
                arg = self._llvm.ir_builder.sext(v, self._llvm.INT64_TYPE)
            return arg, "%lld"
        if w == INT64_WIDTH:
            return v, "%lld"
        raise Exception(
            "Casting integers wider than 64 bits to string is not supported"
        )

    def _create_malloc_decl(self) -> ir.Function:
        """
        title: Declare malloc.
        returns:
          type: ir.Function
        """
        return self.require_runtime_symbol("libc", "malloc")

    def _snprintf_heap(
        self, fmt_gv: ir.GlobalVariable, args: list[ir.Value]
    ) -> ir.Value:
        """
        title: Format into a heap buffer and return i8* (char*).
        parameters:
          fmt_gv:
            type: ir.GlobalVariable
          args:
            type: list[ir.Value]
        returns:
          type: ir.Value
        """
        snprintf = self._create_snprintf_decl()
        malloc = self._create_malloc_decl()

        zero_size = ir.Constant(self._llvm.SIZE_T_TYPE, 0)
        null_ptr = ir.Constant(self._llvm.INT8_TYPE.as_pointer(), None)

        fmt_ptr = self._llvm.ir_builder.gep(
            fmt_gv,
            [
                ir.Constant(self._llvm.INT32_TYPE, 0),
                ir.Constant(self._llvm.INT32_TYPE, 0),
            ],
            inbounds=True,
        )

        needed_i32 = self._llvm.ir_builder.call(
            snprintf, [null_ptr, zero_size, fmt_ptr, *args]
        )

        # Guard: snprintf returns negative on error; clamp to 1
        zero_i32 = ir.Constant(self._llvm.INT32_TYPE, 0)
        min_needed = self._llvm.ir_builder.select(
            self._llvm.ir_builder.icmp_signed("<", needed_i32, zero_i32),
            ir.Constant(self._llvm.INT32_TYPE, 1),
            needed_i32,
        )
        need_plus_1 = self._llvm.ir_builder.add(
            min_needed, ir.Constant(self._llvm.INT32_TYPE, 1)
        )
        need_szt = self._llvm.ir_builder.zext(
            need_plus_1, self._llvm.SIZE_T_TYPE
        )

        # allocate and print
        mem = self._llvm.ir_builder.call(malloc, [need_szt])
        _ = self._llvm.ir_builder.call(
            snprintf, [mem, need_szt, fmt_ptr, *args]
        )
        return mem

    def _create_snprintf_decl(self) -> ir.Function:
        """
        title: Declare (or return) the external snprintf (varargs).
        returns:
          type: ir.Function
        """
        return self.require_runtime_symbol("libc", "snprintf")

    def _get_or_create_format_global(self, fmt: str) -> ir.GlobalVariable:
        """
        title: Create a constant global format string.
        parameters:
          fmt:
            type: str
        returns:
          type: ir.GlobalVariable
        """
        # safe unique name for the format
        name = f"fmt_{abs(hash(fmt))}"
        if name in self._llvm.module.globals:
            gv = self._llvm.module.get_global(name)
            # compute pointer (gep) at use time; return gv (array) here
            return gv

        data = bytearray(fmt + "\0", "utf8")
        arr_ty = ir.ArrayType(self._llvm.INT8_TYPE, len(data))
        gv = ir.GlobalVariable(self._llvm.module, arr_ty, name=name)
        gv.linkage = "internal"
        gv.global_constant = True
        gv.initializer = ir.Constant(arr_ty, data)
        return gv

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: system.Cast) -> None:
        """
        title: Translate Cast expression to LLVM-IR.
        parameters:
          node:
            type: system.Cast
        """
        self.visit(node.value)
        value = self.result_stack.pop()
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

            # floats / doubles / half -> print as double with fixed format
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

        else:
            raise Exception(
                f"Unsupported cast from {value.type} to {target_type}"
            )

        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: system.PrintExpr) -> None:
        """
        title: Generate LLVM IR for a PrintExpr node.
        parameters:
          node:
            type: system.PrintExpr
        """
        self.visit(node.message)
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
            float_arg: ir.Value
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

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: irx_arrow.ArrowInt32ArrayLength) -> None:
        """
        title: Lower the internal Arrow int32 array length helper.
        parameters:
          node:
            type: irx_arrow.ArrowInt32ArrayLength
        """
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
            self.visit(item)
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

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.Identifier) -> None:
        """
        title: Translate ASTx Identifier to LLVM-IR.
        parameters:
          node:
            type: astx.Identifier
        """
        expr_var = self.named_values.get(node.name)

        if not expr_var:
            raise Exception(f"Unknown variable name: {node.name}")

        result = self._llvm.ir_builder.load(expr_var, node.name)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.VariableDeclaration) -> None:
        """
        title: Translate ASTx VariableDeclaration to LLVM-IR.
        parameters:
          node:
            type: astx.VariableDeclaration
        """
        if self.named_values.get(node.name):
            raise Exception(f"Identifier already declared: {node.name}")

        type_str = node.type_.__class__.__name__.lower()

        # Emit the initializer
        if node.value is not None:
            self.visit(node.value)
            init_val = self.result_stack.pop()
            if init_val is None:
                raise Exception("Initializer code generation failed.")

            if type_str == "string":
                alloca = self.create_entry_block_alloca(
                    node.name, "stringascii"
                )
                self._llvm.ir_builder.store(init_val, alloca)
            else:
                alloca = self.create_entry_block_alloca(node.name, type_str)
                self._llvm.ir_builder.store(init_val, alloca)

        else:
            if type_str == "string":
                # For strings, create empty string
                empty_str_type = ir.ArrayType(self._llvm.INT8_TYPE, 1)
                empty_str_global = ir.GlobalVariable(
                    self._llvm.module,
                    empty_str_type,
                    name=f"empty_str_{node.name}",
                )
                empty_str_global.linkage = "internal"
                empty_str_global.global_constant = True
                empty_str_global.initializer = ir.Constant(
                    empty_str_type, bytearray(b"\0")
                )

                init_val = self._llvm.ir_builder.gep(
                    empty_str_global,
                    [
                        ir.Constant(ir.IntType(32), 0),
                        ir.Constant(ir.IntType(32), 0),
                    ],
                    inbounds=True,
                )
                alloca = self.create_entry_block_alloca(
                    node.name, "stringascii"
                )

            elif "float" in type_str:
                init_val = ir.Constant(self._llvm.get_data_type(type_str), 0.0)
                alloca = self.create_entry_block_alloca(node.name, type_str)

            else:
                # If not specified, use 0 as the initializer.
                init_val = ir.Constant(self._llvm.get_data_type(type_str), 0)
                alloca = self.create_entry_block_alloca(node.name, type_str)

            # Store the initial value.
            self._llvm.ir_builder.store(init_val, alloca)

        if node.mutability == astx.MutabilityKind.constant:
            self.const_vars.add(node.name)
        self.named_values[node.name] = alloca

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralInt16) -> None:
        """
        title: Translate ASTx LiteralInt16 to LLVM-IR.
        parameters:
          node:
            type: astx.LiteralInt16
        """
        result = ir.Constant(self._llvm.INT16_TYPE, node.value)
        self.result_stack.append(result)


@public
class LLVMLiteIR(Builder):
    """
    title: LLVM-IR transpiler and compiler.
    attributes:
      translator:
        type: LLVMLiteIRVisitor
    """

    def __init__(self) -> None:
        """
        title: Initialize LLVMIR.
        """
        super().__init__()
        self.translator: LLVMLiteIRVisitor = self._new_translator()

    def _new_translator(self) -> LLVMLiteIRVisitor:
        """
        title: Create a fresh translator for one compilation unit.
        returns:
          type: LLVMLiteIRVisitor
        """
        return LLVMLiteIRVisitor(
            active_runtime_features=set(self.runtime_feature_names)
        )

    def translate(self, expr: astx.AST) -> str:
        """
        title: Transpile ASTx to LLVM-IR with a fresh translator.
        parameters:
          expr:
            type: astx.AST
        returns:
          type: str
        """
        self.translator = self._new_translator()
        return self.translator.translate(expr)

    def build(self, node: astx.AST, output_file: str) -> None:
        """
        title: >-
          Transpile the ASTx to LLVM-IR and build it to an executable file.
        parameters:
          node:
            type: astx.AST
          output_file:
            type: str
        """
        result = self.translate(node)

        result_mod = llvm.parse_assembly(result)
        result_object = self.translator.target_machine.emit_object(result_mod)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.tmp_path = temp_dir
            file_path_o = Path(temp_dir) / "irx_module.o"

            with open(file_path_o, "wb") as f:
                f.write(result_object)

            self.output_file = output_file

            link_executable(
                primary_object=file_path_o,
                output_file=Path(self.output_file),
                artifacts=self.translator.runtime_features.native_artifacts(),
                linker_flags=self.translator.runtime_features.linker_flags(),
            )
        os.chmod(self.output_file, 0o755)

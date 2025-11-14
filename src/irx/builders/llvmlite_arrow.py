"""
LLVM-IR builder with Arrow C Data Interface (experimental).

This backend lowers literals to Arrow-compatible shapes using ONLY emitted
LLVM IR (no external C/C++ shims). For now, we model a scalar as an ArrowArray
of length 1 (C Data Interface).

"""

from __future__ import annotations

import tempfile

from typing import Any, Callable

import astx
import xh

from llvmlite import binding as llvm
from llvmlite import ir
from plum import dispatch
from public import public

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIRVisitor


class LLVMLiteArrowIRVisitor(LLVMLiteIRVisitor):
    """IR visitor that lowers literals to Arrow C Data Interface objects."""

    _arrow_array_ty: ir.IdentifiedStructType

    def __init__(self) -> None:
        super().__init__()
        self._init_arrow_types()

    # C Data Interface: ArrowArray
    #   struct ArrowArray {
    #     int64_t length;
    #     int64_t null_count;
    #     int64_t offset;
    #     int64_t n_buffers;
    #     int64_t n_children;
    #     const void** buffers;
    #     struct ArrowArray** children;
    #     void* dictionary;                // we model as i8*
    #     void (*release)(struct ArrowArray*);
    #     void* private_data;
    #   };
    def _init_arrow_types(self) -> None:
        ctx = ir.global_context
        self._arrow_array_ty = ctx.get_identified_type("struct.ArrowArray")

        i64 = ir.IntType(64)
        i8p = ir.IntType(8).as_pointer()

        arr_ptr = self._arrow_array_ty.as_pointer()
        # Function pointer type: void (*release)(ArrowArray*)
        release_fn_ty = ir.FunctionType(ir.VoidType(), [arr_ptr]).as_pointer()

        # buffers: i8** (const void**)
        buffers_ptr_ty = i8p.as_pointer()
        # children: ArrowArray**  (we won't use it yet; set to null)
        children_ptr_ty = arr_ptr.as_pointer()

        self._arrow_array_ty.set_body(
            i64,  # length
            i64,  # null_count
            i64,  # offset
            i64,  # n_buffers
            i64,  # n_children
            buffers_ptr_ty,  # buffers
            children_ptr_ty,  # children
            i8p,  # dictionary (opaque)
            release_fn_ty,  # release
            i8p,  # private_data
        )

    def _entry_alloca(self, ty: ir.Type, name: str) -> ir.Instruction:
        """Allocate in the function entry block (mem2reg-friendly)."""
        ib = self._llvm.ir_builder
        cur = ib.block
        ib.position_at_start(ib.function.entry_basic_block)
        slot = ib.alloca(ty, name=name)
        ib.position_at_end(cur)
        return slot

    @dispatch
    def visit(self, node: astx.AST) -> None:
        """Define the Generic visit method for AST node."""
        raise Exception("Not implemented.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralInt32) -> None:
        """
        Lower LiteralInt32 to an ArrowArray(length=1).

        Layout (C Data Interface):
          - length      = 1
          - null_count  = 0
          - offset      = 0
          - n_buffers   = 2 (validity bitmap, values)
          - n_children  = 0
          - buffers[0]  = &validity_byte (i8*, bit 0 set to 1)
          - buffers[1]  = &value_i32     (i8* to 4-byte i32 storage)
          - children    = null
          - dictionary  = null
          - release     = null (stack lifetime only)
          - private_data= null
        """
        ib = self._llvm.ir_builder
        i8 = ir.IntType(8)
        i8p = i8.as_pointer()
        i32 = self._llvm.INT32_TYPE
        i64 = ir.IntType(64)

        arr_ptr = self._entry_alloca(
            self._arrow_array_ty, name="arrow.i32.scalar"
        )

        # Allocate buffers array [2 x i8*] in entry block.
        buffers_arr_ty = ir.ArrayType(i8p, 2)
        buffers_slot = self._entry_alloca(buffers_arr_ty, name="arrow.buffers")

        # Allocate and initialize validity byte (bitmap) on stack:
        #   bit 0 = 1 (valid)
        valid_slot = self._entry_alloca(i8, name="arrow.valid")
        ib.store(ir.Constant(i8, 1), valid_slot)  # 0000_0001

        # Allocate and initialize 4-byte value on stack
        value_slot = self._entry_alloca(i32, name="arrow.i32.value")
        ib.store(ir.Constant(i32, node.value), value_slot)

        # Compute i8* pointers for buffers[0] and buffers[1]
        valid_i8p = ib.bitcast(valid_slot, i8p, name="valid_i8p")
        value_i8p = ib.bitcast(value_slot, i8p, name="value_i8p")

        # Fill buffers array
        i32_ty = ir.IntType(32)
        buf0_ptr = ib.gep(
            buffers_slot,
            [ir.Constant(i32_ty, 0), ir.Constant(i32_ty, 0)],
            inbounds=True,
        )
        buf1_ptr = ib.gep(
            buffers_slot,
            [ir.Constant(i32_ty, 0), ir.Constant(i32_ty, 1)],
            inbounds=True,
        )
        ib.store(valid_i8p, buf0_ptr)
        ib.store(value_i8p, buf1_ptr)

        # Pointer-to-first element: i8**  (const void**)
        buffers_i8pp = ib.gep(
            buffers_slot,
            [ir.Constant(i32_ty, 0), ir.Constant(i32_ty, 0)],
            inbounds=True,
        )

        # Set ArrowArray fields
        # GEP helpers for fields [0..9]
        def fld(idx: int):
            return ib.gep(
                arr_ptr,
                [ir.Constant(i32_ty, 0), ir.Constant(i32_ty, idx)],
                inbounds=True,
            )

        ib.store(ir.Constant(i64, 1), fld(0))  # length
        ib.store(ir.Constant(i64, 0), fld(1))  # null_count
        ib.store(ir.Constant(i64, 0), fld(2))  # offset
        ib.store(ir.Constant(i64, 2), fld(3))  # n_buffers
        ib.store(ir.Constant(i64, 0), fld(4))  # n_children
        ib.store(buffers_i8pp, fld(5))  # buffers
        # children = null
        children_ty = self._arrow_array_ty.as_pointer().as_pointer()
        ib.store(ir.Constant(children_ty, None), fld(6))
        # dictionary = null
        ib.store(ir.Constant(i8p, None), fld(7))
        # release = null (stack lifetime; do not export)
        rel_fn_ptr_ty = ir.FunctionType(
            ir.VoidType(), [self._arrow_array_ty.as_pointer()]
        ).as_pointer()
        ib.store(ir.Constant(rel_fn_ptr_ty, None), fld(8))
        # private_data = null
        ib.store(ir.Constant(i8p, None), fld(9))

        # Result: %ArrowArray* (stack-allocated)
        self.result_stack.append(arr_ptr)


@public
class LLVMLiteArrowIR(Builder):
    """LLVM-IR transpiler that uses LLVMLiteArrowIRVisitor."""

    def __init__(self) -> None:
        super().__init__()
        self.translator: LLVMLiteArrowIRVisitor = LLVMLiteArrowIRVisitor()
        self.output_file = ""
        self.tmp_path = ""

    def build(self, node: astx.AST, output_file: str) -> None:
        """
        Transpile ASTx to LLVM-IR and build an executable via clang.

        NOTE:
          - no extra libs linked.
        """
        # Fresh visitor per build (mirrors your LLVMLiteIR)
        self.translator = LLVMLiteArrowIRVisitor()
        ir_text = self.translator.translate(node)

        mod = llvm.parse_assembly(ir_text)
        obj = self.translator.target_machine.emit_object(mod)

        with tempfile.NamedTemporaryFile(suffix="", delete=False) as temp_file:
            self.tmp_path = temp_file.name

        obj_path = f"{self.tmp_path}.o"
        with open(obj_path, "wb") as f:
            f.write(obj)

        self.output_file = output_file

        clang: Callable[..., Any] = xh.clang
        clang(obj_path, "-o", self.output_file)

        import os

        os.chmod(self.output_file, 0o755)

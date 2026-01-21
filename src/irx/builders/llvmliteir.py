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
        self.named_values = {}
        self.function_protos = {}
        self.result_stack = []
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

        # SINGLE STRING REPRESENTATION
        self._llvm.STRING_PTR_TYPE = ir.IntType(8).as_pointer()

        self._llvm.TIMESTAMP_TYPE = ir.LiteralStructType(
            [self._llvm.INT32_TYPE] * 7
        )

        self._llvm.SIZE_T_TYPE = ir.IntType(64)

    

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

        fn = ir.Function(
            self._llvm.module,
            ir.FunctionType(
                self._llvm.BOOLEAN_TYPE,
                [self._llvm.STRING_PTR_TYPE, self._llvm.STRING_PTR_TYPE],
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

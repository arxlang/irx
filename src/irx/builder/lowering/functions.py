# mypy: disable-error-code=no-redef

"""
title: Function visitor mixins for llvmliteir.
"""

from llvmlite import ir

from irx import astx
from irx.builder.core import (
    VisitorCore,
    semantic_function_key,
    semantic_symbol_key,
)
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builder.types import is_int_type
from irx.typecheck import typechecked


@typechecked
class FunctionVisitorMixin(VisitorMixinBase):
    @VisitorCore.visit.dispatch
    def visit(self, node: astx.FunctionCall) -> None:
        """
        title: Visit FunctionCall nodes.
        parameters:
          node:
            type: astx.FunctionCall
        """
        callee_f = self.get_function(semantic_function_key(node, node.fn))
        if not callee_f:
            raise Exception("Unknown function referenced")

        if len(callee_f.args) != len(node.args):
            raise Exception("codegen: Incorrect # arguments passed.")

        llvm_args = []
        for arg in node.args:
            self.visit_child(arg)
            llvm_arg = safe_pop(self.result_stack)
            if llvm_arg is None:
                raise Exception("codegen: Invalid callee argument.")
            llvm_args.append(llvm_arg)

        result = self._llvm.ir_builder.call(callee_f, llvm_args, "calltmp")
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.FunctionDef) -> None:
        """
        title: Visit FunctionDef nodes.
        parameters:
          node:
            type: astx.FunctionDef
        """
        proto = node.prototype
        function_key = semantic_function_key(proto, proto.name)
        self.function_protos[function_key] = proto
        fn = self.get_function(function_key)
        if not fn:
            raise Exception("Invalid function.")
        if function_key in self._emitted_function_bodies:
            self.result_stack.append(fn)
            return

        basic_block = fn.append_basic_block("entry")
        self._llvm.ir_builder = ir.IRBuilder(basic_block)

        for idx, llvm_arg in enumerate(fn.args):
            arg_ast = proto.args.nodes[idx]
            type_str = arg_ast.type_.__class__.__name__.lower()
            arg_type = self._llvm.get_data_type(type_str)
            symbol_key = semantic_symbol_key(arg_ast, llvm_arg.name)
            alloca = self._llvm.ir_builder.alloca(arg_type, name=llvm_arg.name)
            self._llvm.ir_builder.store(llvm_arg, alloca)
            self.named_values[symbol_key] = alloca

        self.visit_child(node.body)
        if not self._llvm.ir_builder.block.is_terminated:
            return_type = fn.function_type.return_type
            if isinstance(return_type, ir.VoidType):
                self._llvm.ir_builder.ret_void()
            else:
                raise SyntaxError(
                    f"Function '{proto.name}' with return type "
                    f"'{return_type}' is missing a return statement"
                )

        self._emitted_function_bodies.add(function_key)
        self.result_stack.append(fn)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.FunctionPrototype) -> None:
        """
        title: Visit FunctionPrototype nodes.
        parameters:
          node:
            type: astx.FunctionPrototype
        """
        args_type = []
        for arg in node.args.nodes:
            type_str = arg.type_.__class__.__name__.lower()
            args_type.append(self._llvm.get_data_type(type_str))

        return_type = self._llvm.get_data_type(
            node.return_type.__class__.__name__.lower()
        )
        fn_type = ir.FunctionType(return_type, args_type, False)
        function_key = semantic_function_key(node, node.name)
        existing = self.llvm_functions_by_symbol_id.get(function_key)
        if existing is not None:
            self.result_stack.append(existing)
            return

        llvm_name = self.llvm_function_name_for_node(node, node.name)
        fn = ir.Function(self._llvm.module, fn_type, llvm_name)
        self.function_protos[function_key] = node
        self.llvm_functions_by_symbol_id[function_key] = fn

        for idx, llvm_arg in enumerate(fn.args):
            llvm_arg.name = node.args.nodes[idx].name

        self.result_stack.append(fn)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.FunctionReturn) -> None:
        """
        title: Visit FunctionReturn nodes.
        parameters:
          node:
            type: astx.FunctionReturn
        """
        if node.value is not None:
            self.visit_child(node.value)
            retval = safe_pop(self.result_stack)
        else:
            retval = None

        if retval is not None:
            fn_return_type = (
                self._llvm.ir_builder.function.function_type.return_type
            )
            if is_int_type(fn_return_type) and fn_return_type.width == 1:
                if is_int_type(retval.type) and retval.type.width != 1:
                    retval = self._llvm.ir_builder.trunc(retval, ir.IntType(1))
            self._llvm.ir_builder.ret(retval)
            return

        self._llvm.ir_builder.ret_void()

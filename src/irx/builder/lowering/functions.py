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
        resolved_function = getattr(
            getattr(node, "semantic", None),
            "resolved_function",
            None,
        )
        param_types = (
            [param.type_ for param in resolved_function.args]
            if resolved_function is not None
            else [None] * len(node.args)
        )
        for arg, param_type in zip(node.args, param_types):
            self.visit_child(arg)
            llvm_arg = safe_pop(self.result_stack)
            if llvm_arg is None:
                raise Exception("codegen: Invalid callee argument.")
            llvm_arg = self._cast_ast_value(
                llvm_arg,
                source_type=self._resolved_ast_type(arg),
                target_type=param_type,
            )
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
        previous_return_type = self._current_function_return_type
        self._current_function_return_type = proto.return_type

        try:
            for idx, llvm_arg in enumerate(fn.args):
                arg_ast = proto.args.nodes[idx]
                symbol_key = semantic_symbol_key(arg_ast, llvm_arg.name)
                arg_type = self._llvm_type_for_ast_type(arg_ast.type_)
                if arg_type is None:
                    raise Exception(
                        "codegen: Unknown LLVM type for function argument "
                        f"'{llvm_arg.name}'."
                    )
                alloca = self._llvm.ir_builder.alloca(
                    arg_type,
                    name=llvm_arg.name,
                )
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
        finally:
            self._current_function_return_type = previous_return_type

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
            llvm_type = self._llvm_type_for_ast_type(arg.type_)
            if llvm_type is None:
                raise Exception(
                    "codegen: Unknown LLVM type for function argument "
                    f"'{arg.name}'."
                )
            args_type.append(llvm_type)

        return_type = self._llvm_type_for_ast_type(node.return_type)
        if return_type is None:
            raise Exception(
                "codegen: Unknown LLVM return type for function "
                f"'{node.name}'."
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
            retval = self._cast_ast_value(
                retval,
                source_type=self._resolved_ast_type(node.value),
                target_type=self._current_function_return_type,
            )
            fn_return_type = (
                self._llvm.ir_builder.function.function_type.return_type
            )
            if is_int_type(fn_return_type) and fn_return_type.width == 1:
                if is_int_type(retval.type) and retval.type.width != 1:
                    retval = self._llvm.ir_builder.trunc(retval, ir.IntType(1))
            self._llvm.ir_builder.ret(retval)
            return

        self._llvm.ir_builder.ret_void()

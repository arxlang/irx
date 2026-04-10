# mypy: disable-error-code=no-redef

"""
title: Function visitor mixins for llvmliteir.
"""

from typing import cast

from llvmlite import ir

from irx import astx
from irx.analysis.resolved_nodes import (
    CallingConvention,
    CallResolution,
    FunctionSignature,
    ReturnResolution,
    SemanticFunction,
)
from irx.builder.core import (
    VisitorCore,
    semantic_symbol_key,
)
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builder.types import is_int_type
from irx.typecheck import typechecked


@typechecked
class FunctionVisitorMixin(VisitorMixinBase):
    def _semantic_function(
        self,
        node: astx.AST,
        *,
        label: str,
    ) -> SemanticFunction:
        """
        title: Return the resolved semantic function for one node.
        parameters:
          node:
            type: astx.AST
          label:
            type: str
        returns:
          type: SemanticFunction
        """
        semantic = getattr(node, "semantic", None)
        function = getattr(semantic, "resolved_function", None)
        if function is None:
            raise Exception(f"codegen: Missing semantic function for {label}.")
        return cast(SemanticFunction, function)

    def _semantic_signature(
        self,
        node: astx.AST,
        *,
        label: str,
    ) -> FunctionSignature:
        """
        title: Return the resolved semantic signature for one node.
        parameters:
          node:
            type: astx.AST
          label:
            type: str
        returns:
          type: FunctionSignature
        """
        return self._semantic_function(node, label=label).signature

    def _semantic_call_resolution(
        self,
        node: astx.FunctionCall,
    ) -> CallResolution:
        """
        title: Return the resolved semantic call metadata for one call.
        parameters:
          node:
            type: astx.FunctionCall
        returns:
          type: CallResolution
        """
        semantic = getattr(node, "semantic", None)
        resolution = getattr(semantic, "resolved_call", None)
        if resolution is None:
            raise Exception("codegen: Missing semantic call resolution.")
        return cast(CallResolution, resolution)

    def _semantic_return_resolution(
        self,
        node: astx.FunctionReturn,
    ) -> ReturnResolution:
        """
        title: Return the resolved semantic return metadata for one return.
        parameters:
          node:
            type: astx.FunctionReturn
        returns:
          type: ReturnResolution
        """
        semantic = getattr(node, "semantic", None)
        resolution = getattr(semantic, "resolved_return", None)
        if resolution is None:
            raise Exception("codegen: Missing semantic return resolution.")
        return cast(ReturnResolution, resolution)

    def _apply_calling_convention(
        self,
        signature: FunctionSignature,
    ) -> None:
        """
        title: Preserve semantic calling-convention intent in lowering.
        parameters:
          signature:
            type: FunctionSignature
        """
        if signature.calling_convention in {
            CallingConvention.IRX_DEFAULT,
            CallingConvention.C,
        }:
            return
        raise Exception(
            "codegen: Unsupported semantic calling convention "
            f"'{signature.calling_convention.value}'."
        )

    def _llvm_function_type_for_signature(
        self,
        signature: FunctionSignature,
    ) -> ir.FunctionType:
        """
        title: Return the LLVM function type for one semantic signature.
        parameters:
          signature:
            type: FunctionSignature
        returns:
          type: ir.FunctionType
        """
        args_type: list[ir.Type] = []
        for parameter in signature.parameters:
            llvm_type = self._llvm_type_for_ast_type(parameter.type_)
            if llvm_type is None:
                raise Exception(
                    "codegen: Unknown LLVM type for function argument "
                    f"'{parameter.name}'."
                )
            args_type.append(llvm_type)

        return_type = self._llvm_type_for_ast_type(signature.return_type)
        if return_type is None:
            raise Exception(
                "codegen: Unknown LLVM return type for function "
                f"'{signature.name}'."
            )
        self._apply_calling_convention(signature)
        return ir.FunctionType(
            return_type,
            args_type,
            signature.is_variadic,
        )

    def _declare_semantic_function(
        self,
        function: SemanticFunction,
    ) -> ir.Function:
        """
        title: Declare or reuse one LLVM function from semantic metadata.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: ir.Function
        """
        function_key = function.symbol_id
        existing = self.llvm_functions_by_symbol_id.get(function_key)
        if existing is not None:
            return existing

        signature = function.signature
        fn_type = self._llvm_function_type_for_signature(signature)
        llvm_name = self.llvm_function_name_for_node(
            function.prototype,
            function.name,
        )
        global_value = self._llvm.module.globals.get(llvm_name)
        if global_value is not None:
            if not isinstance(global_value, ir.Function):
                raise Exception(
                    f"codegen: Global '{llvm_name}' is not a function."
                )
            if global_value.function_type != fn_type:
                raise Exception(
                    f"codegen: Function '{llvm_name}' already exists with a "
                    "different signature."
                )
            fn = global_value
        else:
            fn = ir.Function(self._llvm.module, fn_type, llvm_name)
            if signature.is_extern or function.definition is None:
                fn.linkage = "external"

        for idx, llvm_arg in enumerate(fn.args):
            llvm_arg.name = function.args[idx].name

        self.function_protos[function_key] = function.prototype
        self.llvm_functions_by_symbol_id[function_key] = fn
        return fn

    def _lower_call_arguments(
        self,
        node: astx.FunctionCall,
        resolution: CallResolution,
    ) -> list[ir.Value]:
        """
        title: Lower one semantically validated call argument list.
        parameters:
          node:
            type: astx.FunctionCall
          resolution:
            type: CallResolution
        returns:
          type: list[ir.Value]
        """
        llvm_args: list[ir.Value] = []
        for index, arg in enumerate(node.args):
            self.visit_child(arg)
            llvm_arg = safe_pop(self.result_stack)
            if llvm_arg is None or not isinstance(llvm_arg, ir.Value):
                raise Exception("codegen: Invalid callee argument.")
            target_type = (
                resolution.resolved_argument_types[index]
                if index < len(resolution.resolved_argument_types)
                else None
            )
            llvm_args.append(
                self._cast_ast_value(
                    llvm_arg,
                    source_type=self._resolved_ast_type(arg),
                    target_type=target_type,
                )
            )
        return llvm_args

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.FunctionCall) -> None:
        """
        title: Visit FunctionCall nodes.
        parameters:
          node:
            type: astx.FunctionCall
        """
        resolution = self._semantic_call_resolution(node)
        callee_f = self._declare_semantic_function(resolution.callee.function)
        llvm_args = self._lower_call_arguments(node, resolution)
        self._apply_calling_convention(resolution.signature)
        if isinstance(callee_f.function_type.return_type, ir.VoidType):
            self._llvm.ir_builder.call(callee_f, llvm_args)
            return
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
        function = self._semantic_function(
            node,
            label="function definition",
        )
        signature = function.signature
        function_key = function.symbol_id
        fn = self._declare_semantic_function(function)
        if function_key in self._emitted_function_bodies:
            self.result_stack.append(fn)
            return

        basic_block = fn.append_basic_block("entry")
        self._llvm.ir_builder = ir.IRBuilder(basic_block)
        previous_return_type = self._current_function_return_type
        previous_signature = self._current_function_signature
        self._current_function_return_type = signature.return_type
        self._current_function_signature = signature

        try:
            for idx, llvm_arg in enumerate(fn.args):
                arg_symbol = function.args[idx]
                arg_type = self._llvm_type_for_ast_type(
                    signature.parameters[idx].type_
                )
                if arg_type is None:
                    raise Exception(
                        "codegen: Unknown LLVM type for function argument "
                        f"'{arg_symbol.name}'."
                    )
                alloca = self._llvm.ir_builder.alloca(
                    arg_type,
                    name=arg_symbol.name,
                )
                self._llvm.ir_builder.store(llvm_arg, alloca)
                symbol_key = semantic_symbol_key(
                    function.prototype.args.nodes[idx],
                    arg_symbol.symbol_id,
                )
                self.named_values[symbol_key] = alloca

            self.visit_child(node.body)
            if not self._llvm.ir_builder.block.is_terminated:
                return_type = fn.function_type.return_type
                if isinstance(return_type, ir.VoidType):
                    self._llvm.ir_builder.ret_void()
                else:
                    raise SyntaxError(
                        f"Function '{function.name}' with return type "
                        f"'{return_type}' is missing a return statement"
                    )
        finally:
            self._current_function_return_type = previous_return_type
            self._current_function_signature = previous_signature

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
        function = self._semantic_function(
            node,
            label="function prototype",
        )
        fn = self._declare_semantic_function(function)
        self.result_stack.append(fn)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.FunctionReturn) -> None:
        """
        title: Visit FunctionReturn nodes.
        parameters:
          node:
            type: astx.FunctionReturn
        """
        return_resolution = self._semantic_return_resolution(node)
        if return_resolution.returns_void:
            self._llvm.ir_builder.ret_void()
            return

        if node.value is not None:
            self.visit_child(node.value)
            retval = safe_pop(self.result_stack)
        else:
            retval = None

        if retval is None or not isinstance(retval, ir.Value):
            raise Exception("codegen: Invalid return value.")

        retval = self._cast_ast_value(
            retval,
            source_type=self._resolved_ast_type(node.value),
            target_type=return_resolution.expected_type,
        )
        fn_return_type = (
            self._llvm.ir_builder.function.function_type.return_type
        )
        if is_int_type(fn_return_type) and fn_return_type.width == 1:
            if is_int_type(retval.type) and retval.type.width != 1:
                retval = self._llvm.ir_builder.trunc(retval, ir.IntType(1))
        self._llvm.ir_builder.ret(retval)

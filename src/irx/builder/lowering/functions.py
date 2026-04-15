# mypy: disable-error-code=no-redef

"""
title: Function visitor mixins for llvmliteir.
"""

from typing import Sequence, cast

from llvmlite import ir

from irx import astx
from irx.analysis.resolved_nodes import (
    CallingConvention,
    CallResolution,
    ClassHeaderFieldKind,
    FunctionSignature,
    MethodDispatchKind,
    ResolvedMethodCall,
    ResolvedMethodRuntimeCandidate,
    ResolvedMethodRuntimeCase,
    ReturnResolution,
    SemanticFunction,
)
from irx.analysis.types import display_type_name
from irx.builder.core import (
    VisitorCore,
    semantic_symbol_key,
)
from irx.builder.diagnostics import (
    raise_lowering_error,
    raise_lowering_internal_error,
    require_lowered_value,
    require_semantic_metadata,
)
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builder.types import is_int_type
from irx.diagnostics import DiagnosticCodes
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
        return require_semantic_metadata(
            cast(SemanticFunction | None, function),
            node=node,
            metadata="resolved_function",
            context=label,
        )

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
        return require_semantic_metadata(
            cast(CallResolution | None, resolution),
            node=node,
            metadata="resolved_call",
            context=f"call to '{node.fn}'",
        )

    def _semantic_method_call(
        self,
        node: astx.BaseMethodCall | astx.MethodCall | astx.StaticMethodCall,
    ) -> ResolvedMethodCall:
        """
        title: Return the resolved semantic method-call metadata for one call.
        parameters:
          node:
            type: astx.BaseMethodCall | astx.MethodCall | astx.StaticMethodCall
        returns:
          type: ResolvedMethodCall
        """
        semantic = getattr(node, "semantic", None)
        resolution = getattr(semantic, "resolved_method_call", None)
        return require_semantic_metadata(
            cast(ResolvedMethodCall | None, resolution),
            node=node,
            metadata="resolved_method_call",
            context="method call lowering",
        )

    def _lower_explicit_call_arguments(
        self,
        *,
        args: Sequence[astx.AST],
        resolution: CallResolution,
        label: str,
    ) -> list[ir.Value]:
        """
        title: Lower one semantically validated explicit call argument list.
        parameters:
          args:
            type: Sequence[astx.AST]
          resolution:
            type: CallResolution
          label:
            type: str
        returns:
          type: list[ir.Value]
        """
        llvm_args: list[ir.Value] = []
        for index, arg in enumerate(args):
            self.visit_child(arg)
            llvm_arg = require_lowered_value(
                safe_pop(self.result_stack),
                node=arg,
                context=f"argument {index + 1} of {label}",
            )
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

    def _indirect_method_callee(
        self,
        *,
        node: astx.MethodCall,
        method_resolution: ResolvedMethodCall,
        receiver_value: ir.Value,
    ) -> ir.Value:
        """
        title: Lower one dispatch-table lookup for an instance method.
        parameters:
          node:
            type: astx.MethodCall
          method_resolution:
            type: ResolvedMethodCall
          receiver_value:
            type: ir.Value
        returns:
          type: ir.Value
        """
        receiver_class = method_resolution.receiver_class
        if receiver_class is None or receiver_class.layout is None:
            raise_lowering_internal_error(
                "method call is missing receiver class layout metadata",
                node=node,
            )
        dispatch_header = next(
            (
                field
                for field in receiver_class.layout.header_fields
                if field.kind is ClassHeaderFieldKind.DISPATCH_TABLE
            ),
            None,
        )
        if dispatch_header is None or method_resolution.slot_index is None:
            raise_lowering_internal_error(
                "method call is missing dispatch slot metadata",
                node=node,
            )
        dispatch_addr = self._llvm.ir_builder.gep(
            receiver_value,
            [
                ir.Constant(self._llvm.INT32_TYPE, 0),
                ir.Constant(
                    self._llvm.INT32_TYPE,
                    dispatch_header.storage_index,
                ),
            ],
            inbounds=True,
            name=f"{method_resolution.member.name}_dispatch_addr",
        )
        dispatch_raw = self._llvm.ir_builder.load(
            dispatch_addr,
            f"{method_resolution.member.name}_dispatch",
        )
        dispatch_type = ir.ArrayType(
            self._llvm.OPAQUE_POINTER_TYPE,
            receiver_class.layout.dispatch_table_size,
        )
        dispatch_ptr = self._llvm.ir_builder.bitcast(
            dispatch_raw,
            dispatch_type.as_pointer(),
            name=f"{method_resolution.member.name}_dispatch_ptr",
        )
        callee_addr = self._llvm.ir_builder.gep(
            dispatch_ptr,
            [
                ir.Constant(self._llvm.INT32_TYPE, 0),
                ir.Constant(
                    self._llvm.INT32_TYPE,
                    method_resolution.slot_index,
                ),
            ],
            inbounds=True,
            name=f"{method_resolution.member.name}_slot",
        )
        callee_raw = self._llvm.ir_builder.load(
            callee_addr,
            f"{method_resolution.member.name}_raw",
        )
        function_ptr_type = self._llvm_function_type_for_signature(
            method_resolution.function.signature
        ).as_pointer()
        return self._llvm.ir_builder.bitcast(
            callee_raw,
            function_ptr_type,
            name=f"{method_resolution.member.name}_callee",
        )

    def _multimethod_dispatcher_name(
        self,
        method_resolution: ResolvedMethodCall,
    ) -> str:
        """
        title: Return one emitted multimethod dispatcher name.
        parameters:
          method_resolution:
            type: ResolvedMethodCall
        returns:
          type: str
        """
        if method_resolution.dispatcher_symbol_name is None:
            raise_lowering_internal_error(
                "multimethod call is missing dispatcher metadata",
                node=None,
            )
        return method_resolution.dispatcher_symbol_name

    def _multimethod_signature_source_types(
        self,
        node: astx.BaseMethodCall | astx.MethodCall | astx.StaticMethodCall,
        method_resolution: ResolvedMethodCall,
    ) -> tuple[astx.DataType | None, list[astx.DataType | None]]:
        """
        title: Return dispatcher source types for one multimethod call.
        parameters:
          node:
            type: astx.BaseMethodCall | astx.MethodCall | astx.StaticMethodCall
          method_resolution:
            type: ResolvedMethodCall
        returns:
          type: tuple[astx.DataType | None, list[astx.DataType | None]]
        """
        parameters = list(method_resolution.function.signature.parameters)
        if isinstance(node, astx.StaticMethodCall):
            return None, [parameter.type_ for parameter in parameters]
        if not parameters:
            raise_lowering_internal_error(
                "multimethod receiver signature metadata is missing",
                node=node,
            )
        return parameters[0].type_, [
            parameter.type_ for parameter in parameters[1:]
        ]

    def _emit_multimethod_candidate_return(
        self,
        *,
        node: astx.BaseMethodCall | astx.MethodCall | astx.StaticMethodCall,
        candidate: ResolvedMethodRuntimeCandidate,
        receiver_value: ir.Value | None,
        receiver_source_type: astx.DataType | None,
        explicit_args: list[ir.Value],
        explicit_source_types: list[astx.DataType | None],
    ) -> None:
        """
        title: Emit one runtime multimethod candidate call and return.
        parameters:
          node:
            type: astx.BaseMethodCall | astx.MethodCall | astx.StaticMethodCall
          candidate:
            type: ResolvedMethodRuntimeCandidate
          receiver_value:
            type: ir.Value | None
          receiver_source_type:
            type: astx.DataType | None
          explicit_args:
            type: list[ir.Value]
          explicit_source_types:
            type: list[astx.DataType | None]
        """
        callee = self._declare_semantic_function(candidate.function)
        signature = candidate.function.signature
        llvm_args: list[ir.Value] = []
        parameter_offset = 0
        if receiver_value is not None:
            if not signature.parameters:
                raise_lowering_internal_error(
                    "runtime multimethod candidate is missing self metadata",
                    node=node,
                )
            llvm_args.append(
                self._cast_ast_value(
                    receiver_value,
                    source_type=receiver_source_type,
                    target_type=signature.parameters[0].type_,
                )
            )
            parameter_offset = 1
        for index, explicit_arg in enumerate(explicit_args):
            parameter_index = parameter_offset + index
            if parameter_index >= len(signature.parameters):
                raise_lowering_internal_error(
                    "runtime multimethod candidate arity is invalid",
                    node=node,
                )
            llvm_args.append(
                self._cast_ast_value(
                    explicit_arg,
                    source_type=explicit_source_types[index],
                    target_type=signature.parameters[parameter_index].type_,
                )
            )
        self._apply_calling_convention(signature)
        if isinstance(signature.return_type, astx.NoneType):
            self._llvm.ir_builder.call(callee, llvm_args)
            self._llvm.ir_builder.ret_void()
            return
        result = self._llvm.ir_builder.call(callee, llvm_args, "calltmp")
        self._llvm.ir_builder.ret(result)

    def _emit_multimethod_candidate_chain(
        self,
        *,
        node: astx.BaseMethodCall | astx.MethodCall | astx.StaticMethodCall,
        dispatcher: ir.Function,
        case: ResolvedMethodRuntimeCase,
        receiver_value: ir.Value | None,
        receiver_source_type: astx.DataType | None,
        explicit_args: list[ir.Value],
        explicit_source_types: list[astx.DataType | None],
        label_prefix: str,
    ) -> None:
        """
        title: Emit one ordered runtime multimethod candidate chain.
        parameters:
          node:
            type: astx.BaseMethodCall | astx.MethodCall | astx.StaticMethodCall
          dispatcher:
            type: ir.Function
          case:
            type: ResolvedMethodRuntimeCase
          receiver_value:
            type: ir.Value | None
          receiver_source_type:
            type: astx.DataType | None
          explicit_args:
            type: list[ir.Value]
          explicit_source_types:
            type: list[astx.DataType | None]
          label_prefix:
            type: str
        """
        argument_descriptors: list[ir.Value | None] = []
        for index, source_type in enumerate(explicit_source_types):
            if isinstance(source_type, astx.ClassType):
                argument_descriptors.append(
                    self._class_descriptor_from_value(
                        explicit_args[index],
                        value_type=source_type,
                        name_hint=f"{label_prefix}_arg{index}",
                    )
                )
            else:
                argument_descriptors.append(None)

        for index, candidate in enumerate(case.candidates):
            next_block = dispatcher.append_basic_block(
                f"{label_prefix}_next_{index}"
            )
            match_block = dispatcher.append_basic_block(
                f"{label_prefix}_match_{index}"
            )
            condition: ir.Value | None = None
            for arg_index, allowed_classes in enumerate(
                candidate.allowed_argument_classes
            ):
                if allowed_classes is None:
                    continue
                descriptor = argument_descriptors[arg_index]
                if descriptor is None:
                    raise_lowering_internal_error(
                        (
                            "runtime multimethod is missing argument "
                            "descriptor metadata"
                        ),
                        node=node,
                    )
                allowed_condition: ir.Value | None = None
                for allowed_class in allowed_classes:
                    equality = self._llvm.ir_builder.icmp_unsigned(
                        "==",
                        descriptor,
                        self._class_descriptor_global(allowed_class),
                        name=(
                            f"{label_prefix}_{candidate.member.name}"
                            f"_arg{arg_index}_{allowed_class.name}"
                        ),
                    )
                    if allowed_condition is None:
                        allowed_condition = equality
                    else:
                        allowed_condition = self._llvm.ir_builder.or_(
                            allowed_condition,
                            equality,
                            name=(
                                f"{label_prefix}_{candidate.member.name}"
                                f"_arg{arg_index}_allowed"
                            ),
                        )
                if allowed_condition is None:
                    raise_lowering_internal_error(
                        (
                            "runtime multimethod candidate produced "
                            "no descriptor checks"
                        ),
                        node=node,
                    )
                if condition is None:
                    condition = allowed_condition
                else:
                    condition = self._llvm.ir_builder.and_(
                        condition,
                        allowed_condition,
                        name=(f"{label_prefix}_{candidate.member.name}_match"),
                    )
            if condition is None:
                self._emit_multimethod_candidate_return(
                    node=node,
                    candidate=candidate,
                    receiver_value=receiver_value,
                    receiver_source_type=receiver_source_type,
                    explicit_args=explicit_args,
                    explicit_source_types=explicit_source_types,
                )
                return
            self._llvm.ir_builder.cbranch(condition, match_block, next_block)
            self._llvm.ir_builder = ir.IRBuilder(match_block)
            self._emit_multimethod_candidate_return(
                node=node,
                candidate=candidate,
                receiver_value=receiver_value,
                receiver_source_type=receiver_source_type,
                explicit_args=explicit_args,
                explicit_source_types=explicit_source_types,
            )
            self._llvm.ir_builder = ir.IRBuilder(next_block)
        self._llvm.ir_builder.unreachable()

    def _declare_multimethod_dispatcher(
        self,
        *,
        node: astx.BaseMethodCall | astx.MethodCall | astx.StaticMethodCall,
        method_resolution: ResolvedMethodCall,
    ) -> ir.Function:
        """
        title: Declare or emit one runtime multimethod dispatcher.
        parameters:
          node:
            type: astx.BaseMethodCall | astx.MethodCall | astx.StaticMethodCall
          method_resolution:
            type: ResolvedMethodCall
        returns:
          type: ir.Function
        """
        dispatcher_name = self._multimethod_dispatcher_name(method_resolution)
        existing = self._llvm.module.globals.get(dispatcher_name)
        if existing is not None:
            return cast(ir.Function, existing)
        dispatcher_type = self._llvm_function_type_for_signature(
            method_resolution.function.signature
        )
        dispatcher = ir.Function(
            self._llvm.module,
            dispatcher_type,
            name=dispatcher_name,
        )
        dispatcher.linkage = "internal"
        entry = dispatcher.append_basic_block("entry")
        previous_builder = self._llvm.ir_builder
        self._llvm.ir_builder = ir.IRBuilder(entry)
        try:
            receiver_source_type, explicit_source_types = (
                self._multimethod_signature_source_types(
                    node,
                    method_resolution,
                )
            )
            receiver_value: ir.Value | None = None
            explicit_args: list[ir.Value] = []
            if isinstance(node, astx.StaticMethodCall):
                explicit_args = list(dispatcher.args)
            else:
                receiver_value = dispatcher.args[0]
                explicit_args = list(dispatcher.args[1:])
            runtime_cases = method_resolution.runtime_cases
            if not runtime_cases:
                raise_lowering_internal_error(
                    "multimethod call is missing runtime cases",
                    node=node,
                )
            if isinstance(node, astx.MethodCall):
                receiver_descriptor = self._class_descriptor_from_value(
                    receiver_value,
                    value_type=receiver_source_type,
                    name_hint=f"{method_resolution.member.name}_receiver",
                )
                fallthrough = dispatcher.append_basic_block("receiver_fail")
                for index, case in enumerate(runtime_cases):
                    if case.receiver_class is None:
                        raise_lowering_internal_error(
                            (
                                "instance multimethod is missing "
                                "receiver case metadata"
                            ),
                            node=node,
                        )
                    match_block = dispatcher.append_basic_block(
                        f"receiver_case_{index}"
                    )
                    next_block = (
                        fallthrough
                        if index == len(runtime_cases) - 1
                        else dispatcher.append_basic_block(
                            f"receiver_case_next_{index}"
                        )
                    )
                    matches_receiver = self._llvm.ir_builder.icmp_unsigned(
                        "==",
                        receiver_descriptor,
                        self._class_descriptor_global(case.receiver_class),
                        name=f"receiver_case_{index}_match",
                    )
                    self._llvm.ir_builder.cbranch(
                        matches_receiver,
                        match_block,
                        next_block,
                    )
                    self._llvm.ir_builder = ir.IRBuilder(match_block)
                    self._emit_multimethod_candidate_chain(
                        node=node,
                        dispatcher=dispatcher,
                        case=case,
                        receiver_value=receiver_value,
                        receiver_source_type=receiver_source_type,
                        explicit_args=explicit_args,
                        explicit_source_types=explicit_source_types,
                        label_prefix=f"receiver_case_{index}",
                    )
                    if next_block is not fallthrough:
                        self._llvm.ir_builder = ir.IRBuilder(next_block)
                self._llvm.ir_builder = ir.IRBuilder(fallthrough)
                self._llvm.ir_builder.unreachable()
            else:
                self._emit_multimethod_candidate_chain(
                    node=node,
                    dispatcher=dispatcher,
                    case=runtime_cases[0],
                    receiver_value=receiver_value,
                    receiver_source_type=receiver_source_type,
                    explicit_args=explicit_args,
                    explicit_source_types=explicit_source_types,
                    label_prefix=f"{method_resolution.member.name}_dispatch",
                )
        finally:
            self._llvm.ir_builder = previous_builder
        return dispatcher

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
        return require_semantic_metadata(
            cast(ReturnResolution | None, resolution),
            node=node,
            metadata="resolved_return",
            context="return lowering",
        )

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
        raise_lowering_internal_error(
            "unsupported semantic calling convention "
            f"'{signature.calling_convention.value}'",
            node=None,
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
                raise_lowering_error(
                    "cannot lower parameter "
                    f"'{parameter.name}' of '{signature.name}' with type "
                    f"{display_type_name(parameter.type_)}",
                    code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
                )
            args_type.append(llvm_type)

        return_type = self._llvm_type_for_ast_type(signature.return_type)
        if return_type is None:
            raise_lowering_error(
                "cannot lower return type "
                f"{display_type_name(signature.return_type)} for "
                f"'{signature.name}'",
                code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
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
        fn: ir.Function | None = None
        declared_feature_name: str | None = None
        for feature_name in signature.required_runtime_features:
            if self.runtime_features.feature_declares_symbol(
                feature_name,
                signature.symbol_name,
            ):
                fn = self.require_runtime_symbol(
                    feature_name,
                    signature.symbol_name,
                )
                declared_feature_name = feature_name
                if fn.function_type != fn_type:
                    raise_lowering_error(
                        f"runtime feature '{feature_name}' declares symbol "
                        f"'{signature.symbol_name}' with an incompatible "
                        "LLVM signature",
                        code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
                        node=function.prototype,
                    )
                break

        for feature_name in signature.required_runtime_features:
            if feature_name == declared_feature_name:
                continue
            self.activate_runtime_feature(feature_name)

        if fn is None:
            global_value = self._llvm.module.globals.get(llvm_name)
            if global_value is not None:
                if not isinstance(global_value, ir.Function):
                    raise_lowering_internal_error(
                        f"global '{llvm_name}' is not a function",
                        node=function.prototype,
                    )
                if global_value.function_type != fn_type:
                    raise_lowering_error(
                        f"function '{llvm_name}' already exists with an "
                        "incompatible LLVM signature",
                        code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
                        node=function.prototype,
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
        return self._lower_explicit_call_arguments(
            args=list(node.args),
            resolution=resolution,
            label=f"call to '{node.fn}'",
        )

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
    def visit(self, node: astx.MethodCall) -> None:
        """
        title: Visit MethodCall nodes.
        parameters:
          node:
            type: astx.MethodCall
        """
        method_resolution = self._semantic_method_call(node)
        llvm_args = self._lower_explicit_call_arguments(
            args=list(node.args),
            resolution=method_resolution.call,
            label=f"method call '{method_resolution.member.name}'",
        )
        self.visit_child(node.receiver)
        receiver_value = require_lowered_value(
            safe_pop(self.result_stack),
            node=node.receiver,
            context=f"receiver for '{method_resolution.member.name}'",
        )
        receiver_parameter_type = (
            method_resolution.function.signature.parameters[0].type_
            if method_resolution.function.signature.parameters
            else None
        )
        lowered_receiver = self._cast_ast_value(
            receiver_value,
            source_type=self._resolved_ast_type(node.receiver),
            target_type=receiver_parameter_type,
        )
        lowered_args = [lowered_receiver, *llvm_args]
        callee: ir.Value
        if method_resolution.dispatch_kind is MethodDispatchKind.MULTIMETHOD:
            callee = self._declare_multimethod_dispatcher(
                node=node,
                method_resolution=method_resolution,
            )
        elif method_resolution.dispatch_kind is MethodDispatchKind.INDIRECT:
            callee = self._indirect_method_callee(
                node=node,
                method_resolution=method_resolution,
                receiver_value=receiver_value,
            )
        else:
            callee = self._declare_semantic_function(
                method_resolution.function
            )
        self._apply_calling_convention(method_resolution.function.signature)
        if isinstance(
            method_resolution.function.signature.return_type,
            astx.NoneType,
        ):
            self._llvm.ir_builder.call(callee, lowered_args)
            return
        result = self._llvm.ir_builder.call(callee, lowered_args, "calltmp")
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.BaseMethodCall) -> None:
        """
        title: Visit BaseMethodCall nodes.
        parameters:
          node:
            type: astx.BaseMethodCall
        """
        method_resolution = self._semantic_method_call(node)
        llvm_args = self._lower_explicit_call_arguments(
            args=list(node.args),
            resolution=method_resolution.call,
            label=(
                f"base method call '{method_resolution.class_.name}."
                f"{method_resolution.member.name}'"
            ),
        )
        self.visit_child(node.receiver)
        receiver_value = require_lowered_value(
            safe_pop(self.result_stack),
            node=node.receiver,
            context=(
                f"receiver for '{method_resolution.class_.name}."
                f"{method_resolution.member.name}'"
            ),
        )
        receiver_parameter_type = (
            method_resolution.function.signature.parameters[0].type_
            if method_resolution.function.signature.parameters
            else None
        )
        lowered_receiver = self._cast_ast_value(
            receiver_value,
            source_type=self._resolved_ast_type(node.receiver),
            target_type=receiver_parameter_type,
        )
        lowered_args = [lowered_receiver, *llvm_args]
        if method_resolution.dispatch_kind is MethodDispatchKind.MULTIMETHOD:
            callee = self._declare_multimethod_dispatcher(
                node=node,
                method_resolution=method_resolution,
            )
        else:
            callee = self._declare_semantic_function(
                method_resolution.function
            )
        self._apply_calling_convention(method_resolution.function.signature)
        if isinstance(
            method_resolution.function.signature.return_type,
            astx.NoneType,
        ):
            self._llvm.ir_builder.call(callee, lowered_args)
            return
        result = self._llvm.ir_builder.call(callee, lowered_args, "calltmp")
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.StaticMethodCall) -> None:
        """
        title: Visit StaticMethodCall nodes.
        parameters:
          node:
            type: astx.StaticMethodCall
        """
        method_resolution = self._semantic_method_call(node)
        if method_resolution.dispatch_kind is MethodDispatchKind.MULTIMETHOD:
            callee = self._declare_multimethod_dispatcher(
                node=node,
                method_resolution=method_resolution,
            )
        else:
            callee = self._declare_semantic_function(
                method_resolution.function
            )
        llvm_args = self._lower_explicit_call_arguments(
            args=list(node.args),
            resolution=method_resolution.call,
            label=(
                f"static method call '{method_resolution.class_.name}."
                f"{method_resolution.member.name}'"
            ),
        )
        self._apply_calling_convention(method_resolution.function.signature)
        if isinstance(callee.function_type.return_type, ir.VoidType):
            self._llvm.ir_builder.call(callee, llvm_args)
            return
        result = self._llvm.ir_builder.call(callee, llvm_args, "calltmp")
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
            hidden_parameter_count = len(function.args) - len(
                function.prototype.args.nodes
            )
            for idx, llvm_arg in enumerate(fn.args):
                arg_symbol = function.args[idx]
                arg_type = self._llvm_type_for_ast_type(
                    signature.parameters[idx].type_
                )
                if arg_type is None:
                    parameter_type_name = display_type_name(
                        signature.parameters[idx].type_
                    )
                    parameter_node = (
                        function.prototype.args.nodes[
                            idx - hidden_parameter_count
                        ]
                        if idx >= hidden_parameter_count
                        else function.prototype
                    )
                    raise_lowering_error(
                        "cannot lower parameter "
                        f"'{arg_symbol.name}' of '{function.name}' with "
                        f"type {parameter_type_name}",
                        code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
                        node=parameter_node,
                    )
                alloca = self._llvm.ir_builder.alloca(
                    arg_type,
                    name=arg_symbol.name,
                )
                self._llvm.ir_builder.store(llvm_arg, alloca)
                if idx < hidden_parameter_count:
                    symbol_key = arg_symbol.symbol_id
                else:
                    symbol_key = semantic_symbol_key(
                        function.prototype.args.nodes[
                            idx - hidden_parameter_count
                        ],
                        arg_symbol.symbol_id,
                    )
                self.named_values[symbol_key] = alloca

            self.visit_child(node.body)
            if not self._llvm.ir_builder.block.is_terminated:
                return_type = fn.function_type.return_type
                if isinstance(return_type, ir.VoidType):
                    self._llvm.ir_builder.ret_void()
                else:
                    raise_lowering_internal_error(
                        f"function '{function.name}' reached lowering "
                        "without a terminating return",
                        node=node,
                        notes=(
                            "semantic analysis should reject reachable "
                            "non-void fallthrough before lowering",
                        ),
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
            retval = require_lowered_value(
                safe_pop(self.result_stack),
                node=node.value,
                context="return expression",
            )
        else:
            retval = None

        if retval is None:
            raise_lowering_internal_error(
                "return expression did not lower to a value",
                node=node,
            )

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

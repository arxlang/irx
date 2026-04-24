"""
title: Shared concrete core for llvmliteir visitors.
"""

from __future__ import annotations

import ctypes

from typing import Any, cast

from llvmlite import binding as llvm
from llvmlite import ir
from llvmlite.ir import DoubleType, FloatType, HalfType
from plum import dispatch
from public import private

from irx import astx
from irx.buffer import BUFFER_VIEW_TYPE_NAME

try:  # FP128 may not exist depending on llvmlite build.
    from llvmlite.ir import FP128Type
except ImportError:  # pragma: no cover - optional
    FP128Type = None

from irx.analysis import analyze, analyze_modules
from irx.analysis.module_interfaces import ImportResolver, ParsedModule
from irx.analysis.module_symbols import (
    mangle_class_name,
    mangle_function_name,
    mangle_namespace_name,
    mangle_struct_name,
    qualified_class_name,
    qualified_struct_name,
)
from irx.analysis.resolved_nodes import FunctionSignature
from irx.analysis.types import (
    bit_width,
    common_numeric_type,
    float_promotion_width_for_integer_width,
    is_boolean_type,
    is_float_type,
    is_integer_type,
    is_unsigned_type,
)
from irx.builder.base import BuilderVisitor
from irx.builder.protocols import VisitorProtocol
from irx.builder.runtime import safe_pop
from irx.builder.runtime.registry import (
    RuntimeFeatureState,
    get_default_runtime_feature_registry,
)
from irx.builder.state import (
    CleanupEmitter,
    LoopTargets,
    NamedValueMap,
    ResultStackValue,
)
from irx.builder.types import (
    VariablesLLVM,
    is_fp_type,
    is_int_type,
)
from irx.builder.vector import (
    is_vector,
    splat_scalar,
)
from irx.typecheck import typechecked

FLOAT16_BITS = 16
FLOAT32_BITS = 32
FLOAT64_BITS = 64
FLOAT128_BITS = 128


@private
@typechecked
def is_unsigned_node(node: astx.AST) -> bool:
    """
    title: Is unsigned node.
    parameters:
      node:
        type: astx.AST
    returns:
      type: bool
    """
    type_ = getattr(node, "type_", None)
    return isinstance(type_, astx.UnsignedInteger)


@private
@typechecked
def uses_unsigned_semantics(node: astx.AST) -> bool:
    """
    title: Uses unsigned semantics.
    parameters:
      node:
        type: astx.AST
    returns:
      type: bool
    """
    semantic = getattr(node, "semantic", None)
    semantic_flags = getattr(semantic, "semantic_flags", None)
    semantic_unsigned = getattr(semantic_flags, "unsigned", None)
    if semantic_unsigned is not None:
        return cast(bool, semantic_unsigned)

    explicit_unsigned = cast(bool | None, getattr(node, "unsigned", None))
    if explicit_unsigned is not None:
        return explicit_unsigned
    return is_unsigned_node(node)


@private
@typechecked
def semantic_symbol_key(node: astx.AST, fallback: str) -> str:
    """
    title: Semantic symbol key.
    parameters:
      node:
        type: astx.AST
      fallback:
        type: str
    returns:
      type: str
    """
    semantic = getattr(node, "semantic", None)
    symbol = getattr(semantic, "resolved_symbol", None)
    symbol_id = getattr(symbol, "symbol_id", None)
    if symbol_id is not None:
        return cast(str, symbol_id)
    return fallback


@private
@typechecked
def semantic_assignment_key(node: astx.AST, fallback: str) -> str:
    """
    title: Semantic assignment key.
    parameters:
      node:
        type: astx.AST
      fallback:
        type: str
    returns:
      type: str
    """
    semantic = getattr(node, "semantic", None)
    assignment = getattr(semantic, "resolved_assignment", None)
    target = getattr(assignment, "target", None)
    symbol_id = getattr(target, "symbol_id", None)
    if symbol_id is not None:
        return cast(str, symbol_id)
    return fallback


@private
@typechecked
def semantic_function_key(node: astx.AST, fallback: str) -> str:
    """
    title: Semantic function key.
    parameters:
      node:
        type: astx.AST
      fallback:
        type: str
    returns:
      type: str
    """
    semantic = getattr(node, "semantic", None)
    function = getattr(semantic, "resolved_function", None)
    symbol_id = getattr(function, "symbol_id", None)
    if symbol_id is not None:
        return cast(str, symbol_id)
    return fallback


@private
@typechecked
def semantic_function_name(node: astx.AST, fallback: str) -> str:
    """
    title: Semantic LLVM function name.
    parameters:
      node:
        type: astx.AST
      fallback:
        type: str
    returns:
      type: str
    """
    semantic = getattr(node, "semantic", None)
    function = getattr(semantic, "resolved_function", None)
    signature = getattr(function, "signature", None)
    signature_symbol_name = getattr(signature, "symbol_name", None)
    signature_is_extern = getattr(signature, "is_extern", False)
    module_key = getattr(function, "module_key", None)
    name = getattr(function, "name", None)
    if signature_is_extern and isinstance(signature_symbol_name, str):
        return signature_symbol_name
    if module_key is not None and name is not None:
        base_name = (
            signature_symbol_name
            if isinstance(signature_symbol_name, str) and signature_symbol_name
            else name
        )
        return mangle_function_name(module_key, base_name)
    return fallback


@private
@typechecked
def semantic_struct_key(node: astx.AST, fallback: str) -> str:
    """
    title: Semantic struct key.
    parameters:
      node:
        type: astx.AST
      fallback:
        type: str
    returns:
      type: str
    """
    semantic = getattr(node, "semantic", None)
    struct = getattr(semantic, "resolved_struct", None)
    qualified_name = getattr(struct, "qualified_name", None)
    if qualified_name is not None:
        return cast(str, qualified_name)
    return fallback


@private
@typechecked
def semantic_struct_name(node: astx.AST, fallback: str) -> str:
    """
    title: Semantic LLVM struct name.
    parameters:
      node:
        type: astx.AST
      fallback:
        type: str
    returns:
      type: str
    """
    semantic = getattr(node, "semantic", None)
    struct = getattr(semantic, "resolved_struct", None)
    module_key = getattr(struct, "module_key", None)
    name = getattr(struct, "name", None)
    if module_key is not None and name is not None:
        return mangle_struct_name(module_key, name)
    return fallback


@private
@typechecked
def semantic_class_key(node: astx.AST, fallback: str) -> str:
    """
    title: Semantic class key.
    parameters:
      node:
        type: astx.AST
      fallback:
        type: str
    returns:
      type: str
    """
    semantic = getattr(node, "semantic", None)
    class_ = getattr(semantic, "resolved_class", None)
    qualified_name = getattr(class_, "qualified_name", None)
    if qualified_name is not None:
        return cast(str, qualified_name)
    return fallback


@private
@typechecked
def semantic_class_name(node: astx.AST, fallback: str) -> str:
    """
    title: Semantic LLVM class-object name.
    parameters:
      node:
        type: astx.AST
      fallback:
        type: str
    returns:
      type: str
    """
    semantic = getattr(node, "semantic", None)
    class_ = getattr(semantic, "resolved_class", None)
    layout = getattr(class_, "layout", None)
    llvm_name = getattr(layout, "llvm_name", None)
    if isinstance(llvm_name, str) and llvm_name:
        return llvm_name
    module_key = getattr(class_, "module_key", None)
    name = getattr(class_, "name", None)
    if module_key is not None and name is not None:
        return mangle_class_name(module_key, name)
    return fallback


@private
@typechecked
def semantic_flag(node: astx.AST, name: str, default: bool = False) -> bool:
    """
    title: Semantic flag.
    parameters:
      node:
        type: astx.AST
      name:
        type: str
      default:
        type: bool
    returns:
      type: bool
    """
    semantic = getattr(node, "semantic", None)
    semantic_flags = getattr(semantic, "semantic_flags", None)
    if semantic_flags is not None and hasattr(semantic_flags, name):
        return bool(getattr(semantic_flags, name))
    return bool(getattr(node, name, default))


@private
@typechecked
def semantic_fma_rhs(node: astx.AST) -> astx.AST | None:
    """
    title: Semantic fma rhs.
    parameters:
      node:
        type: astx.AST
    returns:
      type: astx.AST | None
    """
    semantic = getattr(node, "semantic", None)
    semantic_flags = getattr(semantic, "semantic_flags", None)
    fma_rhs = getattr(semantic_flags, "fma_rhs", None)
    if fma_rhs is not None:
        return cast(astx.AST, fma_rhs)
    return getattr(node, "fma_rhs", None)


@private
@typechecked
class VisitorCore(BuilderVisitor):
    named_values: NamedValueMap
    _llvm: VariablesLLVM
    function_protos: dict[str, astx.FunctionPrototype]
    llvm_functions_by_symbol_id: dict[str, ir.Function]
    result_stack: list[ResultStackValue]
    runtime_features: RuntimeFeatureState
    const_vars: set[str]
    loop_stack: list[LoopTargets]
    cleanup_stack: list[CleanupEmitter]
    _set_value_ids: dict[int, ir.Value]
    _buffer_view_global_counter: int
    struct_types: dict[str, ir.Type]
    llvm_structs_by_qualified_name: dict[str, ir.IdentifiedStructType]
    _emitted_function_bodies: set[str]
    _module_display_names: dict[int, str]
    _current_module_display_name: str | None
    _interned_c_strings: dict[tuple[str, str], ir.GlobalVariable]
    _c_string_global_counter: int
    _namespace_globals: dict[tuple[str, str], ir.GlobalVariable]
    entry_function_symbol_id: str | None
    _fast_math_enabled: bool
    _current_function_return_type: astx.DataType | None
    _current_function_signature: FunctionSignature | None
    _generator_frame_types: dict[str, ir.IdentifiedStructType]
    _generator_frame_slots_by_symbol_id: dict[str, dict[str, int]]
    _generator_resume_functions: dict[str, ir.Function]
    _current_generator_frame_ptr: ir.Value | None
    _current_generator_frame_slots: dict[str, int]
    _current_generator_out_ptr: ir.Value | None
    _current_generator_next_state: int | None
    target: llvm.TargetRef
    target_machine: llvm.TargetMachine

    def __init__(
        self,
        active_runtime_features: set[str] | None = None,
    ) -> None:
        """
        title: Initialize VisitorCore.
        parameters:
          active_runtime_features:
            type: set[str] | None
        """
        super().__init__()
        self.named_values = {}
        self.const_vars = set()
        self.function_protos = {}
        self.llvm_functions_by_symbol_id = {}
        self.result_stack = []
        self.loop_stack = []
        self.cleanup_stack = []
        self._set_value_ids = {}
        self._buffer_view_global_counter = 0
        self.struct_types = {}
        self.llvm_structs_by_qualified_name = {}
        self._emitted_function_bodies = set()
        self._module_display_names = {}
        self._current_module_display_name = None
        self._interned_c_strings = {}
        self._c_string_global_counter = 0
        self._namespace_globals = {}
        self.entry_function_symbol_id = None
        self._fast_math_enabled = False
        self._current_function_return_type = None
        self._current_function_signature = None
        self._generator_frame_types = {}
        self._generator_frame_slots_by_symbol_id = {}
        self._generator_resume_functions = {}
        self._current_generator_frame_ptr = None
        self._current_generator_frame_slots = {}
        self._current_generator_out_ptr = None
        self._current_generator_next_state = None

        self.initialize()
        self.target = llvm.Target.from_default_triple()
        try:
            self.target_machine = self.target.create_target_machine(
                codemodel="small",
                reloc="pic",
            )
        except TypeError:
            self.target_machine = self.target.create_target_machine(
                codemodel="small"
            )

        self._llvm.module.triple = self.target_machine.triple
        self._llvm.module.data_layout = str(self.target_machine.target_data)

        if self._llvm.SIZE_T_TYPE is None:
            self._llvm.SIZE_T_TYPE = self._get_size_t_type_from_triple()

        self._add_builtins()
        self.runtime_features = RuntimeFeatureState(
            owner=cast(VisitorProtocol, self),
            registry=get_default_runtime_feature_registry(),
            active_features=active_runtime_features,
        )

    @dispatch
    def visit(self, node: astx.AST) -> None:
        """
        title: Visit AST nodes.
        parameters:
          node:
            type: astx.AST
        """
        super().visit(node)

    def _emit_active_cleanups(self, start_depth: int = 0) -> None:
        """
        title: Emit all active cleanup actions in innermost-first order.
        parameters:
          start_depth:
            type: int
        """
        for cleanup in reversed(self.cleanup_stack[start_depth:]):
            cleanup()

    def translate(self, node: astx.AST) -> str:
        """
        title: Translate.
        parameters:
          node:
            type: astx.AST
        returns:
          type: str
        """
        analyzed = analyze(node)
        self._module_display_names = {}
        self._current_module_display_name = None
        self._interned_c_strings = {}
        self._c_string_global_counter = 0
        if isinstance(analyzed, astx.Module):
            self._module_display_names[id(analyzed)] = (
                getattr(analyzed, "name", "") or "<module>"
            )
            self._set_entry_function_from_module(analyzed)
            self._translate_modules([analyzed])
        else:
            self.visit(analyzed)
        return str(self._llvm.module)

    def translate_modules(
        self,
        root: ParsedModule,
        resolver: ImportResolver,
    ) -> str:
        """
        title: Translate a reachable graph of parsed modules to LLVM IR.
        parameters:
          root:
            type: ParsedModule
          resolver:
            type: ImportResolver
        returns:
          type: str
        """
        session = analyze_modules(root, resolver)
        ordered_modules = session.ordered_modules()
        self._module_display_names = {
            id(parsed_module.ast): (
                parsed_module.display_name
                or getattr(parsed_module.ast, "name", "")
                or str(parsed_module.key)
            )
            for parsed_module in ordered_modules
        }
        self._set_entry_function_from_module(root.ast)
        self._translate_modules(
            [parsed_module.ast for parsed_module in ordered_modules]
        )
        return str(self._llvm.module)

    def _set_entry_function_from_module(self, module: astx.Module) -> None:
        """
        title: >-
          Record the root entrypoint semantic id when a main function exists.
        parameters:
          module:
            type: astx.Module
        """
        self.entry_function_symbol_id = None
        for node in module.nodes:
            candidate = None
            if (
                isinstance(node, astx.FunctionPrototype)
                and node.name == "main"
            ):
                candidate = node
            elif isinstance(node, astx.FunctionDef) and node.name == "main":
                candidate = node.prototype
            if candidate is None:
                continue
            self.entry_function_symbol_id = semantic_function_key(
                candidate,
                "main",
            )
            return

    def llvm_function_name_for_node(
        self,
        node: astx.AST,
        fallback: str,
    ) -> str:
        """
        title: Return the LLVM symbol name for a function node.
        parameters:
          node:
            type: astx.AST
          fallback:
            type: str
        returns:
          type: str
        """
        function_key = semantic_function_key(node, fallback)
        if (
            self.entry_function_symbol_id is not None
            and function_key == self.entry_function_symbol_id
        ):
            return "main"
        return semantic_function_name(node, fallback)

    def _namespace_global(
        self,
        namespace_type: astx.NamespaceType,
    ) -> ir.GlobalVariable:
        """
        title: Return one stable LLVM global backing a namespace handle.
        parameters:
          namespace_type:
            type: astx.NamespaceType
        returns:
          type: ir.GlobalVariable
        """
        cache_key = (
            namespace_type.namespace_key,
            namespace_type.namespace_kind.value,
        )
        existing = self._namespace_globals.get(cache_key)
        if existing is not None:
            return existing

        global_name = mangle_namespace_name(
            namespace_type.namespace_key,
            namespace_type.namespace_kind.value,
        )
        module_globals = self._llvm.module.globals
        global_value = module_globals.get(global_name)
        if global_value is not None:
            namespace_global = cast(ir.GlobalVariable, global_value)
            self._namespace_globals[cache_key] = namespace_global
            return namespace_global

        namespace_global = ir.GlobalVariable(
            self._llvm.module,
            self._llvm.INT8_TYPE,
            name=global_name,
        )
        namespace_global.linkage = "internal"
        namespace_global.global_constant = True
        namespace_global.initializer = ir.Constant(self._llvm.INT8_TYPE, 0)
        self._namespace_globals[cache_key] = namespace_global
        return namespace_global

    def _namespace_value_for_type(
        self,
        namespace_type: astx.NamespaceType,
    ) -> ir.Value:
        """
        title: Return one lowered value for a namespace type.
        parameters:
          namespace_type:
            type: astx.NamespaceType
        returns:
          type: ir.Value
        """
        namespace_global = self._namespace_global(namespace_type)
        return namespace_global.bitcast(self._llvm.OPAQUE_POINTER_TYPE)

    def _namespace_value(
        self,
        node: astx.AST,
    ) -> ir.Value | None:
        """
        title: Return one lowered namespace value for an analyzed node.
        parameters:
          node:
            type: astx.AST
        returns:
          type: ir.Value | None
        """
        semantic = getattr(node, "semantic", None)
        resolved_module = getattr(semantic, "resolved_module", None)
        if resolved_module is not None:
            return self._namespace_value_for_type(
                astx.NamespaceType(
                    resolved_module.module_key,
                    namespace_kind=astx.NamespaceKind.MODULE,
                    display_name=resolved_module.display_name,
                )
            )
        resolved_type = self._resolved_ast_type(node)
        if isinstance(resolved_type, astx.NamespaceType):
            return self._namespace_value_for_type(resolved_type)
        return None

    def _translate_modules(self, modules: list[astx.Module]) -> None:
        """
        title: Translate a list of already-analyzed modules.
        parameters:
          modules:
            type: list[astx.Module]
        """
        for module in modules:
            self._current_module_display_name = self._module_display_names.get(
                id(module),
                getattr(module, "name", "") or "<module>",
            )
            for node in module.nodes:
                if isinstance(node, (astx.StructDefStmt, astx.ClassDefStmt)):
                    self.visit(node)

        for module in modules:
            self._current_module_display_name = self._module_display_names.get(
                id(module),
                getattr(module, "name", "") or "<module>",
            )
            function_nodes = [
                *module.nodes,
                *astx.generated_template_nodes(module),
            ]
            for node in function_nodes:
                if isinstance(node, astx.FunctionPrototype):
                    if astx.is_template_node(
                        node
                    ) and not astx.is_template_specialization(node):
                        continue
                    self.visit(node)
                elif isinstance(node, astx.FunctionDef):
                    if astx.is_template_node(
                        node.prototype
                    ) and not astx.is_template_specialization(node):
                        continue
                    self.visit(node.prototype)

        for module in modules:
            self._current_module_display_name = self._module_display_names.get(
                id(module),
                getattr(module, "name", "") or "<module>",
            )
            function_nodes = [
                *module.nodes,
                *astx.generated_template_nodes(module),
            ]
            for node in function_nodes:
                if isinstance(node, astx.FunctionDef):
                    if astx.is_template_node(
                        node.prototype
                    ) and not astx.is_template_specialization(node):
                        continue
                    self.visit(node)

        self._current_module_display_name = None

    def activate_runtime_feature(self, feature_name: str) -> None:
        """
        title: Activate runtime feature.
        parameters:
          feature_name:
            type: str
        """
        self.runtime_features.activate(feature_name)

    def require_runtime_symbol(
        self, feature_name: str, symbol_name: str
    ) -> ir.Function:
        """
        title: Require runtime symbol.
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
        title: Init native size types.
        """
        self._llvm.POINTER_BITS = ctypes.sizeof(ctypes.c_void_p) * 8
        self._llvm.SIZE_T_TYPE = None

    def _get_size_t_type_from_triple(self) -> ir.IntType:
        """
        title: Get size t type from triple.
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
        if any(arch in triple for arch in ["i386", "i686", "arm", "mips"]):
            if "64" in triple:
                return ir.IntType(64)
            return ir.IntType(32)

        return ir.IntType(ctypes.sizeof(ctypes.c_size_t) * 8)

    def initialize(self) -> None:
        """
        title: Initialize.
        """
        self._llvm = VariablesLLVM()
        # Keep identified class/struct types isolated per translation so
        # reused semantic names never retain stale LLVM bodies.
        llvm_context = ir.Context()
        self._llvm.module = ir.module.Module(
            "Arx",
            context=llvm_context,
        )
        self._llvm.context = llvm_context
        self._init_native_size_types()

        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        llvm.initialize_native_target()
        llvm.initialize_native_asmparser()
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
        self._llvm.UINT8_TYPE = ir.IntType(8)
        self._llvm.UINT16_TYPE = ir.IntType(16)
        self._llvm.UINT32_TYPE = ir.IntType(32)
        self._llvm.UINT64_TYPE = ir.IntType(64)
        self._llvm.UINT128_TYPE = ir.IntType(128)
        self._llvm.VOID_TYPE = ir.VoidType()
        self._llvm.ASCII_STRING_TYPE = ir.IntType(8).as_pointer()
        self._llvm.UTF8_STRING_TYPE = self._llvm.ASCII_STRING_TYPE
        self._llvm.OPAQUE_POINTER_TYPE = self._llvm.INT8_TYPE.as_pointer()
        self._llvm.BUFFER_OWNER_HANDLE_TYPE = self._llvm.OPAQUE_POINTER_TYPE
        buffer_view_type = self._llvm.module.context.get_identified_type(
            BUFFER_VIEW_TYPE_NAME
        )
        if buffer_view_type.is_opaque:
            buffer_view_type.set_body(
                self._llvm.OPAQUE_POINTER_TYPE,
                self._llvm.BUFFER_OWNER_HANDLE_TYPE,
                self._llvm.OPAQUE_POINTER_TYPE,
                self._llvm.INT32_TYPE,
                self._llvm.INT64_TYPE.as_pointer(),
                self._llvm.INT64_TYPE.as_pointer(),
                self._llvm.INT64_TYPE,
                self._llvm.INT32_TYPE,
            )
        self._llvm.BUFFER_VIEW_TYPE = buffer_view_type
        self._llvm.ARRAY_BUILDER_HANDLE_TYPE = self._llvm.OPAQUE_POINTER_TYPE
        self._llvm.ARRAY_HANDLE_TYPE = self._llvm.OPAQUE_POINTER_TYPE
        self._llvm.ARROW_ARRAY_BUILDER_HANDLE_TYPE = (
            self._llvm.ARRAY_BUILDER_HANDLE_TYPE
        )
        self._llvm.ARROW_ARRAY_HANDLE_TYPE = self._llvm.ARRAY_HANDLE_TYPE
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

    def _add_builtins(self) -> None:
        """
        title: Add builtins.
        """
        putchar_ty = ir.FunctionType(
            self._llvm.INT32_TYPE,
            [self._llvm.INT32_TYPE],
        )
        putchar = ir.Function(self._llvm.module, putchar_ty, "putchar")

        putchard_ty = ir.FunctionType(
            self._llvm.INT32_TYPE,
            [self._llvm.DOUBLE_TYPE],
        )
        putchard = ir.Function(self._llvm.module, putchard_ty, "putchard")

        ir_builder = ir.IRBuilder(putchard.append_basic_block("entry"))
        ival = ir_builder.fptoui(
            putchard.args[0], self._llvm.INT32_TYPE, "intcast"
        )
        ir_builder.call(putchar, [ival])
        ir_builder.ret(ir.Constant(self._llvm.INT32_TYPE, 0))

    def get_function(self, name: str) -> ir.Function | None:
        """
        title: Get function.
        parameters:
          name:
            type: str
        returns:
          type: ir.Function | None
        """
        if name in self.llvm_functions_by_symbol_id:
            return self.llvm_functions_by_symbol_id[name]

        if name in self._llvm.module.globals:
            return cast(ir.Function, self._llvm.module.get_global(name))

        if name in self.function_protos:
            self.visit(self.function_protos[name])
            return cast(ir.Function, safe_pop(self.result_stack))

        return None

    def create_entry_block_alloca(
        self,
        var_name: str,
        type_name: str | ir.Type,
    ) -> Any:
        """
        title: Create entry block alloca.
        parameters:
          var_name:
            type: str
          type_name:
            type: str | ir.Type
        returns:
          type: Any
        """
        llvm_type = (
            self._llvm.get_data_type(type_name)
            if isinstance(type_name, str)
            else type_name
        )
        current_block = self._llvm.ir_builder.block
        self._llvm.ir_builder.position_at_start(
            self._llvm.ir_builder.function.entry_basic_block
        )
        alloca = self._llvm.ir_builder.alloca(llvm_type, None, var_name)
        if current_block is not None:
            self._llvm.ir_builder.position_at_end(current_block)
        return alloca

    def _get_fma_function(self, ty: ir.Type) -> ir.Function:
        """
        title: Get fma function.
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
            return cast(ir.Function, self._llvm.module.get_global(name))

        fn_ty = ir.FunctionType(ty, [ty, ty, ty])
        fn = ir.Function(self._llvm.module, fn_ty, name)
        fn.linkage = "external"
        return fn

    def _emit_fma(
        self,
        lhs: ir.Value,
        rhs: ir.Value,
        addend: ir.Value,
    ) -> ir.Value:
        """
        title: Emit fma.
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
        if not isinstance(lhs.type, ir.VectorType) and hasattr(builder, "fma"):
            inst = builder.fma(lhs, rhs, addend, name="vfma")
            self._apply_fast_math(inst)
            return inst
        fma_fn = self._get_fma_function(lhs.type)
        inst = builder.call(fma_fn, [lhs, rhs, addend], name="vfma")
        self._apply_fast_math(inst)
        return inst

    def set_fast_math(self, enabled: bool) -> None:
        """
        title: Set fast math.
        parameters:
          enabled:
            type: bool
        """
        self._fast_math_enabled = enabled

    def _apply_fast_math(self, inst: ir.Instruction) -> None:
        """
        title: Apply fast math.
        parameters:
          inst:
            type: ir.Instruction
        """
        if not self._fast_math_enabled:
            return

        type_ = inst.type
        if isinstance(type_, ir.VectorType):
            if not is_fp_type(type_.element):
                return
        elif not is_fp_type(type_):
            return

        flags = getattr(inst, "flags", None)
        if flags is None or "fast" in flags:
            return

        try:
            flags.append("fast")
        except (AttributeError, TypeError):
            return

    def _is_numeric_value(self, value: ir.Value) -> bool:
        """
        title: Is numeric value.
        parameters:
          value:
            type: ir.Value
        returns:
          type: bool
        """
        if is_vector(value):
            elem_ty = value.type.element
            return isinstance(elem_ty, ir.IntType) or is_fp_type(elem_ty)
        return isinstance(value.type, ir.IntType) or is_fp_type(value.type)

    def _resolved_ast_type(
        self, node: astx.AST | None
    ) -> astx.DataType | None:
        """
        title: Resolved ast type.
        parameters:
          node:
            type: astx.AST | None
        returns:
          type: astx.DataType | None
        """
        if node is None:
            return None
        semantic = getattr(node, "semantic", None)
        resolved_type = getattr(semantic, "resolved_type", None)
        if isinstance(resolved_type, astx.DataType):
            return resolved_type
        fallback = getattr(node, "type_", None)
        return fallback if isinstance(fallback, astx.DataType) else None

    def _llvm_type_for_ast_type(
        self, type_: astx.DataType | None
    ) -> ir.Type | None:
        """
        title: Llvm type for ast type.
        parameters:
          type_:
            type: astx.DataType | None
        returns:
          type: ir.Type | None
        """
        if type_ is None:
            return None
        if isinstance(type_, astx.NamespaceType):
            return self._llvm.OPAQUE_POINTER_TYPE
        if isinstance(type_, astx.GeneratorType):
            return ir.LiteralStructType(
                [
                    self._llvm.OPAQUE_POINTER_TYPE,
                    self._llvm.OPAQUE_POINTER_TYPE,
                ]
            )
        if isinstance(type_, astx.BufferOwnerType):
            return self._llvm.BUFFER_OWNER_HANDLE_TYPE
        if isinstance(type_, astx.OpaqueHandleType):
            return self._llvm.OPAQUE_POINTER_TYPE
        if isinstance(type_, astx.PointerType):
            if type_.pointee_type is None:
                return self._llvm.OPAQUE_POINTER_TYPE
            pointee_type = self._llvm_type_for_ast_type(type_.pointee_type)
            if pointee_type is None:
                return None
            return pointee_type.as_pointer()
        if isinstance(type_, astx.ListType):
            return ir.LiteralStructType(
                [
                    self._llvm.INT8_TYPE.as_pointer(),
                    self._llvm.INT64_TYPE,
                    self._llvm.INT64_TYPE,
                    self._llvm.INT64_TYPE,
                ]
            )
        if isinstance(type_, astx.BufferViewType):
            return self._llvm.BUFFER_VIEW_TYPE
        if isinstance(type_, astx.NDArrayType):
            return self._llvm.BUFFER_VIEW_TYPE
        if isinstance(type_, astx.StructType):
            struct_key = type_.qualified_name
            if struct_key is None and type_.module_key is not None:
                struct_key = qualified_struct_name(
                    type_.module_key,
                    type_.resolved_name or type_.name,
                )
            if struct_key is None:
                return None
            return self.struct_types.get(struct_key)
        if isinstance(type_, astx.ClassType):
            class_key = type_.qualified_name
            if class_key is None and type_.module_key is not None:
                class_key = qualified_class_name(
                    type_.module_key,
                    type_.resolved_name or type_.name,
                )
            if class_key is None:
                return None
            class_type = self.struct_types.get(class_key)
            if class_type is None:
                return None
            return class_type.as_pointer()
        type_name = type_.__class__.__name__.lower()
        return self._llvm.get_data_type(type_name)

    def _resolved_class_receiver_field_address(
        self,
        *,
        receiver: astx.AST,
        member_name: str,
        storage_index: int,
    ) -> ir.Value:
        """
        title: Lower one resolved class-instance field slot to an address.
        parameters:
          receiver:
            type: astx.AST
          member_name:
            type: str
          storage_index:
            type: int
        returns:
          type: ir.Value
        """
        self.visit(receiver)
        base_value = safe_pop(self.result_stack)
        if base_value is None:
            raise Exception("codegen: invalid class field access base.")
        base_type = self._resolved_ast_type(receiver)
        llvm_base_type = self._llvm_type_for_ast_type(base_type)
        if llvm_base_type is not None and base_value.type != llvm_base_type:
            base_value = self._llvm.ir_builder.bitcast(
                base_value,
                llvm_base_type,
                name=f"{member_name}_class_base",
            )
        return self._llvm.ir_builder.gep(
            base_value,
            [
                ir.Constant(self._llvm.INT32_TYPE, 0),
                ir.Constant(self._llvm.INT32_TYPE, storage_index),
            ],
            inbounds=True,
            name=f"{member_name}_addr",
        )

    def _field_address(self, node: astx.FieldAccess) -> ir.Value:
        """
        title: Lower one field-access expression to an address.
        parameters:
          node:
            type: astx.FieldAccess
        returns:
          type: ir.Value
        """
        semantic = getattr(node, "semantic", None)
        resolved_class_field_access = getattr(
            semantic,
            "resolved_class_field_access",
            None,
        )
        if resolved_class_field_access is not None:
            return self._resolved_class_receiver_field_address(
                receiver=node.value,
                member_name=resolved_class_field_access.member.name,
                storage_index=resolved_class_field_access.field.storage_index,
            )

        resolved_field_access = getattr(
            semantic,
            "resolved_field_access",
            None,
        )
        if resolved_field_access is None:
            raise Exception("codegen: unresolved field access.")

        if isinstance(node.value, astx.Identifier):
            base_key = semantic_symbol_key(node.value, node.value.name)
            base_ptr = self.named_values.get(base_key)
            if base_ptr is None:
                raise Exception(f"Unknown variable name: {node.value.name}")
        elif isinstance(node.value, astx.FieldAccess):
            base_ptr = self._field_address(node.value)
        else:
            self.visit(node.value)
            base_value = safe_pop(self.result_stack)
            if base_value is None:
                raise Exception("codegen: invalid field access base.")
            base_type = self._resolved_ast_type(node.value)
            llvm_base_type = self._llvm_type_for_ast_type(base_type)
            if llvm_base_type is None:
                llvm_base_type = base_value.type
            temp_name = f"fieldtmp_{id(node.value)}"
            base_ptr = self.create_entry_block_alloca(
                temp_name,
                llvm_base_type,
            )
            self._llvm.ir_builder.store(base_value, base_ptr)

        indices = [
            ir.Constant(self._llvm.INT32_TYPE, 0),
            ir.Constant(
                self._llvm.INT32_TYPE,
                resolved_field_access.field.index,
            ),
        ]
        source_etype = self._llvm_type_for_ast_type(
            self._resolved_ast_type(node.value)
        )
        if not isinstance(node.value, astx.FieldAccess):
            return self._llvm.ir_builder.gep(
                base_ptr,
                indices,
                inbounds=True,
                name=f"{resolved_field_access.field.name}_addr",
            )
        if source_etype is not None:
            typed_ptr = source_etype.as_pointer()
            if base_ptr.type != typed_ptr:
                base_ptr = self._llvm.ir_builder.bitcast(
                    base_ptr,
                    typed_ptr,
                    name=f"{resolved_field_access.field.name}_baseptr",
                )
        return self._llvm.ir_builder.gep(
            base_ptr,
            indices,
            inbounds=True,
            name=f"{resolved_field_access.field.name}_addr",
        )

    def _base_class_field_address(
        self,
        node: astx.BaseFieldAccess,
    ) -> ir.Value:
        """
        title: Lower one base-qualified class field access to an address.
        parameters:
          node:
            type: astx.BaseFieldAccess
        returns:
          type: ir.Value
        """
        semantic = getattr(node, "semantic", None)
        resolved_base_field_access = getattr(
            semantic,
            "resolved_base_class_field_access",
            None,
        )
        if resolved_base_field_access is None:
            raise Exception("codegen: unresolved base class field access")
        return self._resolved_class_receiver_field_address(
            receiver=node.receiver,
            member_name=resolved_base_field_access.member.name,
            storage_index=resolved_base_field_access.field.storage_index,
        )

    def _static_class_field_address(
        self,
        node: astx.StaticFieldAccess,
    ) -> ir.Value:
        """
        title: Lower one static class field access to a global address.
        parameters:
          node:
            type: astx.StaticFieldAccess
        returns:
          type: ir.Value
        """
        semantic = getattr(node, "semantic", None)
        resolved_static_field_access = getattr(
            semantic,
            "resolved_static_class_field_access",
            None,
        )
        if resolved_static_field_access is None:
            raise Exception("codegen: unresolved static class field access")
        global_name = resolved_static_field_access.storage.global_name
        global_value = self._llvm.module.globals.get(global_name)
        if global_value is None:
            raise Exception(
                f"codegen: missing static class field global '{global_name}'"
            )
        return cast(ir.Value, global_value)

    def _lvalue_address(
        self,
        node: astx.AST,
    ) -> ir.Value:
        """
        title: Lower one mutable lvalue target to an address.
        parameters:
          node:
            type: astx.AST
        returns:
          type: ir.Value
        """
        if isinstance(node, astx.Identifier):
            symbol_key = semantic_symbol_key(node, node.name)
            address = self.named_values.get(symbol_key)
            if address is None:
                raise Exception(f"Unknown variable name: {node.name}")
            return address
        if isinstance(node, astx.FieldAccess):
            return self._field_address(node)
        if isinstance(node, astx.BaseFieldAccess):
            return self._base_class_field_address(node)
        if isinstance(node, astx.StaticFieldAccess):
            return self._static_class_field_address(node)
        raise Exception("codegen: invalid mutable target")

    def _bool_value_from_numeric(
        self,
        value: ir.Value,
        source_type: astx.DataType | None,
        *,
        name: str,
    ) -> ir.Value:
        """
        title: Convert one numeric value to boolean truthiness.
        parameters:
          value:
            type: ir.Value
          source_type:
            type: astx.DataType | None
          name:
            type: str
        returns:
          type: ir.Value
        """
        if is_boolean_type(source_type):
            return value
        if is_float_type(source_type):
            zero = ir.Constant(value.type, 0.0)
            return self._emit_numeric_compare(
                "!=",
                value,
                zero,
                unsigned=False,
                name=name,
            )
        if is_integer_type(source_type):
            zero = ir.Constant(value.type, 0)
            return self._emit_numeric_compare(
                "!=",
                value,
                zero,
                unsigned=True,
                name=name,
            )
        raise Exception(f"Unsupported boolean conversion from {source_type!r}")

    def _emit_numeric_compare(
        self,
        op_code: str,
        lhs: ir.Value,
        rhs: ir.Value,
        *,
        unsigned: bool,
        name: str,
    ) -> ir.Value:
        """
        title: Emit one numeric compare using shared signedness policy.
        parameters:
          op_code:
            type: str
          lhs:
            type: ir.Value
          rhs:
            type: ir.Value
          unsigned:
            type: bool
          name:
            type: str
        returns:
          type: ir.Value
        """
        if is_fp_type(lhs.type):
            return self._llvm.ir_builder.fcmp_ordered(op_code, lhs, rhs, name)
        if unsigned:
            return self._llvm.ir_builder.icmp_unsigned(
                op_code,
                lhs,
                rhs,
                name,
            )
        return self._llvm.ir_builder.icmp_signed(op_code, lhs, rhs, name)

    def _cast_ast_value(
        self,
        value: ir.Value,
        *,
        source_type: astx.DataType | None,
        target_type: astx.DataType | None,
    ) -> ir.Value:
        """
        title: Cast one lowered value using semantic scalar types.
        parameters:
          value:
            type: ir.Value
          source_type:
            type: astx.DataType | None
          target_type:
            type: astx.DataType | None
        returns:
          type: ir.Value
        """
        if source_type is None or target_type is None:
            return value

        target_llvm_type = self._llvm_type_for_ast_type(target_type)
        if target_llvm_type is None:
            return value
        if value.type == target_llvm_type:
            return value

        builder = self._llvm.ir_builder

        if is_boolean_type(target_type):
            return self._bool_value_from_numeric(
                value,
                source_type,
                name="boolcast",
            )

        if is_boolean_type(source_type):
            if is_integer_type(target_type):
                return builder.zext(value, target_llvm_type, "bool_zext")
            if is_float_type(target_type):
                return builder.uitofp(value, target_llvm_type, "bool_to_fp")

        if is_integer_type(source_type) and is_integer_type(target_type):
            source_width = bit_width(source_type)
            target_width = bit_width(target_type)

            if source_width == target_width:
                return value
            if source_width < target_width:
                if is_unsigned_type(source_type):
                    return builder.zext(value, target_llvm_type, "zext")
                return builder.sext(value, target_llvm_type, "sext")
            return builder.trunc(value, target_llvm_type, "trunc")

        if is_integer_type(source_type) and is_float_type(target_type):
            if is_unsigned_type(source_type):
                return builder.uitofp(value, target_llvm_type, "uitofp")
            return builder.sitofp(value, target_llvm_type, "sitofp")

        if is_float_type(source_type) and is_integer_type(target_type):
            if is_unsigned_type(target_type):
                return builder.fptoui(value, target_llvm_type, "fptoui")
            return builder.fptosi(value, target_llvm_type, "fptosi")

        if is_float_type(source_type) and is_float_type(target_type):
            if bit_width(source_type) < bit_width(target_type):
                return builder.fpext(value, target_llvm_type, "fpext")
            return builder.fptrunc(value, target_llvm_type, "fptrunc")

        if isinstance(source_type, astx.ClassType) and isinstance(
            target_type,
            astx.ClassType,
        ):
            return builder.bitcast(value, target_llvm_type, "classcast")

        raise Exception(
            f"Unsupported scalar cast from {source_type!r} to {target_type!r}"
        )

    def _coerce_numeric_operands_for_types(
        self,
        lhs: ir.Value,
        rhs: ir.Value,
        *,
        lhs_type: astx.DataType | None,
        rhs_type: astx.DataType | None,
    ) -> tuple[ir.Value, ir.Value]:
        """
        title: Coerce numeric operands from semantic types.
        parameters:
          lhs:
            type: ir.Value
          rhs:
            type: ir.Value
          lhs_type:
            type: astx.DataType | None
          rhs_type:
            type: astx.DataType | None
        returns:
          type: tuple[ir.Value, ir.Value]
        """
        target_type = common_numeric_type(lhs_type, rhs_type)
        if target_type is None:
            return lhs, rhs
        return (
            self._cast_ast_value(
                lhs,
                source_type=lhs_type,
                target_type=target_type,
            ),
            self._cast_ast_value(
                rhs,
                source_type=rhs_type,
                target_type=target_type,
            ),
        )

    def _unify_numeric_operands(
        self,
        lhs: ir.Value,
        rhs: ir.Value,
        unsigned: bool = False,
    ) -> tuple[ir.Value, ir.Value]:
        """
        title: Unify numeric operands for raw LLVM values.
        summary: >-
          This is a fallback helper for low-level builder/test usage when
          semantic operand types are unavailable. Normal AST lowering should
          prefer semantic-aware coercion instead.
        parameters:
          lhs:
            type: ir.Value
          rhs:
            type: ir.Value
          unsigned:
            type: bool
        returns:
          type: tuple[ir.Value, ir.Value]
        """
        lhs_is_vec = is_vector(lhs)
        rhs_is_vec = is_vector(rhs)

        if lhs_is_vec and rhs_is_vec:
            if lhs.type.count != rhs.type.count:
                raise Exception(
                    "Vector size mismatch: "
                    f"{lhs.type.count} vs {rhs.type.count}"
                )
            if lhs.type.element != rhs.type.element:
                raise Exception(
                    "Vector element type mismatch: "
                    f"{lhs.type.element} vs {rhs.type.element}"
                )
            return lhs, rhs

        if lhs_is_vec:
            target_lanes = lhs.type.count
            target_scalar_ty = lhs.type.element
        elif rhs_is_vec:
            target_lanes = rhs.type.count
            target_scalar_ty = rhs.type.element
        else:
            target_lanes = None
            lhs_base_ty = lhs.type
            rhs_base_ty = rhs.type
            if is_fp_type(lhs_base_ty) or is_fp_type(rhs_base_ty):
                float_candidates = [
                    type_
                    for type_ in (lhs_base_ty, rhs_base_ty)
                    if is_fp_type(type_)
                ]
                integer_width = max(
                    (
                        getattr(type_, "width", 0)
                        for type_ in (lhs_base_ty, rhs_base_ty)
                        if is_int_type(type_)
                    ),
                    default=0,
                )
                float_width = max(
                    (
                        self._float_bit_width(type_)
                        for type_ in float_candidates
                    ),
                    default=0,
                )
                target_width = max(
                    float_width,
                    float_promotion_width_for_integer_width(integer_width),
                )
                target_scalar_ty = self._float_type_from_width(target_width)
            else:
                lhs_width = getattr(lhs_base_ty, "width", 0)
                rhs_width = getattr(rhs_base_ty, "width", 0)
                target_scalar_ty = ir.IntType(max(lhs_width, rhs_width, 1))

        lhs = self._cast_value_to_type(
            lhs, target_scalar_ty, unsigned=unsigned
        )
        rhs = self._cast_value_to_type(
            rhs, target_scalar_ty, unsigned=unsigned
        )

        if target_lanes:
            vec_ty = ir.VectorType(target_scalar_ty, target_lanes)
            if not is_vector(lhs):
                lhs = splat_scalar(self._llvm.ir_builder, lhs, vec_ty)
            if not is_vector(rhs):
                rhs = splat_scalar(self._llvm.ir_builder, rhs, vec_ty)

        return lhs, rhs

    def _select_float_type(self, candidates: list[ir.Type]) -> ir.Type:
        """
        title: Select float type.
        parameters:
          candidates:
            type: list[ir.Type]
        returns:
          type: ir.Type
        """
        if not candidates:
            return self._llvm.FLOAT_TYPE

        width = max(self._float_bit_width(type_) for type_ in candidates)
        return self._float_type_from_width(width)

    def _float_type_from_width(self, width: int) -> ir.Type:
        """
        title: Float type from width.
        parameters:
          width:
            type: int
        returns:
          type: ir.Type
        """
        if width <= FLOAT16_BITS and hasattr(self._llvm, "FLOAT16_TYPE"):
            return self._llvm.FLOAT16_TYPE
        if width <= FLOAT32_BITS:
            return self._llvm.FLOAT_TYPE
        if width <= FLOAT64_BITS:
            return self._llvm.DOUBLE_TYPE
        if FP128Type is not None and width >= FLOAT128_BITS:
            return FP128Type()
        return self._llvm.FLOAT_TYPE

    def _float_bit_width(self, type_: ir.Type) -> int:
        """
        title: Float bit width.
        parameters:
          type_:
            type: ir.Type
        returns:
          type: int
        """
        if isinstance(type_, DoubleType):
            return FLOAT64_BITS
        if isinstance(type_, FloatType):
            return FLOAT32_BITS
        if isinstance(type_, HalfType):
            return FLOAT16_BITS
        if FP128Type is not None and isinstance(type_, FP128Type):
            return FLOAT128_BITS
        raise Exception(f"Unknown floating-point type: {type_}")

    def _cast_value_to_type(
        self,
        value: ir.Value,
        target_scalar_ty: ir.Type,
        unsigned: bool = False,
    ) -> ir.Value:
        """
        title: Cast value to type.
        parameters:
          value:
            type: ir.Value
          target_scalar_ty:
            type: ir.Type
          unsigned:
            type: bool
        returns:
          type: ir.Value
        """
        builder = self._llvm.ir_builder
        value_is_vec = is_vector(value)
        if value_is_vec:
            lanes = value.type.count
            current_scalar_ty = value.type.element
            target_ty = ir.VectorType(target_scalar_ty, lanes)
        else:
            lanes = None
            current_scalar_ty = value.type
            target_ty = target_scalar_ty

        if current_scalar_ty == target_scalar_ty and value.type == target_ty:
            return value

        current_is_float = is_fp_type(current_scalar_ty)
        target_is_float = is_fp_type(target_scalar_ty)

        if target_is_float:
            if current_is_float:
                current_bits = self._float_bit_width(current_scalar_ty)
                target_bits = self._float_bit_width(target_scalar_ty)
                if current_bits == target_bits:
                    if value.type != target_ty:
                        return builder.bitcast(value, target_ty)
                    return value
                if current_bits < target_bits:
                    return builder.fpext(value, target_ty, "fpext")
                return builder.fptrunc(value, target_ty, "fptrunc")
            if unsigned:
                return builder.uitofp(value, target_ty, "uitofp")
            return builder.sitofp(value, target_ty, "sitofp")

        if current_is_float:
            raise Exception(
                "Cannot implicitly convert floating-point to integer"
            )

        current_width = getattr(current_scalar_ty, "width", 0)
        target_width = getattr(target_scalar_ty, "width", 0)
        if current_width == target_width:
            if value.type != target_ty:
                return builder.bitcast(value, target_ty)
            return value

        if current_width < target_width:
            if unsigned:
                return builder.zext(value, target_ty, "zext")
            return builder.sext(value, target_ty, "sext")

        return builder.trunc(value, target_ty, "trunc")

    def _common_list_element_type(
        self, lhs_ty: ir.Type, rhs_ty: ir.Type
    ) -> ir.Type:
        """
        title: Common list element type.
        parameters:
          lhs_ty:
            type: ir.Type
          rhs_ty:
            type: ir.Type
        returns:
          type: ir.Type
        """
        if lhs_ty == rhs_ty:
            return lhs_ty

        if is_int_type(lhs_ty) and is_int_type(rhs_ty):
            return lhs_ty if lhs_ty.width >= rhs_ty.width else rhs_ty

        lhs_is_fp = is_fp_type(lhs_ty)
        rhs_is_fp = is_fp_type(rhs_ty)
        if lhs_is_fp and rhs_is_fp:
            return self._select_float_type([lhs_ty, rhs_ty])

        if is_int_type(lhs_ty) and rhs_is_fp:
            return rhs_ty
        if is_int_type(rhs_ty) and lhs_is_fp:
            return lhs_ty

        if isinstance(lhs_ty, ir.PointerType) and isinstance(
            rhs_ty, ir.PointerType
        ):
            if lhs_ty == rhs_ty:
                return lhs_ty
            raise TypeError(
                "LiteralList: incompatible pointer types "
                f"{lhs_ty} and {rhs_ty}"
            )

        raise TypeError(
            f"LiteralList: cannot find common type for {lhs_ty} and {rhs_ty}"
        )

    def _coerce_to(self, value: ir.Value, target_ty: ir.Type) -> ir.Value:
        """
        title: Coerce to.
        parameters:
          value:
            type: ir.Value
          target_ty:
            type: ir.Type
        returns:
          type: ir.Value
        """
        if value.type == target_ty:
            return value

        if isinstance(value, ir.Constant):
            raw = value.constant
            if is_int_type(value.type) and is_int_type(target_ty):
                return ir.Constant(target_ty, int(raw))
            if is_int_type(value.type) and is_fp_type(target_ty):
                return ir.Constant(target_ty, float(raw))
            if is_fp_type(value.type) and is_fp_type(target_ty):
                return ir.Constant(target_ty, float(raw))
            if is_fp_type(value.type) and is_int_type(target_ty):
                return ir.Constant(target_ty, int(raw))

        builder = self._llvm.ir_builder
        if is_int_type(value.type) and is_int_type(target_ty):
            if value.type.width < target_ty.width:
                return builder.sext(value, target_ty, "list_sext")
            return builder.trunc(value, target_ty, "list_trunc")

        if is_int_type(value.type) and is_fp_type(target_ty):
            return builder.sitofp(value, target_ty, "list_itofp")

        if is_fp_type(value.type) and is_fp_type(target_ty):
            if self._float_bit_width(value.type) < self._float_bit_width(
                target_ty
            ):
                return builder.fpext(value, target_ty, "list_fpext")
            return builder.fptrunc(value, target_ty, "list_fptrunc")

        if is_fp_type(value.type) and is_int_type(target_ty):
            return builder.fptosi(value, target_ty, "list_fptosi")

        raise TypeError(
            f"LiteralList: cannot coerce {value.type} to {target_ty}"
        )

    def _mark_set_value(self, value: ir.Value) -> ir.Value:
        """
        title: Mark set value.
        parameters:
          value:
            type: ir.Value
        returns:
          type: ir.Value
        """
        self._set_value_ids[id(value)] = value
        return value

    def _is_set_value(self, value: ir.Value | None) -> bool:
        """
        title: Is set value.
        parameters:
          value:
            type: ir.Value | None
        returns:
          type: bool
        """
        if value is None:
            return False
        return self._set_value_ids.get(id(value)) is value

    def _try_set_binary_op(
        self,
        lhs: ir.Value | None,
        rhs: ir.Value | None,
        op_code: str,
    ) -> bool:
        """
        title: Try set binary op.
        parameters:
          lhs:
            type: ir.Value | None
          rhs:
            type: ir.Value | None
          op_code:
            type: str
        returns:
          type: bool
        """
        if op_code not in ("|", "&", "-", "^"):
            return False

        if not (self._is_set_value(lhs) and self._is_set_value(rhs)):
            return False

        if not (
            isinstance(lhs, ir.Constant)
            and isinstance(lhs.type, ir.ArrayType)
            and isinstance(lhs.type.element, ir.IntType)
            and isinstance(rhs, ir.Constant)
            and isinstance(rhs.type, ir.ArrayType)
            and isinstance(rhs.type.element, ir.IntType)
        ):
            return False

        lhs_values: set[int] = {element.constant for element in lhs.constant}
        rhs_values: set[int] = {element.constant for element in rhs.constant}

        if op_code == "|":
            result_vals = lhs_values | rhs_values
        elif op_code == "&":
            result_vals = lhs_values & rhs_values
        elif op_code == "-":
            result_vals = lhs_values - rhs_values
        else:
            result_vals = lhs_values ^ rhs_values

        widest = max(lhs.type.element.width, rhs.type.element.width)
        elem_ty = ir.IntType(widest)
        consts = [ir.Constant(elem_ty, value) for value in sorted(result_vals)]
        arr_ty = ir.ArrayType(elem_ty, len(consts))
        self.result_stack.append(
            self._mark_set_value(ir.Constant(arr_ty, consts))
        )
        return True

    def _subscript_uses_unsigned_semantics(
        self, node: astx.SubscriptExpr
    ) -> bool:
        """
        title: Subscript uses unsigned semantics.
        parameters:
          node:
            type: astx.SubscriptExpr
        returns:
          type: bool
        """
        if uses_unsigned_semantics(node.index):
            return True

        if isinstance(node.value, astx.LiteralDict) and node.value.elements:
            first_key_node = next(iter(node.value.elements))
            return uses_unsigned_semantics(first_key_node)

        return False

    def _subscript_compare_type(
        self, lhs_ty: ir.Type, rhs_ty: ir.Type
    ) -> ir.Type:
        """
        title: Subscript compare type.
        parameters:
          lhs_ty:
            type: ir.Type
          rhs_ty:
            type: ir.Type
        returns:
          type: ir.Type
        """
        if lhs_ty == rhs_ty:
            if is_int_type(lhs_ty) or is_fp_type(lhs_ty):
                return lhs_ty
            raise TypeError(
                "SubscriptExpr: only integer and floating-point dict keys "
                "are supported"
            )

        if is_int_type(lhs_ty) and is_int_type(rhs_ty):
            return ir.IntType(max(lhs_ty.width, rhs_ty.width))

        if is_fp_type(lhs_ty) and is_fp_type(rhs_ty):
            return self._select_float_type([lhs_ty, rhs_ty])

        raise TypeError(
            "SubscriptExpr: key type "
            f"{rhs_ty} is incompatible with dict key type {lhs_ty}"
        )

    def _coerce_subscript_key_for_compare(
        self,
        key_val: ir.Value,
        compare_ty: ir.Type,
        *,
        unsigned: bool,
    ) -> ir.Value:
        """
        title: Coerce subscript key for compare.
        parameters:
          key_val:
            type: ir.Value
          compare_ty:
            type: ir.Type
          unsigned:
            type: bool
        returns:
          type: ir.Value
        """
        if key_val.type == compare_ty:
            return key_val

        if isinstance(key_val, ir.Constant):
            if is_int_type(key_val.type) and is_int_type(compare_ty):
                return ir.Constant(compare_ty, int(key_val.constant))
            if is_fp_type(key_val.type) and is_fp_type(compare_ty):
                return ir.Constant(compare_ty, float(key_val.constant))

        if (is_int_type(key_val.type) and is_int_type(compare_ty)) or (
            is_fp_type(key_val.type) and is_fp_type(compare_ty)
        ):
            return self._cast_value_to_type(
                key_val, compare_ty, unsigned=unsigned
            )

        raise TypeError(
            "SubscriptExpr: cannot compare dict key type "
            f"{key_val.type} against {compare_ty}"
        )

    def _constant_subscript_key_matches(
        self,
        entry_key: ir.Constant,
        key_val: ir.Constant,
    ) -> bool:
        """
        title: Constant subscript key matches.
        parameters:
          entry_key:
            type: ir.Constant
          key_val:
            type: ir.Constant
        returns:
          type: bool
        """
        compare_ty = self._subscript_compare_type(entry_key.type, key_val.type)
        lhs = cast(
            ir.Constant,
            self._coerce_subscript_key_for_compare(
                entry_key, compare_ty, unsigned=False
            ),
        )
        rhs = cast(
            ir.Constant,
            self._coerce_subscript_key_for_compare(
                key_val, compare_ty, unsigned=False
            ),
        )
        return bool(lhs.constant == rhs.constant)

    def _emit_subscript_miss(self) -> None:
        """
        title: Emit subscript miss.
        """
        builder = self._llvm.ir_builder
        exit_fn = self.require_runtime_symbol("libc", "exit")
        builder.call(exit_fn, [ir.Constant(self._llvm.INT32_TYPE, 1)])
        builder.unreachable()

    def _emit_runtime_subscript_lookup(
        self,
        dict_val: ir.Constant,
        key_val: ir.Value,
        *,
        unsigned: bool,
    ) -> None:
        """
        title: Emit runtime subscript lookup.
        parameters:
          dict_val:
            type: ir.Constant
          key_val:
            type: ir.Value
          unsigned:
            type: bool
        """
        key_type = dict_val.type.element.elements[0]
        compare_ty = self._subscript_compare_type(key_type, key_val.type)
        key_val = self._coerce_subscript_key_for_compare(
            key_val, compare_ty, unsigned=unsigned
        )

        if isinstance(compare_ty, ir.IntType):
            self._emit_integer_subscript_switch(dict_val, key_val, compare_ty)
            return

        if is_fp_type(compare_ty):
            self._emit_float_subscript_select(dict_val, key_val, compare_ty)
            return

        raise TypeError(
            "SubscriptExpr: runtime lookup supports only integer and "
            "floating-point dict keys"
        )

    def _emit_integer_subscript_switch(
        self,
        dict_val: ir.Constant,
        key_val: ir.Value,
        compare_ty: ir.Type,
    ) -> None:
        """
        title: Emit integer subscript switch.
        parameters:
          dict_val:
            type: ir.Constant
          key_val:
            type: ir.Value
          compare_ty:
            type: ir.Type
        """
        builder = self._llvm.ir_builder
        function = builder.function
        count = dict_val.type.count
        val_type = dict_val.type.element.elements[1]

        miss_bb = function.append_basic_block("dict.miss")
        merge_bb = function.append_basic_block("dict.merge")

        case_blocks: list[tuple[ir.Constant, Any]] = []
        for index in range(count):
            entry_key = dict_val.constant[index].constant[0]
            case_key = self._coerce_subscript_key_for_compare(
                entry_key, compare_ty, unsigned=False
            )
            case_bb = function.append_basic_block(f"dict.case.{index}")
            case_blocks.append((cast(ir.Constant, case_key), case_bb))

        switch = builder.switch(key_val, miss_bb)
        for case_key, case_bb in case_blocks:
            switch.add_case(case_key, case_bb)

        with builder.goto_block(miss_bb):
            self._emit_subscript_miss()

        for _, case_bb in case_blocks:
            with builder.goto_block(case_bb):
                builder.branch(merge_bb)

        builder.position_at_end(merge_bb)

        phi = builder.phi(val_type, name="dict.result")
        for index, (_, case_bb) in enumerate(case_blocks):
            entry_val = dict_val.constant[index].constant[1]
            phi.add_incoming(entry_val, case_bb)

        self.result_stack.append(phi)

    def _emit_float_subscript_select(
        self,
        dict_val: ir.Constant,
        key_val: ir.Value,
        compare_ty: ir.Type,
    ) -> None:
        """
        title: Emit float subscript select.
        parameters:
          dict_val:
            type: ir.Constant
          key_val:
            type: ir.Value
          compare_ty:
            type: ir.Type
        """
        builder = self._llvm.ir_builder
        count = dict_val.type.count

        zero = ir.Constant(ir.IntType(32), 0)
        key_field = ir.Constant(ir.IntType(32), 0)
        val_field = ir.Constant(ir.IntType(32), 1)

        current_bb = builder.block
        builder.position_at_start(builder.function.entry_basic_block)
        arr_alloca = builder.alloca(dict_val.type, name="dict_arr")
        builder.position_at_end(current_bb)
        builder.store(dict_val, arr_alloca)

        last_idx = ir.Constant(ir.IntType(32), count - 1)
        last_key_ptr = builder.gep(
            arr_alloca, [zero, last_idx, key_field], inbounds=True
        )
        last_key = builder.load(last_key_ptr, name=f"k_{count - 1}")
        last_key = self._coerce_subscript_key_for_compare(
            last_key, compare_ty, unsigned=False
        )
        matched = builder.fcmp_ordered(
            "==", last_key, key_val, name=f"cmp_{count - 1}"
        )

        last_val_ptr = builder.gep(
            arr_alloca, [zero, last_idx, val_field], inbounds=True
        )
        result: ir.Value = builder.load(last_val_ptr, name=f"v_{count - 1}")

        for index in reversed(range(count - 1)):
            idx = ir.Constant(ir.IntType(32), index)
            key_ptr = builder.gep(
                arr_alloca, [zero, idx, key_field], inbounds=True
            )
            key_i = builder.load(key_ptr, name=f"k_{index}")
            key_i = self._coerce_subscript_key_for_compare(
                key_i, compare_ty, unsigned=False
            )
            cmp = builder.fcmp_ordered(
                "==", key_i, key_val, name=f"cmp_{index}"
            )
            val_ptr = builder.gep(
                arr_alloca, [zero, idx, val_field], inbounds=True
            )
            val_i = builder.load(val_ptr, name=f"v_{index}")
            result = builder.select(cmp, val_i, result, name=f"sel_{index}")
            matched = builder.or_(cmp, matched, name=f"match_{index}")

        found_bb = builder.function.append_basic_block("dict.found")
        miss_bb = builder.function.append_basic_block("dict.miss")
        builder.cbranch(matched, found_bb, miss_bb)

        with builder.goto_block(miss_bb):
            self._emit_subscript_miss()

        builder.position_at_end(found_bb)
        self.result_stack.append(result)

    def _create_string_concat_function(self) -> ir.Function:
        """
        title: Create string concat function.
        returns:
          type: ir.Function
        """
        func_name = "string_concat"
        if func_name in self._llvm.module.globals:
            return cast(ir.Function, self._llvm.module.get_global(func_name))

        func_type = ir.FunctionType(
            self._llvm.ASCII_STRING_TYPE,
            [self._llvm.ASCII_STRING_TYPE, self._llvm.ASCII_STRING_TYPE],
        )
        func = ir.Function(self._llvm.module, func_type, func_name)
        func.linkage = "external"
        return func

    def _create_string_length_function(self) -> ir.Function:
        """
        title: Create string length function.
        returns:
          type: ir.Function
        """
        func_name = "string_length"
        if func_name in self._llvm.module.globals:
            return cast(ir.Function, self._llvm.module.get_global(func_name))

        func_type = ir.FunctionType(
            self._llvm.INT32_TYPE,
            [self._llvm.ASCII_STRING_TYPE],
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
        title: Create string equals function.
        returns:
          type: ir.Function
        """
        func_name = "string_equals"
        if func_name in self._llvm.module.globals:
            return cast(ir.Function, self._llvm.module.get_global(func_name))

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
        title: Create string substring function.
        returns:
          type: ir.Function
        """
        func_name = "string_substring"
        if func_name in self._llvm.module.globals:
            return cast(ir.Function, self._llvm.module.get_global(func_name))

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
        title: Handle string concatenation.
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
        title: Create strcat inline.
        returns:
          type: ir.Function
        """
        func_name = "strcat_inline"
        if func_name in self._llvm.module.globals:
            return cast(ir.Function, self._llvm.module.get_global(func_name))

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

        total_len = builder.add(len1, len2, "total_len")
        total_len = builder.add(
            total_len,
            ir.Constant(self._llvm.INT32_TYPE, 1),
            "total_len_with_null",
        )

        malloc = self._create_malloc_decl()
        total_len_szt = builder.zext(total_len, self._llvm.SIZE_T_TYPE)
        result_ptr = builder.call(malloc, [total_len_szt], "result")

        self._generate_strcpy(builder, result_ptr, func.args[0])
        result_end = builder.gep(result_ptr, [len1], inbounds=True)
        self._generate_strcpy(builder, result_end, func.args[1])

        builder.ret(result_ptr)
        return func

    def _generate_strcpy(
        self,
        builder: ir.IRBuilder,
        dest: ir.Value,
        src: ir.Value,
    ) -> None:
        """
        title: Generate strcpy.
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
        title: Create strcmp inline.
        returns:
          type: ir.Function
        """
        func_name = "strcmp_inline"
        if func_name in self._llvm.module.globals:
            return cast(ir.Function, self._llvm.module.get_global(func_name))

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
        title: Create strlen inline.
        returns:
          type: ir.Function
        """
        func_name = "strlen_inline"
        if func_name in self._llvm.module.globals:
            return cast(ir.Function, self._llvm.module.get_global(func_name))

        func_type = ir.FunctionType(
            self._llvm.INT32_TYPE,
            [self._llvm.INT8_TYPE.as_pointer()],
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
            count_val,
            ir.Constant(self._llvm.INT32_TYPE, 1),
        )
        builder.store(next_count, counter)
        builder.cbranch(is_null, end, loop)

        builder.position_at_start(end)
        builder.ret(count_val)
        return func

    def _handle_string_comparison(
        self,
        lhs: ir.Value,
        rhs: ir.Value,
        op: str,
    ) -> ir.Value:
        """
        title: Handle string comparison.
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
        if op == "!=":
            equals_func = self._create_string_equals_function()
            equals_result = self._llvm.ir_builder.call(
                equals_func, [lhs, rhs], "str_equals"
            )
            return self._llvm.ir_builder.xor(
                equals_result,
                ir.Constant(self._llvm.BOOLEAN_TYPE, 1),
                "str_not_equals",
            )
        raise Exception(f"String comparison operator {op} not implemented")

    def _normalize_int_for_printf(
        self,
        value: ir.Value,
        *,
        unsigned: bool = False,
    ) -> tuple[ir.Value, str]:
        """
        title: Normalize int for printf.
        parameters:
          value:
            type: ir.Value
          unsigned:
            type: bool
        returns:
          type: tuple[ir.Value, str]
        """
        int64_width = 64
        if not is_int_type(value.type):
            raise Exception("Expected integer value")
        width = value.type.width
        if width < int64_width:
            if width == 1 or unsigned:
                arg = self._llvm.ir_builder.zext(value, self._llvm.INT64_TYPE)
                return arg, "%llu"
            arg = self._llvm.ir_builder.sext(value, self._llvm.INT64_TYPE)
            return arg, "%lld"
        if width == int64_width:
            if unsigned:
                return value, "%llu"
            return value, "%lld"
        raise Exception(
            "Casting integers wider than 64 bits to string is not supported"
        )

    def _create_malloc_decl(self) -> ir.Function:
        """
        title: Create malloc decl.
        returns:
          type: ir.Function
        """
        return self.require_runtime_symbol("libc", "malloc")

    def _snprintf_heap(
        self,
        fmt_gv: ir.GlobalVariable,
        args: list[ir.Value],
    ) -> ir.Value:
        """
        title: Snprintf heap.
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

        zero_i32 = ir.Constant(self._llvm.INT32_TYPE, 0)
        min_needed = self._llvm.ir_builder.select(
            self._llvm.ir_builder.icmp_signed("<", needed_i32, zero_i32),
            ir.Constant(self._llvm.INT32_TYPE, 1),
            needed_i32,
        )
        need_plus_1 = self._llvm.ir_builder.add(
            min_needed,
            ir.Constant(self._llvm.INT32_TYPE, 1),
        )
        need_szt = self._llvm.ir_builder.zext(
            need_plus_1, self._llvm.SIZE_T_TYPE
        )
        mem = self._llvm.ir_builder.call(malloc, [need_szt])
        self._llvm.ir_builder.call(snprintf, [mem, need_szt, fmt_ptr, *args])
        return mem

    def _create_snprintf_decl(self) -> ir.Function:
        """
        title: Create snprintf decl.
        returns:
          type: ir.Function
        """
        return self.require_runtime_symbol("libc", "snprintf")

    def _get_or_create_format_global(self, fmt: str) -> ir.GlobalVariable:
        """
        title: Get or create format global.
        parameters:
          fmt:
            type: str
        returns:
          type: ir.GlobalVariable
        """
        name = f"fmt_{abs(hash(fmt))}"
        if name in self._llvm.module.globals:
            return cast(ir.GlobalVariable, self._llvm.module.get_global(name))

        data = bytearray(fmt + "\0", "utf8")
        arr_ty = ir.ArrayType(self._llvm.INT8_TYPE, len(data))
        gv = ir.GlobalVariable(self._llvm.module, arr_ty, name=name)
        gv.linkage = "internal"
        gv.global_constant = True
        gv.initializer = ir.Constant(arr_ty, data)
        return gv

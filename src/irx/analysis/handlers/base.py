"""
title: Shared semantic-analyzer core.
summary: >-
  Hold analyzer state, semantic sidecar helpers, and shared traversal utilities
  that specialized visitor mixins build on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from plum import dispatch

from irx import astx
from irx.analysis.bindings import VisibleBindings
from irx.analysis.context import SemanticContext
from irx.analysis.factories import SemanticEntityFactory
from irx.analysis.module_interfaces import ModuleKey, ParsedModule
from irx.analysis.registry import SemanticRegistry
from irx.analysis.resolved_nodes import (
    CallResolution,
    ResolvedAssignment,
    ResolvedBaseClassFieldAccess,
    ResolvedClassConstruction,
    ResolvedClassFieldAccess,
    ResolvedFieldAccess,
    ResolvedImportBinding,
    ResolvedMethodCall,
    ResolvedModuleMemberAccess,
    ResolvedOperator,
    ResolvedStaticClassFieldAccess,
    ReturnResolution,
    SemanticClass,
    SemanticFlags,
    SemanticFunction,
    SemanticInfo,
    SemanticModule,
    SemanticStruct,
    SemanticSymbol,
)
from irx.analysis.session import CompilationSession
from irx.analysis.types import clone_type
from irx.base.visitors.base import BaseVisitor
from irx.typecheck import typechecked


# Keep the typed helper contract out of the runtime MRO without defining
# type-only classes inside TYPE_CHECKING blocks.
@typechecked
class SemanticVisitorMixinTypingBase:
    """
    title: Type-checking-only base for semantic visitor mixins.
    attributes:
      context:
        type: SemanticContext
      session:
        type: CompilationSession | None
      factory:
        type: SemanticEntityFactory
      registry:
        type: SemanticRegistry
      bindings:
        type: VisibleBindings
    """

    context: SemanticContext
    session: CompilationSession | None
    factory: SemanticEntityFactory
    registry: SemanticRegistry
    bindings: VisibleBindings

    def visit(self, node: astx.AST) -> None:
        """
        title: Visit AST nodes.
        parameters:
          node:
            type: astx.AST
        """
        raise NotImplementedError

    def _semantic(self, node: astx.AST) -> SemanticInfo:
        """
        title: Return one node's semantic sidecar.
        parameters:
          node:
            type: astx.AST
        returns:
          type: SemanticInfo
        """
        raise NotImplementedError

    def _set_type(
        self,
        node: astx.AST,
        type_: astx.DataType | None,
    ) -> astx.DataType | None:
        """
        title: Attach one resolved type to a node.
        parameters:
          node:
            type: astx.AST
          type_:
            type: astx.DataType | None
        returns:
          type: astx.DataType | None
        """
        raise NotImplementedError

    def _set_symbol(
        self,
        node: astx.AST,
        symbol: SemanticSymbol | None,
    ) -> SemanticSymbol | None:
        """
        title: Attach one resolved symbol to a node.
        parameters:
          node:
            type: astx.AST
          symbol:
            type: SemanticSymbol | None
        returns:
          type: SemanticSymbol | None
        """
        raise NotImplementedError

    def _set_function(
        self,
        node: astx.AST,
        function: SemanticFunction | None,
    ) -> SemanticFunction | None:
        """
        title: Attach one resolved function to a node.
        parameters:
          node:
            type: astx.AST
          function:
            type: SemanticFunction | None
        returns:
          type: SemanticFunction | None
        """
        raise NotImplementedError

    def _set_call(
        self,
        node: astx.AST,
        call: CallResolution | None,
    ) -> CallResolution | None:
        """
        title: Attach one resolved call site.
        parameters:
          node:
            type: astx.AST
          call:
            type: CallResolution | None
        returns:
          type: CallResolution | None
        """
        raise NotImplementedError

    def _set_return(
        self,
        node: astx.AST,
        return_resolution: ReturnResolution | None,
    ) -> ReturnResolution | None:
        """
        title: Attach one resolved return site.
        parameters:
          node:
            type: astx.AST
          return_resolution:
            type: ReturnResolution | None
        returns:
          type: ReturnResolution | None
        """
        raise NotImplementedError

    def _set_struct(
        self,
        node: astx.AST,
        struct: SemanticStruct | None,
    ) -> SemanticStruct | None:
        """
        title: Attach one resolved struct to a node.
        parameters:
          node:
            type: astx.AST
          struct:
            type: SemanticStruct | None
        returns:
          type: SemanticStruct | None
        """
        raise NotImplementedError

    def _set_class(
        self,
        node: astx.AST,
        class_: SemanticClass | None,
    ) -> SemanticClass | None:
        """
        title: Attach one resolved class to a node.
        parameters:
          node:
            type: astx.AST
          class_:
            type: SemanticClass | None
        returns:
          type: SemanticClass | None
        """
        raise NotImplementedError

    def _set_module(
        self,
        node: astx.AST,
        module: SemanticModule | None,
    ) -> SemanticModule | None:
        """
        title: Attach one resolved module to a node.
        parameters:
          node:
            type: astx.AST
          module:
            type: SemanticModule | None
        returns:
          type: SemanticModule | None
        """
        raise NotImplementedError

    def _resolve_template_call_target(
        self,
        function: SemanticFunction,
        _arg_types: list[astx.DataType | None],
        node: astx.AST,
    ) -> SemanticFunction | None:
        """
        title: Resolve one template function call target.
        parameters:
          function:
            type: SemanticFunction
          _arg_types:
            type: list[astx.DataType | None]
          node:
            type: astx.AST
        returns:
          type: SemanticFunction | None
        """
        raise NotImplementedError

    def _resolve_template_method_call_target(
        self,
        function: SemanticFunction,
        _visible_function: SemanticFunction,
        _arg_types: list[astx.DataType | None],
        node: astx.AST,
    ) -> SemanticFunction | None:
        """
        title: Resolve one template method call target.
        parameters:
          function:
            type: SemanticFunction
          _visible_function:
            type: SemanticFunction
          _arg_types:
            type: list[astx.DataType | None]
          node:
            type: astx.AST
        returns:
          type: SemanticFunction | None
        """
        raise NotImplementedError

    def _specialization_bindings_map(
        self,
        function: SemanticFunction,
    ) -> dict[str, astx.DataType]:
        """
        title: Return one specialization binding map.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: dict[str, astx.DataType]
        """
        raise NotImplementedError

    def _specialize_signature(
        self,
        function: SemanticFunction,
        bindings: dict[str, astx.DataType],
    ) -> SemanticFunction:
        """
        title: Return one callable wrapper with substituted types.
        parameters:
          function:
            type: SemanticFunction
          bindings:
            type: dict[str, astx.DataType]
        returns:
          type: SemanticFunction
        """
        raise NotImplementedError

    def _prepare_function_template_specializations(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Prepare concrete specializations for one template function.
        parameters:
          function:
            type: SemanticFunction
        """
        raise NotImplementedError

    def _analyze_function_template_specializations(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Analyze prepared specializations for one template function.
        parameters:
          function:
            type: SemanticFunction
        """
        raise NotImplementedError

    def _set_imports(
        self,
        node: astx.AST,
        imports: tuple[ResolvedImportBinding, ...],
    ) -> None:
        """
        title: Attach resolved imports to a node.
        parameters:
          node:
            type: astx.AST
          imports:
            type: tuple[ResolvedImportBinding, Ellipsis]
        """
        raise NotImplementedError

    def _set_flags(self, node: astx.AST, flags: SemanticFlags) -> None:
        """
        title: Attach semantic flags to a node.
        parameters:
          node:
            type: astx.AST
          flags:
            type: SemanticFlags
        """
        raise NotImplementedError

    def _set_operator(
        self,
        node: astx.AST,
        operator: ResolvedOperator | None,
    ) -> None:
        """
        title: Attach one resolved operator to a node.
        parameters:
          node:
            type: astx.AST
          operator:
            type: ResolvedOperator | None
        """
        raise NotImplementedError

    def _set_assignment(
        self,
        node: astx.AST,
        symbol: SemanticSymbol | None,
    ) -> None:
        """
        title: Attach one resolved assignment target to a node.
        parameters:
          node:
            type: astx.AST
          symbol:
            type: SemanticSymbol | None
        """
        raise NotImplementedError

    def _set_field_access(
        self,
        node: astx.AST,
        field_access: ResolvedFieldAccess | None,
    ) -> None:
        """
        title: Attach resolved field access metadata.
        parameters:
          node:
            type: astx.AST
          field_access:
            type: ResolvedFieldAccess | None
        """
        raise NotImplementedError

    def _set_module_member_access(
        self,
        node: astx.AST,
        field_access: ResolvedModuleMemberAccess | None,
    ) -> None:
        """
        title: Attach resolved module-member access metadata.
        parameters:
          node:
            type: astx.AST
          field_access:
            type: ResolvedModuleMemberAccess | None
        """
        raise NotImplementedError

    def _set_class_field_access(
        self,
        node: astx.AST,
        field_access: ResolvedClassFieldAccess | None,
    ) -> None:
        """
        title: Attach resolved class-field access metadata.
        parameters:
          node:
            type: astx.AST
          field_access:
            type: ResolvedClassFieldAccess | None
        """
        raise NotImplementedError

    def _set_base_class_field_access(
        self,
        node: astx.AST,
        field_access: ResolvedBaseClassFieldAccess | None,
    ) -> None:
        """
        title: Attach resolved base-class field access metadata.
        parameters:
          node:
            type: astx.AST
          field_access:
            type: ResolvedBaseClassFieldAccess | None
        """
        raise NotImplementedError

    def _set_static_class_field_access(
        self,
        node: astx.AST,
        field_access: ResolvedStaticClassFieldAccess | None,
    ) -> None:
        """
        title: Attach resolved static class-field access metadata.
        parameters:
          node:
            type: astx.AST
          field_access:
            type: ResolvedStaticClassFieldAccess | None
        """
        raise NotImplementedError

    def _set_method_call(
        self,
        node: astx.AST,
        method_call: ResolvedMethodCall | None,
    ) -> None:
        """
        title: Attach resolved method-call metadata.
        parameters:
          node:
            type: astx.AST
          method_call:
            type: ResolvedMethodCall | None
        """
        raise NotImplementedError

    def _set_class_construction(
        self,
        node: astx.AST,
        construction: ResolvedClassConstruction | None,
    ) -> None:
        """
        title: Attach resolved class-construction metadata.
        parameters:
          node:
            type: astx.AST
          construction:
            type: ResolvedClassConstruction | None
        """
        raise NotImplementedError

    def _resolve_struct_from_type(
        self,
        type_: astx.DataType | None,
        *,
        node: astx.AST,
        unknown_message: str,
    ) -> SemanticStruct | None:
        """
        title: Resolve one struct-valued type reference.
        parameters:
          type_:
            type: astx.DataType | None
          node:
            type: astx.AST
          unknown_message:
            type: str
        returns:
          type: SemanticStruct | None
        """
        raise NotImplementedError

    def _resolve_class_from_type(
        self,
        type_: astx.DataType | None,
        *,
        node: astx.AST,
        unknown_message: str,
    ) -> SemanticClass | None:
        """
        title: Resolve one class-valued type reference.
        parameters:
          type_:
            type: astx.DataType | None
          node:
            type: astx.AST
          unknown_message:
            type: str
        returns:
          type: SemanticClass | None
        """
        raise NotImplementedError

    def _resolve_declared_type(
        self,
        type_: astx.DataType,
        *,
        node: astx.AST,
        unknown_message: str = "Unknown type '{name}'",
    ) -> astx.DataType:
        """
        title: Resolve one declared type in place.
        parameters:
          type_:
            type: astx.DataType
          node:
            type: astx.AST
          unknown_message:
            type: str
        returns:
          type: astx.DataType
        """
        raise NotImplementedError

    def _root_assignment_symbol(
        self,
        node: astx.AST | None,
    ) -> SemanticSymbol | None:
        """
        title: Resolve the root symbol for an assignment target chain.
        parameters:
          node:
            type: astx.AST | None
        returns:
          type: SemanticSymbol | None
        """
        raise NotImplementedError

    def _predeclare_block_structs(self, block: astx.Block) -> None:
        """
        title: Predeclare composite definitions in one block.
        parameters:
          block:
            type: astx.Block
        """
        raise NotImplementedError

    def _current_module_key(self) -> ModuleKey:
        """
        title: Return the current module key.
        returns:
          type: ModuleKey
        """
        raise NotImplementedError

    def _imports_supported_here(self, node: astx.AST) -> bool:
        """
        title: Return True when imports are allowed here.
        parameters:
          node:
            type: astx.AST
        returns:
          type: bool
        """
        raise NotImplementedError

    def _expr_type(self, node: astx.AST | None) -> astx.DataType | None:
        """
        title: Return one node's resolved expression type.
        parameters:
          node:
            type: astx.AST | None
        returns:
          type: astx.DataType | None
        """
        raise NotImplementedError

    def _require_value_expression(
        self,
        node: astx.AST | None,
        *,
        context: str,
    ) -> bool:
        """
        title: Require one expression context to receive a non-void value.
        parameters:
          node:
            type: astx.AST | None
          context:
            type: str
        returns:
          type: bool
        """
        raise NotImplementedError

    def _visit_module(
        self,
        module: astx.Module,
        *,
        predeclared: bool,
    ) -> None:
        """
        title: Visit one module with optional predeclaration.
        parameters:
          module:
            type: astx.Module
          predeclared:
            type: bool
        """
        raise NotImplementedError

    def _guarantees_return(self, node: astx.AST) -> bool:
        """
        title: Return whether a statement subtree guarantees return.
        parameters:
          node:
            type: astx.AST
        returns:
          type: bool
        """
        raise NotImplementedError


@typechecked
class SemanticVisitorMixinRuntimeBase:
    """
    title: Runtime-empty base for semantic visitor mixins.
    """


if TYPE_CHECKING:
    SemanticVisitorMixinBase = SemanticVisitorMixinTypingBase
else:
    SemanticVisitorMixinBase = SemanticVisitorMixinRuntimeBase


@typechecked
class SemanticAnalyzerCore(BaseVisitor):
    """
    title: Shared semantic analyzer core.
    summary: >-
      Provide shared semantic-analysis state and helper behavior for the
      specialized visitor mixins that implement node-specific rules.
    attributes:
      context:
        type: SemanticContext
      session:
        type: CompilationSession | None
      factory:
        type: SemanticEntityFactory
      registry:
        type: SemanticRegistry
      bindings:
        type: VisibleBindings
    """

    context: SemanticContext
    session: CompilationSession | None
    factory: SemanticEntityFactory
    registry: SemanticRegistry
    bindings: VisibleBindings

    def __init__(
        self,
        *,
        context: SemanticContext | None = None,
        session: CompilationSession | None = None,
    ) -> None:
        """
        title: Initialize SemanticAnalyzerCore.
        parameters:
          context:
            type: SemanticContext | None
          session:
            type: CompilationSession | None
        """
        self.session = session
        self.context = context or SemanticContext()
        if session is not None:
            self.context.diagnostics = session.diagnostics
        self.factory = SemanticEntityFactory(self.context)
        self.registry = SemanticRegistry(self.context, self.factory)
        self.bindings = VisibleBindings(
            context=self.context,
            factory=self.factory,
            bindings=(
                session.visible_bindings if session is not None else None
            ),
        )

    def analyze(self, node: astx.AST) -> astx.AST:
        """
        title: Analyze one AST root.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.AST
        """
        if isinstance(node, astx.Module):
            parsed_module = ParsedModule(node.name, node)
            self.analyze_parsed_module(parsed_module, predeclared=False)
        else:
            with self.context.scope("module"):
                self.visit(node)
        self.context.diagnostics.raise_if_errors()
        return node

    def analyze_parsed_module(
        self,
        parsed_module: ParsedModule,
        *,
        predeclared: bool,
    ) -> astx.Module:
        """
        title: Analyze one parsed module.
        parameters:
          parsed_module:
            type: ParsedModule
          predeclared:
            type: bool
        returns:
          type: astx.Module
        """
        with self.context.in_module(parsed_module.key):
            self._visit_module(parsed_module.ast, predeclared=predeclared)
        return parsed_module.ast

    def _semantic(self, node: astx.AST) -> SemanticInfo:
        """
        title: Return one node's semantic sidecar.
        parameters:
          node:
            type: astx.AST
        returns:
          type: SemanticInfo
        """
        info = cast(SemanticInfo | None, getattr(node, "semantic", None))
        if info is None or not isinstance(info, SemanticInfo):
            info = SemanticInfo()
            setattr(node, "semantic", info)
        return info

    def _set_type(
        self, node: astx.AST, type_: astx.DataType | None
    ) -> astx.DataType | None:
        """
        title: Attach one resolved type to a node.
        parameters:
          node:
            type: astx.AST
          type_:
            type: astx.DataType | None
        returns:
          type: astx.DataType | None
        """
        info = self._semantic(node)
        info.resolved_type = type_
        if type_ is not None and hasattr(node, "type_"):
            try:
                setattr(node, "type_", clone_type(type_))
            except Exception:
                pass
        return type_

    def _set_symbol(
        self, node: astx.AST, symbol: SemanticSymbol | None
    ) -> SemanticSymbol | None:
        """
        title: Attach one resolved symbol to a node.
        parameters:
          node:
            type: astx.AST
          symbol:
            type: SemanticSymbol | None
        returns:
          type: SemanticSymbol | None
        """
        info = self._semantic(node)
        info.resolved_symbol = symbol
        if symbol is not None:
            self._set_type(node, symbol.type_)
        return symbol

    def _set_function(
        self, node: astx.AST, function: SemanticFunction | None
    ) -> SemanticFunction | None:
        """
        title: Attach one resolved function to a node.
        parameters:
          node:
            type: astx.AST
          function:
            type: SemanticFunction | None
        returns:
          type: SemanticFunction | None
        """
        info = self._semantic(node)
        info.resolved_function = function
        if function is not None:
            info.resolved_callable = self.factory.make_callable_resolution(
                function
            )
            self._set_type(node, function.return_type)
        else:
            info.resolved_callable = None
        return function

    def _set_call(
        self,
        node: astx.AST,
        call: CallResolution | None,
    ) -> CallResolution | None:
        """
        title: Attach one resolved call site.
        parameters:
          node:
            type: astx.AST
          call:
            type: CallResolution | None
        returns:
          type: CallResolution | None
        """
        info = self._semantic(node)
        info.resolved_call = call
        if call is not None:
            info.resolved_callable = call.callee
        return call

    def _set_return(
        self,
        node: astx.AST,
        return_resolution: ReturnResolution | None,
    ) -> ReturnResolution | None:
        """
        title: Attach one resolved return site.
        parameters:
          node:
            type: astx.AST
          return_resolution:
            type: ReturnResolution | None
        returns:
          type: ReturnResolution | None
        """
        info = self._semantic(node)
        info.resolved_return = return_resolution
        return return_resolution

    def _set_struct(
        self,
        node: astx.AST,
        struct: SemanticStruct | None,
    ) -> SemanticStruct | None:
        """
        title: Attach one resolved struct to a node.
        parameters:
          node:
            type: astx.AST
          struct:
            type: SemanticStruct | None
        returns:
          type: SemanticStruct | None
        """
        info = self._semantic(node)
        info.resolved_struct = struct
        return struct

    def _set_class(
        self,
        node: astx.AST,
        class_: SemanticClass | None,
    ) -> SemanticClass | None:
        """
        title: Attach one resolved class to a node.
        parameters:
          node:
            type: astx.AST
          class_:
            type: SemanticClass | None
        returns:
          type: SemanticClass | None
        """
        info = self._semantic(node)
        info.resolved_class = class_
        return class_

    def _set_module(
        self,
        node: astx.AST,
        module: SemanticModule | None,
    ) -> SemanticModule | None:
        """
        title: Attach one resolved module to a node.
        parameters:
          node:
            type: astx.AST
          module:
            type: SemanticModule | None
        returns:
          type: SemanticModule | None
        """
        info = self._semantic(node)
        info.resolved_module = module
        return module

    def _set_imports(
        self,
        node: astx.AST,
        imports: tuple[ResolvedImportBinding, ...],
    ) -> None:
        """
        title: Attach resolved imports to a node.
        parameters:
          node:
            type: astx.AST
          imports:
            type: tuple[ResolvedImportBinding, Ellipsis]
        """
        self._semantic(node).resolved_imports = imports

    def _set_flags(self, node: astx.AST, flags: SemanticFlags) -> None:
        """
        title: Attach semantic flags to a node.
        parameters:
          node:
            type: astx.AST
          flags:
            type: SemanticFlags
        """
        self._semantic(node).semantic_flags = flags

    def _set_operator(
        self,
        node: astx.AST,
        operator: ResolvedOperator | None,
    ) -> None:
        """
        title: Attach one resolved operator to a node.
        parameters:
          node:
            type: astx.AST
          operator:
            type: ResolvedOperator | None
        """
        self._semantic(node).resolved_operator = operator

    def _set_assignment(
        self, node: astx.AST, symbol: SemanticSymbol | None
    ) -> None:
        """
        title: Attach one resolved assignment target to a node.
        parameters:
          node:
            type: astx.AST
          symbol:
            type: SemanticSymbol | None
        """
        info = self._semantic(node)
        if symbol is None:
            info.resolved_assignment = None
            return
        info.resolved_assignment = ResolvedAssignment(symbol)

    def _set_field_access(
        self,
        node: astx.AST,
        field_access: ResolvedFieldAccess | None,
    ) -> None:
        """
        title: Attach resolved field access metadata.
        parameters:
          node:
            type: astx.AST
          field_access:
            type: ResolvedFieldAccess | None
        """
        self._semantic(node).resolved_field_access = field_access

    def _set_module_member_access(
        self,
        node: astx.AST,
        field_access: ResolvedModuleMemberAccess | None,
    ) -> None:
        """
        title: Attach resolved module-member access metadata.
        parameters:
          node:
            type: astx.AST
          field_access:
            type: ResolvedModuleMemberAccess | None
        """
        self._semantic(node).resolved_module_member_access = field_access

    def _set_class_field_access(
        self,
        node: astx.AST,
        field_access: ResolvedClassFieldAccess | None,
    ) -> None:
        """
        title: Attach resolved class-field access metadata.
        parameters:
          node:
            type: astx.AST
          field_access:
            type: ResolvedClassFieldAccess | None
        """
        self._semantic(node).resolved_class_field_access = field_access

    def _set_base_class_field_access(
        self,
        node: astx.AST,
        field_access: ResolvedBaseClassFieldAccess | None,
    ) -> None:
        """
        title: Attach resolved base-class field access metadata.
        parameters:
          node:
            type: astx.AST
          field_access:
            type: ResolvedBaseClassFieldAccess | None
        """
        self._semantic(node).resolved_base_class_field_access = field_access

    def _set_static_class_field_access(
        self,
        node: astx.AST,
        field_access: ResolvedStaticClassFieldAccess | None,
    ) -> None:
        """
        title: Attach resolved static class-field access metadata.
        parameters:
          node:
            type: astx.AST
          field_access:
            type: ResolvedStaticClassFieldAccess | None
        """
        self._semantic(node).resolved_static_class_field_access = field_access

    def _set_method_call(
        self,
        node: astx.AST,
        method_call: ResolvedMethodCall | None,
    ) -> None:
        """
        title: Attach resolved method-call metadata.
        parameters:
          node:
            type: astx.AST
          method_call:
            type: ResolvedMethodCall | None
        """
        self._semantic(node).resolved_method_call = method_call

    def _set_class_construction(
        self,
        node: astx.AST,
        construction: ResolvedClassConstruction | None,
    ) -> None:
        """
        title: Attach resolved class-construction metadata.
        parameters:
          node:
            type: astx.AST
          construction:
            type: ResolvedClassConstruction | None
        """
        self._semantic(node).resolved_class_construction = construction

    def _resolve_struct_from_type(
        self,
        type_: astx.DataType | None,
        *,
        node: astx.AST,
        unknown_message: str,
    ) -> SemanticStruct | None:
        """
        title: Resolve one struct-valued type reference.
        parameters:
          type_:
            type: astx.DataType | None
          node:
            type: astx.AST
          unknown_message:
            type: str
        returns:
          type: SemanticStruct | None
        """
        if not isinstance(type_, astx.StructType):
            return None

        binding = self.bindings.resolve(type_.name)
        struct = (
            binding.struct
            if binding is not None and binding.kind == "struct"
            else None
        )
        if struct is None and type_.module_key is not None:
            lookup_name = type_.resolved_name or type_.name
            struct = self.context.get_struct(type_.module_key, lookup_name)
        if struct is None:
            self.context.diagnostics.add(
                unknown_message.format(name=type_.name),
                node=node,
            )
            return None

        type_.resolved_name = struct.name
        type_.module_key = struct.module_key
        type_.qualified_name = struct.qualified_name
        self._set_struct(type_, struct)
        self._set_type(type_, type_)
        return struct

    def _resolve_class_from_type(
        self,
        type_: astx.DataType | None,
        *,
        node: astx.AST,
        unknown_message: str,
    ) -> SemanticClass | None:
        """
        title: Resolve one class-valued type reference.
        parameters:
          type_:
            type: astx.DataType | None
          node:
            type: astx.AST
          unknown_message:
            type: str
        returns:
          type: SemanticClass | None
        """
        if not isinstance(type_, astx.ClassType):
            return None

        binding = self.bindings.resolve(type_.name)
        class_ = (
            binding.class_
            if binding is not None and binding.kind == "class"
            else None
        )
        if class_ is None and type_.module_key is not None:
            lookup_name = type_.resolved_name or type_.name
            class_ = self.context.get_class(type_.module_key, lookup_name)
        if class_ is None:
            self.context.diagnostics.add(
                unknown_message.format(name=type_.name),
                node=node,
            )
            return None

        resolve_structure = getattr(self, "_resolve_class_structure", None)
        active_stack: list[SemanticClass] = cast(
            list[SemanticClass],
            getattr(self, "_class_resolution_stack", lambda: [])(),
        )
        if (
            callable(resolve_structure)
            and not class_.is_structurally_resolved
            and not any(
                item.qualified_name == class_.qualified_name
                for item in active_stack
            )
        ):
            class_ = resolve_structure(class_)

        type_.resolved_name = class_.name
        type_.module_key = class_.module_key
        type_.qualified_name = class_.qualified_name
        type_.ancestor_qualified_names = tuple(
            ancestor.qualified_name for ancestor in class_.mro[1:]
        )
        self._set_class(type_, class_)
        self._set_type(type_, type_)
        return class_

    def _resolve_declared_type(
        self,
        type_: astx.DataType,
        *,
        node: astx.AST,
        unknown_message: str = "Unknown type '{name}'",
    ) -> astx.DataType:
        """
        title: Resolve one declared type in place.
        parameters:
          type_:
            type: astx.DataType
          node:
            type: astx.AST
          unknown_message:
            type: str
        returns:
          type: astx.DataType
        """
        if isinstance(type_, astx.UnionType):
            for member in type_.members:
                self._resolve_declared_type(
                    member,
                    node=node,
                    unknown_message=unknown_message,
                )
            self._set_type(type_, type_)
            return type_
        if isinstance(type_, astx.TemplateTypeVar):
            self._resolve_declared_type(
                type_.bound,
                node=node,
                unknown_message=unknown_message,
            )
            self._set_type(type_, type_)
            return type_
        self._resolve_struct_from_type(
            type_,
            node=node,
            unknown_message=unknown_message,
        )
        self._resolve_class_from_type(
            type_,
            node=node,
            unknown_message=unknown_message,
        )
        self._set_type(type_, type_)
        return type_

    def _root_assignment_symbol(
        self,
        node: astx.AST | None,
    ) -> SemanticSymbol | None:
        """
        title: Resolve the root symbol for an assignment target chain.
        parameters:
          node:
            type: astx.AST | None
        returns:
          type: SemanticSymbol | None
        """
        if node is None:
            return None
        if isinstance(node, astx.Identifier):
            return cast(
                SemanticInfo,
                getattr(node, "semantic", SemanticInfo()),
            ).resolved_symbol
        if isinstance(node, astx.FieldAccess):
            return self._root_assignment_symbol(node.value)
        return None

    def _predeclare_block_structs(self, block: astx.Block) -> None:
        """
        title: Predeclare composite definitions in one block.
        parameters:
          block:
            type: astx.Block
        """
        for node in block.nodes:
            if isinstance(node, astx.StructDefStmt):
                struct = self.registry.register_struct(node)
                self.bindings.bind_struct(node.name, struct, node=node)
                self._set_struct(node, struct)
                continue
            if not isinstance(node, astx.ClassDefStmt):
                continue
            class_ = self.registry.register_class(node)
            self.bindings.bind_class(node.name, class_, node=node)
            self._set_class(node, class_)

    def _current_module_key(self) -> ModuleKey:
        """
        title: Return the current module key.
        returns:
          type: ModuleKey
        """
        return self.context.current_module_key or "<root>"

    def _imports_supported_here(self, node: astx.AST) -> bool:
        """
        title: Return True when imports are allowed here.
        parameters:
          node:
            type: astx.AST
        returns:
          type: bool
        """
        if self.session is None:
            self.context.diagnostics.add(
                "Import statements require analyze_modules(...) with a "
                "host-provided resolver.",
                node=node,
            )
            return False
        current_scope = self.context.scopes.current
        if current_scope is None or current_scope.kind != "module":
            self.context.diagnostics.add(
                "Import statements are supported only at module top level "
                "in this MVP.",
                node=node,
            )
            return False
        return True

    def _expr_type(self, node: astx.AST | None) -> astx.DataType | None:
        """
        title: Return one node's resolved expression type.
        parameters:
          node:
            type: astx.AST | None
        returns:
          type: astx.DataType | None
        """
        if node is None:
            return None
        info = cast(SemanticInfo | None, getattr(node, "semantic", None))
        if info is not None and info.resolved_type is not None:
            return info.resolved_type
        return getattr(node, "type_", None)

    def _require_value_expression(
        self,
        node: astx.AST | None,
        *,
        context: str,
    ) -> bool:
        """
        title: Require one expression context to receive a non-void value.
        parameters:
          node:
            type: astx.AST | None
          context:
            type: str
        returns:
          type: bool
        """
        if node is None:
            return True
        resolved_type = self._expr_type(node)
        if not isinstance(resolved_type, astx.NoneType):
            return True
        semantic = cast(SemanticInfo | None, getattr(node, "semantic", None))
        if semantic is not None and semantic.resolved_call is not None:
            self.context.diagnostics.add(
                f"{context} cannot use the result of void call as a value",
                node=node,
            )
            return False
        self.context.diagnostics.add(
            f"{context} requires a non-void value",
            node=node,
        )
        return False

    def _predeclare_module_members(self, module: astx.Module) -> None:
        """
        title: Predeclare one module's top-level members.
        parameters:
          module:
            type: astx.Module
        """
        for node in module.nodes:
            if isinstance(node, astx.FunctionPrototype):
                setattr(node, "irx_owner_module", module)
                function = self.registry.register_function(
                    node,
                    validate_ffi=False,
                )
                self.bindings.bind_function(node.name, function, node=node)
                self._set_function(node, function)
            elif isinstance(node, astx.FunctionDef):
                setattr(node, "irx_owner_module", module)
                setattr(node.prototype, "irx_owner_module", module)
                function = self.registry.register_function(
                    node.prototype,
                    definition=node,
                    validate_ffi=False,
                )
                self.bindings.bind_function(
                    node.prototype.name,
                    function,
                    node=node,
                )
                self._set_function(node.prototype, function)
                self._set_function(node, function)
            elif isinstance(node, astx.StructDefStmt):
                struct = self.registry.register_struct(node)
                self.bindings.bind_struct(node.name, struct, node=node)
                self._set_struct(node, struct)
            elif isinstance(node, astx.ClassDefStmt):
                class_ = self.registry.register_class(node)
                self.bindings.bind_class(node.name, class_, node=node)
                self._set_class(node, class_)

    def _visit_module(
        self,
        module: astx.Module,
        *,
        predeclared: bool,
    ) -> None:
        """
        title: Visit one module with optional predeclaration.
        parameters:
          module:
            type: astx.Module
          predeclared:
            type: bool
        """
        self._set_type(module, None)
        previous_module = getattr(self, "_current_ast_module", None)
        setattr(self, "_current_ast_module", module)
        try:
            with self.context.scope("module"):
                if not predeclared:
                    self._predeclare_module_members(module)
                    prepare_templates = getattr(
                        self,
                        "_prepare_template_specialization_skeletons",
                        None,
                    )
                    if callable(prepare_templates):
                        prepare_templates(module)
                    analyze_templates = getattr(
                        self,
                        "_analyze_prepared_template_specializations",
                        None,
                    )
                    if callable(analyze_templates):
                        analyze_templates(module)
                for node in module.nodes:
                    self.visit(node)
        finally:
            setattr(self, "_current_ast_module", previous_module)
        main_function = self.context.get_function(
            self._current_module_key(),
            "main",
        )
        if main_function is not None and main_function.definition is None:
            self.context.diagnostics.add(
                "Function 'main' must have a definition",
                node=module,
            )

    def _visit_plain_typed_node(self, node: astx.AST) -> None:
        """
        title: Visit one plain typed node.
        parameters:
          node:
            type: astx.AST
        """
        self._set_type(node, getattr(node, "type_", None))

    def _guarantees_return(self, node: astx.AST) -> bool:
        """
        title: Return whether a statement subtree guarantees return.
        parameters:
          node:
            type: astx.AST
        returns:
          type: bool
        """
        if isinstance(node, astx.FunctionReturn):
            return True
        if isinstance(node, astx.Block):
            for child in node.nodes:
                if self._guarantees_return(child):
                    return True
            return False
        if isinstance(node, astx.IfStmt):
            if node.else_ is None:
                return False
            return self._guarantees_return(
                node.then
            ) and self._guarantees_return(node.else_)
        return False

    @dispatch
    def visit(self, node: astx.AST) -> None:
        """
        title: Visit generic AST nodes.
        parameters:
          node:
            type: astx.AST
        """
        self._visit_plain_typed_node(node)

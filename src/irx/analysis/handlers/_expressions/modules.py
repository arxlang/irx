# mypy: disable-error-code=no-redef
# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=untyped-decorator

"""
Expression helpers for module namespaces and function calls.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.resolved_nodes import (
    ResolvedModuleMemberAccess,
    SemanticBinding,
    SemanticModule,
)
from irx.analysis.validation import validate_call
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ExpressionModuleVisitorMixin(SemanticVisitorMixinBase):
    """
    title: Expression helpers for module namespaces and function calls
    """

    def _module_namespace_type(
        self,
        module: SemanticModule,
    ) -> astx.NamespaceType:
        """
        title: Build one semantic-only namespace type for a module.
        parameters:
          module:
            type: SemanticModule
        returns:
          type: astx.NamespaceType
        """
        return astx.NamespaceType(
            module.module_key,
            namespace_kind=astx.NamespaceKind.MODULE,
            display_name=module.display_name,
        )

    def _module_namespace_from_type(
        self,
        type_: astx.DataType | None,
    ) -> SemanticModule | None:
        """
        title: Reconstruct module identity from a namespace type.
        parameters:
          type_:
            type: astx.DataType | None
        returns:
          type: SemanticModule | None
        """
        if not isinstance(type_, astx.NamespaceType):
            return None
        if type_.namespace_kind is not astx.NamespaceKind.MODULE:
            return None
        return self.factory.make_module(
            type_.namespace_key,
            display_name=type_.display_name,
        )

    def _module_namespace(
        self,
        node: astx.AST,
    ) -> SemanticModule | None:
        """
        title: Return the resolved module namespace for one expression.
        parameters:
          node:
            type: astx.AST
        returns:
          type: SemanticModule | None
        """
        semantic = getattr(node, "semantic", None)
        module = getattr(semantic, "resolved_module", None)
        if isinstance(module, SemanticModule):
            return module
        return self._module_namespace_from_type(self._expr_type(node))

    def _module_namespace_name(
        self,
        node: astx.AST,
        module: SemanticModule | None = None,
    ) -> str:
        """
        title: Return one stable display name for a namespace expression.
        parameters:
          node:
            type: astx.AST
          module:
            type: SemanticModule | None
        returns:
          type: str
        """
        if isinstance(node, astx.Identifier):
            return cast(str, node.name)
        if isinstance(node, astx.FieldAccess):
            base_name = self._module_namespace_name(node.value, module)
            return f"{base_name}.{node.field_name}"
        resolved_module = self._module_namespace(node)
        if resolved_module is not None:
            return resolved_module.display_name or str(
                resolved_module.module_key
            )
        if module is not None:
            return module.display_name or str(module.module_key)
        return "<module>"

    def _resolve_module_member_binding(
        self,
        module: SemanticModule,
        member_name: str,
        *,
        node: astx.AST,
        namespace_name: str,
    ) -> SemanticBinding | None:
        """
        title: Resolve one visible member through a module namespace.
        parameters:
          module:
            type: SemanticModule
          member_name:
            type: str
          node:
            type: astx.AST
          namespace_name:
            type: str
        returns:
          type: SemanticBinding | None
        """
        binding = self.bindings.resolve(
            member_name,
            module_key=module.module_key,
        )
        if binding is not None:
            return binding
        self.context.diagnostics.add(
            (
                f"module namespace '{namespace_name}' has no member "
                f"'{member_name}'"
            ),
            node=node,
            code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
        )
        return None

    def _set_module_member_resolution(
        self,
        node: astx.AST,
        *,
        module: SemanticModule,
        member_name: str,
        binding: SemanticBinding,
    ) -> None:
        """
        title: Attach resolved module-member semantics to one node.
        parameters:
          node:
            type: astx.AST
          module:
            type: SemanticModule
          member_name:
            type: str
          binding:
            type: SemanticBinding
        """
        self._set_module_member_access(
            node,
            ResolvedModuleMemberAccess(
                module=module,
                member_name=member_name,
                binding=binding,
            ),
        )
        if binding.function is not None:
            self._set_function(node, binding.function)
            return
        if binding.module is not None:
            self._set_module(node, binding.module)
            self._set_type(
                node,
                self._module_namespace_type(binding.module),
            )
            return
        if binding.struct is not None:
            self._set_struct(node, binding.struct)
        if binding.class_ is not None:
            self._set_class(node, binding.class_)

    def _resolve_module_namespace_call(
        self,
        node: astx.MethodCall,
        module: SemanticModule,
        arg_types: list[astx.DataType | None],
    ) -> None:
        """
        title: Resolve one callable lookup through a module namespace.
        parameters:
          node:
            type: astx.MethodCall
          module:
            type: SemanticModule
          arg_types:
            type: list[astx.DataType | None]
        """
        namespace_name = self._module_namespace_name(node.receiver, module)
        binding = self._resolve_module_member_binding(
            module,
            node.method_name,
            node=node,
            namespace_name=namespace_name,
        )
        if binding is None:
            self._set_type(node, None)
            return
        self._set_module_member_resolution(
            node,
            module=module,
            member_name=node.method_name,
            binding=binding,
        )
        function = binding.function
        if function is None:
            self.context.diagnostics.add(
                (
                    f"module namespace '{namespace_name}' member "
                    f"'{node.method_name}' is not callable"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            self._set_type(node, None)
            return
        target_function = function
        validation_function = function
        if function.template_params:
            resolved_specialization = self._resolve_template_call_target(
                function,
                arg_types,
                node,
            )
            if resolved_specialization is None:
                self._set_type(node, None)
                return
            target_function = resolved_specialization
            validation_function = replace(
                resolved_specialization,
                name=function.name,
            )
        call_resolution = validate_call(
            self.context.diagnostics,
            function=validation_function,
            arg_types=arg_types,
            node=node,
        )
        if validation_function is not target_function:
            call_resolution = replace(
                call_resolution,
                callee=self.factory.make_callable_resolution(target_function),
                signature=target_function.signature,
                result_type=target_function.signature.return_type,
            )
        self._set_function(node, target_function)
        self._set_call(node, call_resolution)
        self._set_type(node, call_resolution.result_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.Identifier) -> None:
        """
        title: Visit Identifier nodes.
        parameters:
          node:
            type: astx.Identifier
        """
        symbol = self.context.scopes.resolve(node.name)
        if symbol is not None:
            self._set_symbol(node, symbol)
            self._set_type(node, symbol.type_)
            module = self._module_namespace_from_type(symbol.type_)
            if module is not None:
                self._set_module(node, module)
            return

        binding = self.bindings.resolve(node.name)
        if binding is not None and binding.module is not None:
            self._set_module(node, binding.module)
            self._set_type(
                node,
                self._module_namespace_type(binding.module),
            )
            return

        self.context.diagnostics.add(
            f"cannot resolve name '{node.name}'",
            node=node,
            code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
        )
        self._set_type(
            node, cast(astx.DataType | None, getattr(node, "type_", None))
        )

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionCall) -> None:
        """
        title: Visit FunctionCall nodes.
        parameters:
          node:
            type: astx.FunctionCall
        """
        arg_types: list[astx.DataType | None] = []
        for arg in node.args:
            self.visit(arg)
            arg_types.append(self._expr_type(arg))
        binding = self.bindings.resolve(node.fn)
        if binding is None:
            self.context.diagnostics.add(
                f"cannot resolve function '{node.fn}'",
                node=node,
                code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
            )
            return
        if binding.kind != "function" or binding.function is None:
            self.context.diagnostics.add(
                f"name '{node.fn}' does not resolve to a function",
                node=node,
                code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
            )
            return
        function = binding.function
        target_function = function
        validation_function = function
        if function.template_params:
            resolved_specialization = self._resolve_template_call_target(
                function,
                arg_types,
                node,
            )
            if resolved_specialization is None:
                self._set_type(node, None)
                return
            target_function = resolved_specialization
            validation_function = replace(
                resolved_specialization,
                name=function.name,
            )
        self._set_function(node, target_function)
        call_resolution = validate_call(
            self.context.diagnostics,
            function=validation_function,
            arg_types=arg_types,
            node=node,
        )
        if validation_function is not target_function:
            call_resolution = replace(
                call_resolution,
                callee=self.factory.make_callable_resolution(target_function),
                signature=target_function.signature,
                result_type=target_function.signature.return_type,
            )
        self._set_call(node, call_resolution)
        self._set_type(node, call_resolution.result_type)

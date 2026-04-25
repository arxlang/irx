# mypy: disable-error-code=no-redef
# mypy: disable-error-code=untyped-decorator

"""
title: Expression class-access visitors.
summary: >-
  Handle class construction, method calls, and field access using the shared
  class-resolution helpers.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.handlers._expressions.class_support import (
    ExpressionClassSupportVisitorMixin,
)
from irx.analysis.handlers.base import SemanticAnalyzerCore
from irx.analysis.resolved_nodes import (
    MethodDispatchKind,
    ResolvedBaseClassFieldAccess,
    ResolvedClassConstruction,
    ResolvedClassFieldAccess,
    ResolvedFieldAccess,
    ResolvedMethodCall,
    ResolvedStaticClassFieldAccess,
    SemanticClassMember,
)
from irx.analysis.types import display_type_name
from irx.analysis.validation import validate_call
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ExpressionClassAccessVisitorMixin(ExpressionClassSupportVisitorMixin):
    """
    title: Expression class-access visitors.
    """

    def _abstract_method_call_is_invalid(
        self,
        node: astx.AST,
        member: SemanticClassMember,
        *,
        allow_indirect_dispatch: bool,
    ) -> bool:
        """
        title: Diagnose one invalid abstract method call.
        parameters:
          node:
            type: astx.AST
          member:
            type: SemanticClassMember
          allow_indirect_dispatch:
            type: bool
        returns:
          type: bool
        """
        if not member.is_abstract:
            return False
        if (
            allow_indirect_dispatch
            and not member.is_static
            and member.dispatch_slot is not None
        ):
            return False
        self.context.diagnostics.add(
            (
                f"abstract method '{member.owner_name}.{member.name}' "
                "cannot be called directly"
            ),
            node=node,
            code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
        )
        return True

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ClassConstruct) -> None:
        """
        title: Visit ClassConstruct nodes.
        parameters:
          node:
            type: astx.ClassConstruct
        """
        class_ = self._resolve_named_class(node.class_name, node=node)
        if class_ is None:
            self._set_type(node, None)
            return
        resolve_definition = getattr(self, "_resolve_class_definition", None)
        if callable(resolve_definition) and not class_.is_resolved:
            class_ = resolve_definition(class_)
        if class_.is_abstract:
            abstract_method_names = ", ".join(
                f"{member.owner_name}.{member.name}"
                for member in class_.abstract_methods
            )
            suffix = (
                f"; abstract methods: {abstract_method_names}"
                if abstract_method_names
                else ""
            )
            self.context.diagnostics.add(
                f"abstract class '{class_.name}' cannot be constructed"
                f"{suffix}",
                node=node,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
            self._set_type(node, None)
            return
        if class_.layout is None or class_.initialization is None:
            raise TypeError(
                "class construction requires resolved layout metadata"
            )
        resolved_type = astx.ClassType(
            class_.name,
            resolved_name=class_.name,
            module_key=class_.module_key,
            qualified_name=class_.qualified_name,
            ancestor_qualified_names=tuple(
                ancestor.qualified_name for ancestor in class_.mro[1:]
            ),
        )
        self._set_class(node, class_)
        self._set_class_construction(
            node,
            ResolvedClassConstruction(
                class_=class_,
                initialization=class_.initialization,
            ),
        )
        self._set_type(node, resolved_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.MethodCall) -> None:
        """
        title: Visit MethodCall nodes.
        parameters:
          node:
            type: astx.MethodCall
        """
        self.visit(node.receiver)
        arg_types: list[astx.DataType | None] = []
        for arg in node.args:
            self.visit(arg)
            arg_types.append(self._expr_type(arg))
        if not self._require_value_expression(
            node.receiver,
            context="Method call receiver",
        ):
            self._set_type(node, None)
            return
        module = self._module_namespace(node.receiver)
        if module is not None:
            self._resolve_module_namespace_call(node, module, arg_types)
            return
        receiver_type = self._expr_type(node.receiver)
        class_ = self._resolve_class_from_type(
            receiver_type,
            node=node,
            unknown_message="method call requires a class value",
        )
        if class_ is None:
            if not isinstance(receiver_type, astx.ClassType):
                self.context.diagnostics.add(
                    "method call requires a class value, got "
                    f"{display_type_name(receiver_type)}",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
                )
            self._set_type(node, None)
            return
        resolved_overload = self._resolve_method_overload(
            class_,
            node.method_name,
            arg_types,
            is_static=False,
            node=node,
        )
        if resolved_overload is None:
            self._set_type(node, None)
            return
        member, candidates = resolved_overload
        if self._abstract_method_call_is_invalid(
            node,
            member,
            allow_indirect_dispatch=True,
        ):
            self._set_type(node, None)
            return
        function = member.lowered_function
        if function is None:
            raise TypeError("instance method must have a lowered function")
        visible_function = self._visible_method_function(class_, member)
        if function.template_params:
            specialized_function = self._resolve_template_method_call_target(
                function,
                visible_function,
                arg_types,
                node,
            )
            if specialized_function is None:
                self._set_type(node, None)
                return
            visible_function = self._specialize_signature(
                visible_function,
                self._specialization_bindings_map(specialized_function),
            )
            function = specialized_function
        call_resolution = validate_call(
            self.context.diagnostics,
            function=visible_function,
            arg_types=arg_types,
            node=node,
        )
        dispatch_kind = MethodDispatchKind.DIRECT
        if member.dispatch_slot is not None:
            dispatch_kind = MethodDispatchKind.INDIRECT
        self._set_call(node, call_resolution)
        self._set_method_call(
            node,
            ResolvedMethodCall(
                class_=class_,
                member=member,
                function=function,
                overload_key=(member.signature_key or member.qualified_name),
                dispatch_kind=dispatch_kind,
                call=call_resolution,
                candidates=candidates,
                receiver_type=receiver_type,
                receiver_class=class_,
                slot_index=member.dispatch_slot,
            ),
        )
        self._set_type(node, call_resolution.result_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BaseMethodCall) -> None:
        """
        title: Visit BaseMethodCall nodes.
        parameters:
          node:
            type: astx.BaseMethodCall
        """
        self.visit(node.receiver)
        arg_types: list[astx.DataType | None] = []
        for arg in node.args:
            self.visit(arg)
            arg_types.append(self._expr_type(arg))
        if not self._require_value_expression(
            node.receiver,
            context="Base method call receiver",
        ):
            self._set_type(node, None)
            return
        receiver_type = self._expr_type(node.receiver)
        resolved_classes = self._resolve_base_access_classes(
            receiver_type,
            node.base_class_name,
            node=node,
            context="base method call",
        )
        if resolved_classes is None:
            self._set_type(node, None)
            return
        receiver_class, base_class = resolved_classes
        resolved_overload = self._resolve_method_overload(
            base_class,
            node.method_name,
            arg_types,
            is_static=False,
            node=node,
        )
        if resolved_overload is None:
            self._set_type(node, None)
            return
        member, candidates = resolved_overload
        if self._abstract_method_call_is_invalid(
            node,
            member,
            allow_indirect_dispatch=False,
        ):
            self._set_type(node, None)
            return
        function = member.lowered_function
        if function is None:
            raise TypeError("base method must have a lowered function")
        visible_function = self._visible_method_function(base_class, member)
        if function.template_params:
            specialized_function = self._resolve_template_method_call_target(
                function,
                visible_function,
                arg_types,
                node,
            )
            if specialized_function is None:
                self._set_type(node, None)
                return
            visible_function = self._specialize_signature(
                visible_function,
                self._specialization_bindings_map(specialized_function),
            )
            function = specialized_function
        call_resolution = validate_call(
            self.context.diagnostics,
            function=visible_function,
            arg_types=arg_types,
            node=node,
        )
        self._set_call(node, call_resolution)
        self._set_method_call(
            node,
            ResolvedMethodCall(
                class_=base_class,
                member=member,
                function=function,
                overload_key=(member.signature_key or member.qualified_name),
                dispatch_kind=MethodDispatchKind.DIRECT,
                call=call_resolution,
                candidates=candidates,
                receiver_type=receiver_type,
                receiver_class=receiver_class,
            ),
        )
        self._set_type(node, call_resolution.result_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.StaticMethodCall) -> None:
        """
        title: Visit StaticMethodCall nodes.
        parameters:
          node:
            type: astx.StaticMethodCall
        """
        arg_types: list[astx.DataType | None] = []
        for arg in node.args:
            self.visit(arg)
            arg_types.append(self._expr_type(arg))
        class_ = self._resolve_named_class(node.class_name, node=node)
        if class_ is None:
            self._set_type(node, None)
            return
        resolved_overload = self._resolve_method_overload(
            class_,
            node.method_name,
            arg_types,
            is_static=True,
            node=node,
        )
        if resolved_overload is None:
            self._set_type(node, None)
            return
        member, candidates = resolved_overload
        if self._abstract_method_call_is_invalid(
            node,
            member,
            allow_indirect_dispatch=False,
        ):
            self._set_type(node, None)
            return
        function = member.lowered_function
        if function is None:
            raise TypeError("static method must have a lowered function")
        visible_function = self._visible_method_function(class_, member)
        if function.template_params:
            specialized_function = self._resolve_template_method_call_target(
                function,
                visible_function,
                arg_types,
                node,
            )
            if specialized_function is None:
                self._set_type(node, None)
                return
            visible_function = self._specialize_signature(
                visible_function,
                self._specialization_bindings_map(specialized_function),
            )
            function = specialized_function
        call_resolution = validate_call(
            self.context.diagnostics,
            function=visible_function,
            arg_types=arg_types,
            node=node,
        )
        self._set_call(node, call_resolution)
        self._set_method_call(
            node,
            ResolvedMethodCall(
                class_=class_,
                member=member,
                function=function,
                overload_key=(member.signature_key or member.qualified_name),
                dispatch_kind=MethodDispatchKind.DIRECT,
                call=call_resolution,
                candidates=candidates,
            ),
        )
        self._set_type(node, call_resolution.result_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BaseFieldAccess) -> None:
        """
        title: Visit BaseFieldAccess nodes.
        parameters:
          node:
            type: astx.BaseFieldAccess
        """
        self.visit(node.receiver)
        if not self._require_value_expression(
            node.receiver,
            context="Base field access",
        ):
            self._set_type(node, None)
            return
        receiver_type = self._expr_type(node.receiver)
        resolved_classes = self._resolve_base_access_classes(
            receiver_type,
            node.base_class_name,
            node=node,
            context="base field access",
        )
        if resolved_classes is None:
            self._set_type(node, None)
            return
        receiver_class, base_class = resolved_classes
        member = self._resolve_class_attribute_member(
            base_class,
            node.field_name,
            node=node,
        )
        if member is None:
            self._set_type(node, None)
            return
        if member.is_static:
            self.context.diagnostics.add(
                (
                    f"static attribute '{base_class.name}.{node.field_name}' "
                    "must be accessed through the class"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            self._set_type(node, None)
            return
        if not self._member_is_accessible(member):
            self.context.diagnostics.add(
                (
                    "class attribute "
                    f"'{self._member_display_name(member)}' "
                    "is not accessible from this context"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            self._set_type(node, None)
            return
        field = self._resolve_class_field_slot(
            receiver_class,
            member,
        )
        self._set_class(node, base_class)
        self._set_base_class_field_access(
            node,
            ResolvedBaseClassFieldAccess(
                receiver_class=receiver_class,
                base_class=base_class,
                member=member,
                field=field,
            ),
        )
        self._set_type(node, member.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.StaticFieldAccess) -> None:
        """
        title: Visit StaticFieldAccess nodes.
        parameters:
          node:
            type: astx.StaticFieldAccess
        """
        class_ = self._resolve_named_class(node.class_name, node=node)
        if class_ is None:
            self._set_type(node, None)
            return
        member = self._resolve_class_attribute_member(
            class_,
            node.field_name,
            node=node,
        )
        if member is None:
            self._set_type(node, None)
            return
        if not member.is_static:
            self.context.diagnostics.add(
                (
                    f"instance attribute '{class_.name}.{node.field_name}' "
                    "requires a receiver"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            self._set_type(node, None)
            return
        if not self._member_is_accessible(member):
            self.context.diagnostics.add(
                (
                    "class attribute "
                    f"'{self._member_display_name(member)}' "
                    "is not accessible from this context"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            self._set_type(node, None)
            return
        storage = self._resolve_static_class_storage(
            class_,
            member,
            node.field_name,
        )
        self._set_class(node, class_)
        self._set_static_class_field_access(
            node,
            ResolvedStaticClassFieldAccess(class_, member, storage),
        )
        self._set_type(node, member.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FieldAccess) -> None:
        """
        title: Visit FieldAccess nodes.
        parameters:
          node:
            type: astx.FieldAccess
        """
        self.visit(node.value)
        if not self._require_value_expression(
            node.value,
            context="Field access",
        ):
            self._set_type(node, None)
            return
        module = self._module_namespace(node.value)
        if module is not None:
            namespace_name = self._module_namespace_name(node.value, module)
            binding = self._resolve_module_member_binding(
                module,
                node.field_name,
                node=node,
                namespace_name=namespace_name,
            )
            if binding is None:
                self._set_type(node, None)
                return
            self._set_module_member_resolution(
                node,
                module=module,
                member_name=node.field_name,
                binding=binding,
            )
            return
        base_type = self._expr_type(node.value)
        struct = self._resolve_struct_from_type(
            base_type,
            node=node,
            unknown_message="field access requires a struct or class value",
        )
        if struct is not None:
            field_index = struct.field_indices.get(node.field_name)
            if field_index is None or field_index >= len(struct.fields):
                self.context.diagnostics.add(
                    f"struct '{struct.name}' has no field '{node.field_name}'",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
                )
                self._set_type(node, None)
                return
            field = struct.fields[field_index]
            self._set_struct(node, struct)
            self._set_field_access(node, ResolvedFieldAccess(struct, field))
            self._set_type(node, field.type_)
            return

        class_ = self._resolve_class_from_type(
            base_type,
            node=node,
            unknown_message="field access requires a struct or class value",
        )
        if class_ is None:
            if not isinstance(base_type, astx.StructType | astx.ClassType):
                self.context.diagnostics.add(
                    "field access requires a struct or class value, got "
                    f"{display_type_name(base_type)}",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
                )
            self._set_type(node, None)
            return

        member = self._resolve_class_attribute_member(
            class_,
            node.field_name,
            node=node,
        )
        if member is None:
            self._set_type(node, None)
            return
        if member.is_static:
            self.context.diagnostics.add(
                (
                    f"static attribute '{class_.name}.{node.field_name}' "
                    "must be accessed through the class"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            self._set_type(node, None)
            return
        if not self._member_is_accessible(member):
            self.context.diagnostics.add(
                (
                    "class attribute "
                    f"'{self._member_display_name(member)}' "
                    "is not accessible from this context"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            self._set_type(node, None)
            return
        field = self._resolve_class_field_slot(
            class_,
            member,
            visible_name=node.field_name,
        )
        self._set_class(node, class_)
        self._set_class_field_access(
            node,
            ResolvedClassFieldAccess(class_, member, field),
        )
        self._set_type(node, member.type_)

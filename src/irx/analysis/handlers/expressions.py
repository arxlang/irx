# mypy: disable-error-code=no-redef

"""
title: Expression-oriented semantic visitors.
summary: >-
  Resolve lexical identifiers, visible function names, and expression typing
  rules while delegating reusable registration and binding logic elsewhere.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.normalization import normalize_flags, normalize_operator
from irx.analysis.resolved_nodes import (
    ClassMemberKind,
    MethodDispatchKind,
    ResolvedBaseClassFieldAccess,
    ResolvedClassConstruction,
    ResolvedClassFieldAccess,
    ResolvedFieldAccess,
    ResolvedMethodCall,
    ResolvedModuleMemberAccess,
    ResolvedStaticClassFieldAccess,
    SemanticBinding,
    SemanticClass,
    SemanticClassLayoutField,
    SemanticClassMember,
    SemanticClassStaticStorage,
    SemanticFunction,
    SemanticInfo,
    SemanticModule,
    SemanticSymbol,
)
from irx.analysis.types import (
    bit_width,
    display_type_name,
    is_boolean_type,
    is_float_type,
    is_integer_type,
    is_numeric_type,
    is_string_type,
    same_type,
)
from irx.analysis.typing import binary_result_type, unary_result_type
from irx.analysis.validation import (
    validate_assignment,
    validate_call,
    validate_cast,
    validate_literal_datetime,
    validate_literal_time,
    validate_literal_timestamp,
)
from irx.astx.binary_op import (
    SPECIALIZED_BINARY_OP_EXTRA,
    specialize_binary_op,
)
from irx.buffer import (
    BUFFER_FLAG_VALIDITY_BITMAP,
    BUFFER_VIEW_ELEMENT_TYPE_EXTRA,
    BUFFER_VIEW_METADATA_EXTRA,
    BufferMutability,
    BufferOwnership,
    BufferViewMetadata,
    buffer_view_flags,
    buffer_view_has_validity_bitmap,
    buffer_view_is_readonly,
    buffer_view_ownership,
    validate_buffer_view_metadata,
)
from irx.builtins.collections.array import (
    NDARRAY_ELEMENT_TYPE_EXTRA,
    NDARRAY_FLAGS_EXTRA,
    NDARRAY_LAYOUT_EXTRA,
    NDArrayLayout,
    ndarray_byte_bounds,
    ndarray_default_strides,
    ndarray_element_count,
    ndarray_element_size_bytes,
    ndarray_is_c_contiguous,
    ndarray_is_f_contiguous,
    validate_ndarray_layout,
)
from irx.builtins.collections.list import (
    list_element_type,
    list_has_concrete_element_type,
)
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked

RAW_BUFFER_BYTE_BITS = 8
CLASS_QUALIFIED_NAME_SEPARATOR = "::class::"


@typechecked
class ExpressionVisitorMixin(SemanticVisitorMixinBase):
    def _static_buffer_view_metadata(
        self,
        node: astx.AST,
    ) -> BufferViewMetadata | None:
        """
        title: Return static buffer metadata when analysis can prove it.
        parameters:
          node:
            type: astx.AST
        returns:
          type: BufferViewMetadata | None
        """
        semantic = self._semantic(node)
        metadata = semantic.extras.get(BUFFER_VIEW_METADATA_EXTRA)
        if isinstance(metadata, BufferViewMetadata):
            return metadata

        symbol = semantic.resolved_symbol
        declaration = symbol.declaration if symbol is not None else None
        initializer = getattr(declaration, "value", None)
        if not isinstance(initializer, astx.AST):
            return None

        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        metadata = initializer_extras.get(BUFFER_VIEW_METADATA_EXTRA)
        if isinstance(metadata, BufferViewMetadata):
            return metadata
        return None

    def _static_buffer_view_element_type(
        self,
        node: astx.AST,
    ) -> astx.DataType | None:
        """
        title: Return the scalar element type when analysis can prove it.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.DataType | None
        """
        semantic = self._semantic(node)
        element_type = semantic.extras.get(BUFFER_VIEW_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type

        view_type = self._expr_type(node)
        if (
            isinstance(view_type, astx.BufferViewType)
            and view_type.element_type is not None
        ):
            return view_type.element_type

        symbol = semantic.resolved_symbol
        declaration = symbol.declaration if symbol is not None else None
        initializer = getattr(declaration, "value", None)
        if not isinstance(initializer, astx.AST):
            return None

        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        element_type = initializer_extras.get(BUFFER_VIEW_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type

        initializer_type = getattr(
            initializer_semantic,
            "resolved_type",
            getattr(initializer, "type_", None),
        )
        if (
            isinstance(initializer_type, astx.BufferViewType)
            and initializer_type.element_type is not None
        ):
            return initializer_type.element_type
        return None

    def _static_ndarray_layout(
        self,
        node: astx.AST,
    ) -> NDArrayLayout | None:
        """
        title: >-
          Return static ndarray layout metadata when analysis can prove it.
        parameters:
          node:
            type: astx.AST
        returns:
          type: NDArrayLayout | None
        """
        semantic = self._semantic(node)
        layout = semantic.extras.get(NDARRAY_LAYOUT_EXTRA)
        if isinstance(layout, NDArrayLayout):
            return layout

        symbol = semantic.resolved_symbol
        declaration = symbol.declaration if symbol is not None else None
        initializer = getattr(declaration, "value", None)
        if not isinstance(initializer, astx.AST):
            return None

        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        layout = initializer_extras.get(NDARRAY_LAYOUT_EXTRA)
        if isinstance(layout, NDArrayLayout):
            return layout
        return None

    def _static_ndarray_element_type(
        self,
        node: astx.AST,
    ) -> astx.DataType | None:
        """
        title: >-
          Return the scalar ndarray element type when analysis can prove it.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.DataType | None
        """
        semantic = self._semantic(node)
        element_type = semantic.extras.get(NDARRAY_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type

        ndarray_type = self._expr_type(node)
        if (
            isinstance(ndarray_type, astx.NDArrayType)
            and ndarray_type.element_type is not None
        ):
            return ndarray_type.element_type

        symbol = semantic.resolved_symbol
        declaration = symbol.declaration if symbol is not None else None
        initializer = getattr(declaration, "value", None)
        if not isinstance(initializer, astx.AST):
            return None

        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        element_type = initializer_extras.get(NDARRAY_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type

        initializer_type = getattr(
            initializer_semantic,
            "resolved_type",
            getattr(initializer, "type_", None),
        )
        if (
            isinstance(initializer_type, astx.NDArrayType)
            and initializer_type.element_type is not None
        ):
            return initializer_type.element_type
        return None

    def _static_ndarray_flags(
        self,
        node: astx.AST,
    ) -> int | None:
        """
        title: Return static NDArray flags when analysis can prove them.
        parameters:
          node:
            type: astx.AST
        returns:
          type: int | None
        """
        semantic = self._semantic(node)
        flags = semantic.extras.get(NDARRAY_FLAGS_EXTRA)
        if isinstance(flags, int):
            return flags

        symbol = semantic.resolved_symbol
        declaration = symbol.declaration if symbol is not None else None
        initializer = getattr(declaration, "value", None)
        if not isinstance(initializer, astx.AST):
            return None

        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        flags = initializer_extras.get(NDARRAY_FLAGS_EXTRA)
        if isinstance(flags, int):
            return flags
        return None

    def _is_subclass_of(
        self,
        class_: SemanticClass | None,
        ancestor: SemanticClass,
    ) -> bool:
        """
        title: Return whether one class is derived from another.
        parameters:
          class_:
            type: SemanticClass | None
          ancestor:
            type: SemanticClass
        returns:
          type: bool
        """
        if class_ is None:
            return False
        return any(
            item.qualified_name == ancestor.qualified_name
            for item in class_.mro
        )

    def _member_owner_module_key(
        self,
        member: SemanticClassMember,
    ) -> str:
        """
        title: Return the module key for one declaring class member.
        parameters:
          member:
            type: SemanticClassMember
        returns:
          type: str
        """
        return member.owner_qualified_name.partition(
            CLASS_QUALIFIED_NAME_SEPARATOR
        )[0]

    def _member_declaring_class(
        self,
        member: SemanticClassMember,
    ) -> SemanticClass | None:
        """
        title: Return the declaring class for one resolved class member.
        parameters:
          member:
            type: SemanticClassMember
        returns:
          type: SemanticClass | None
        """
        return self.context.get_class(
            self._member_owner_module_key(member),
            member.owner_name,
        )

    def _member_display_name(
        self,
        member: SemanticClassMember,
    ) -> str:
        """
        title: Return one class member name with its declaring owner.
        parameters:
          member:
            type: SemanticClassMember
        returns:
          type: str
        """
        return f"{member.owner_name}.{member.name}"

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

    def _class_member_target_symbol(
        self,
        target: astx.AST,
        member: SemanticClassMember,
        target_type: astx.DataType | None,
    ) -> SemanticSymbol:
        """
        title: >-
          Build one variable-like symbol for a resolved class field target.
        parameters:
          target:
            type: astx.AST
          member:
            type: SemanticClassMember
          target_type:
            type: astx.DataType | None
        returns:
          type: SemanticSymbol
        """
        declaring_class = self._member_declaring_class(member)
        module_key = (
            declaring_class.module_key
            if declaring_class is not None
            else self._member_owner_module_key(member)
        )
        symbol_type = member.type_ or cast(astx.DataType, target_type)
        symbol_kind = (
            "class_static_field" if member.is_static else "class_field"
        )
        return self.factory.make_variable_symbol(
            module_key,
            self._member_display_name(member),
            symbol_type,
            is_mutable=member.is_mutable,
            declaration=target,
            kind=symbol_kind,
        )

    def _resolve_mutation_target(
        self,
        target: astx.AST,
        *,
        node: astx.AST,
        action: str,
        invalid_message: str,
    ) -> tuple[SemanticSymbol, str, astx.DataType | None] | None:
        """
        title: >-
          Resolve one direct mutable target for assignment-like operations.
        parameters:
          target:
            type: astx.AST
          node:
            type: astx.AST
          action:
            type: str
          invalid_message:
            type: str
        returns:
          type: tuple[SemanticSymbol, str, astx.DataType | None] | None
        """
        target_type = self._expr_type(target)
        if isinstance(target, astx.Identifier):
            symbol = cast(
                SemanticInfo, getattr(target, "semantic", SemanticInfo())
            ).resolved_symbol
            if symbol is None:
                self.context.diagnostics.add(
                    invalid_message,
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
                return None
            target_name = symbol.name
            if not symbol.is_mutable:
                self.context.diagnostics.add(
                    f"Cannot {action} '{target_name}': declared as constant",
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
            return symbol, target_name, target_type

        if isinstance(target, astx.FieldAccess):
            semantic = getattr(target, "semantic", None)
            class_field_access = getattr(
                semantic,
                "resolved_class_field_access",
                None,
            )
            if class_field_access is not None:
                member = class_field_access.member
                target_name = self._member_display_name(member)
                symbol = self._class_member_target_symbol(
                    target,
                    member,
                    target_type,
                )
                if not member.is_mutable:
                    message = (
                        f"Cannot {action} '{target_name}': "
                        "declared as constant"
                    )
                    self.context.diagnostics.add(
                        message,
                        node=node,
                        code=(
                            DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET
                        ),
                    )
                return symbol, target_name, target_type

            if isinstance(
                target.value,
                (astx.BaseFieldAccess, astx.StaticFieldAccess),
            ):
                self.context.diagnostics.add(
                    invalid_message.replace(
                        "a variable or field",
                        "a direct variable or field",
                    ),
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
                )
                return None

            symbol = self._root_assignment_symbol(target)
            target_name = target.field_name
            if symbol is None:
                self.context.diagnostics.add(
                    invalid_message,
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
                return None
            if not symbol.is_mutable:
                self.context.diagnostics.add(
                    f"Cannot {action} '{symbol.name}': declared as constant",
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
            return symbol, target_name, target_type

        if isinstance(target, astx.BaseFieldAccess):
            resolved_access = getattr(
                getattr(target, "semantic", None),
                "resolved_base_class_field_access",
                None,
            )
            if resolved_access is None:
                self.context.diagnostics.add(
                    invalid_message,
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
                return None
            member = resolved_access.member
            target_name = self._member_display_name(member)
            symbol = self._class_member_target_symbol(
                target,
                member,
                target_type,
            )
            if not member.is_mutable:
                self.context.diagnostics.add(
                    f"Cannot {action} '{target_name}': declared as constant",
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
            return symbol, target_name, target_type

        if isinstance(target, astx.StaticFieldAccess):
            resolved_access = getattr(
                getattr(target, "semantic", None),
                "resolved_static_class_field_access",
                None,
            )
            if resolved_access is None:
                self.context.diagnostics.add(
                    invalid_message,
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
                return None
            member = resolved_access.member
            target_name = self._member_display_name(member)
            symbol = self._class_member_target_symbol(
                target,
                member,
                target_type,
            )
            if not member.is_mutable:
                self.context.diagnostics.add(
                    f"Cannot {action} '{target_name}': declared as constant",
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
            return symbol, target_name, target_type

        self.context.diagnostics.add(
            invalid_message,
            node=node,
            code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
        )
        return None

    def _declared_members_named(
        self,
        class_: SemanticClass,
        name: str,
    ) -> tuple[SemanticClassMember, ...]:
        """
        title: Return declared class members with one source name.
        parameters:
          class_:
            type: SemanticClass
          name:
            type: str
        returns:
          type: tuple[SemanticClassMember, Ellipsis]
        """
        matches: list[SemanticClassMember] = []
        for owner in class_.mro:
            matches.extend(
                member
                for member in owner.declared_members
                if member.name == name
            )
        return tuple(matches)

    def _hidden_method_candidates(
        self,
        class_: SemanticClass,
        method_name: str,
        *,
        is_static: bool,
    ) -> tuple[SemanticClassMember, ...]:
        """
        title: Return declared methods hidden from the effective lookup group.
        parameters:
          class_:
            type: SemanticClass
          method_name:
            type: str
          is_static:
            type: bool
        returns:
          type: tuple[SemanticClassMember, Ellipsis]
        """
        visible_members = class_.method_groups.get(method_name, ())
        visible_names = {member.qualified_name for member in visible_members}
        return tuple(
            member
            for member in self._declared_members_named(class_, method_name)
            if member.kind is ClassMemberKind.METHOD
            and member.is_static == is_static
            and member.qualified_name not in visible_names
        )

    def _hidden_attribute_candidates(
        self,
        class_: SemanticClass,
        field_name: str,
    ) -> tuple[SemanticClassMember, ...]:
        """
        title: Return declared attributes hidden from effective lookup.
        parameters:
          class_:
            type: SemanticClass
          field_name:
            type: str
        returns:
          type: tuple[SemanticClassMember, Ellipsis]
        """
        visible_member = class_.member_table.get(field_name)
        visible_name = (
            visible_member.qualified_name
            if visible_member is not None
            else None
        )
        return tuple(
            member
            for member in self._declared_members_named(class_, field_name)
            if member.kind is ClassMemberKind.ATTRIBUTE
            and member.qualified_name != visible_name
        )

    def _resolve_class_attribute_member(
        self,
        class_: SemanticClass,
        field_name: str,
        *,
        node: astx.AST,
    ) -> SemanticClassMember | None:
        """
        title: Resolve one visible class attribute candidate by source name.
        parameters:
          class_:
            type: SemanticClass
          field_name:
            type: str
          node:
            type: astx.AST
        returns:
          type: SemanticClassMember | None
        """
        member = class_.member_table.get(field_name)
        if member is not None:
            if member.kind is not ClassMemberKind.ATTRIBUTE:
                self.context.diagnostics.add(
                    (
                        f"class member '{class_.name}.{field_name}' "
                        "is not an attribute"
                    ),
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
                )
                return None
            return member

        if field_name in class_.method_groups:
            self.context.diagnostics.add(
                (
                    f"class member '{class_.name}.{field_name}' "
                    "is not an attribute"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            return None

        hidden_members = self._hidden_attribute_candidates(class_, field_name)
        accessible_hidden = tuple(
            candidate
            for candidate in hidden_members
            if self._member_is_accessible(candidate)
        )
        if accessible_hidden:
            return accessible_hidden[0]
        if hidden_members:
            self.context.diagnostics.add(
                (
                    "class attribute "
                    f"'{self._member_display_name(hidden_members[0])}' "
                    "is not accessible from this context"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            return None

        self.context.diagnostics.add(
            (f"class '{class_.name}' has no attribute '{field_name}'"),
            node=node,
            code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
        )
        return None

    def _resolve_class_field_slot(
        self,
        class_: SemanticClass,
        member: SemanticClassMember,
        *,
        visible_name: str | None = None,
    ) -> SemanticClassLayoutField:
        """
        title: Return one resolved class-instance storage slot.
        parameters:
          class_:
            type: SemanticClass
          member:
            type: SemanticClassMember
          visible_name:
            type: str | None
        returns:
          type: SemanticClassLayoutField
        """
        if class_.layout is None:
            raise TypeError("class field access requires a resolved layout")
        field = None
        if visible_name is not None:
            field = class_.layout.visible_field_slots.get(visible_name)
        if field is None:
            field = class_.layout.field_slots.get(member.qualified_name)
        if field is None:
            raise TypeError(
                "class field access requires a resolved storage slot"
            )
        return field

    def _resolve_static_class_storage(
        self,
        class_: SemanticClass,
        member: SemanticClassMember,
        field_name: str,
    ) -> SemanticClassStaticStorage:
        """
        title: Return one resolved class-static storage binding.
        parameters:
          class_:
            type: SemanticClass
          member:
            type: SemanticClassMember
          field_name:
            type: str
        returns:
          type: SemanticClassStaticStorage
        """
        if class_.layout is None:
            raise TypeError(
                "static class field access requires a resolved layout"
            )
        storage = class_.layout.visible_static_storage.get(field_name)
        if storage is None:
            storage = class_.layout.static_storage.get(member.qualified_name)
        if storage is None:
            raise TypeError(
                "static class field access requires resolved storage"
            )
        return storage

    def _member_is_accessible(
        self,
        member: SemanticClassMember,
    ) -> bool:
        """
        title: Return whether the current analysis context may access a member.
        parameters:
          member:
            type: SemanticClassMember
        returns:
          type: bool
        """
        if member.visibility is astx.VisibilityKind.public:
            return True
        current_class = self.context.current_class
        declaring_class = self._member_declaring_class(member)
        if current_class is None or declaring_class is None:
            return False
        if member.visibility is astx.VisibilityKind.private:
            return (
                current_class.qualified_name == declaring_class.qualified_name
            )
        return self._is_subclass_of(current_class, declaring_class)

    def _resolve_named_class(
        self,
        name: str,
        *,
        node: astx.AST,
    ) -> SemanticClass | None:
        """
        title: Resolve one class name from visible bindings.
        parameters:
          name:
            type: str
          node:
            type: astx.AST
        returns:
          type: SemanticClass | None
        """
        binding = self.bindings.resolve(name)
        if (
            binding is None
            or binding.kind != "class"
            or binding.class_ is None
        ):
            self.context.diagnostics.add(
                f"cannot resolve class '{name}'",
                node=node,
                code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
            )
            return None
        return binding.class_

    def _resolve_base_access_classes(
        self,
        receiver_type: astx.DataType | None,
        base_class_name: str,
        *,
        node: astx.AST,
        context: str,
    ) -> tuple[SemanticClass, SemanticClass] | None:
        """
        title: Resolve the receiver class and one explicit base view.
        parameters:
          receiver_type:
            type: astx.DataType | None
          base_class_name:
            type: str
          node:
            type: astx.AST
          context:
            type: str
        returns:
          type: tuple[SemanticClass, SemanticClass] | None
        """
        receiver_class = self._resolve_class_from_type(
            receiver_type,
            node=node,
            unknown_message=f"{context} requires a class value",
        )
        if receiver_class is None:
            if not isinstance(receiver_type, astx.ClassType):
                self.context.diagnostics.add(
                    f"{context} requires a class value, got "
                    f"{display_type_name(receiver_type)}",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
                )
            return None
        base_class = self._resolve_named_class(base_class_name, node=node)
        if base_class is None:
            return None
        if not self._is_subclass_of(receiver_class, base_class):
            self.context.diagnostics.add(
                (
                    f"class '{receiver_class.name}' does not inherit from "
                    f"'{base_class.name}'"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            return None
        return receiver_class, base_class

    def _visible_method_function(
        self,
        class_: SemanticClass,
        member: SemanticClassMember,
    ) -> SemanticFunction:
        """
        title: >-
          Return one visible-signature callable wrapper for a class method.
        parameters:
          class_:
            type: SemanticClass
          member:
            type: SemanticClassMember
        returns:
          type: SemanticFunction
        """
        declaration = member.declaration
        if member.signature is None or not isinstance(
            declaration, astx.FunctionDef
        ):
            raise TypeError("class method must carry a callable signature")
        return SemanticFunction(
            symbol_id=member.symbol_id,
            name=f"{class_.name}.{member.name}",
            return_type=member.signature.return_type,
            args=(),
            signature=member.signature,
            prototype=declaration.prototype,
            definition=declaration,
            module_key=class_.module_key,
            qualified_name=member.qualified_name,
        )

    def _format_method_signature(self, member: SemanticClassMember) -> str:
        """
        title: Return one human-facing method signature string.
        parameters:
          member:
            type: SemanticClassMember
        returns:
          type: str
        """
        if member.signature is None:
            return member.name
        parameters = ", ".join(
            display_type_name(parameter.type_)
            for parameter in member.signature.parameters
        )
        suffix = " static" if member.is_static else ""
        return (
            f"{member.name}({parameters}) -> "
            f"{display_type_name(member.signature.return_type)}{suffix}"
        )

    def _format_requested_method_signature(
        self,
        method_name: str,
        arg_types: list[astx.DataType | None],
        *,
        is_static: bool,
    ) -> str:
        """
        title: Return one human-facing requested method signature.
        parameters:
          method_name:
            type: str
          arg_types:
            type: list[astx.DataType | None]
          is_static:
            type: bool
        returns:
          type: str
        """
        parameters = ", ".join(display_type_name(arg) for arg in arg_types)
        suffix = " static" if is_static else ""
        return f"{method_name}({parameters}){suffix}"

    def _method_arguments_exactly_match(
        self,
        member: SemanticClassMember,
        arg_types: list[astx.DataType | None],
    ) -> bool:
        """
        title: Return whether one method exactly matches explicit arg types.
        parameters:
          member:
            type: SemanticClassMember
          arg_types:
            type: list[astx.DataType | None]
        returns:
          type: bool
        """
        if member.signature is None:
            return False
        if len(member.signature.parameters) != len(arg_types):
            return False
        return all(
            same_type(parameter.type_, arg_type)
            for parameter, arg_type in zip(
                member.signature.parameters,
                arg_types,
            )
        )

    def _resolve_method_overload(
        self,
        class_: SemanticClass,
        method_name: str,
        arg_types: list[astx.DataType | None],
        *,
        is_static: bool,
        node: astx.AST,
    ) -> tuple[SemanticClassMember, tuple[SemanticClassMember, ...]] | None:
        """
        title: Resolve one class-method overload from the effective group.
        parameters:
          class_:
            type: SemanticClass
          method_name:
            type: str
          arg_types:
            type: list[astx.DataType | None]
          is_static:
            type: bool
          node:
            type: astx.AST
        returns:
          type: >-
            tuple[SemanticClassMember, tuple[SemanticClassMember, Ellipsis]] |
            None
        """
        member = class_.member_table.get(method_name)
        group = class_.method_groups.get(method_name, ())
        if not group:
            if (
                member is not None
                and member.kind is not ClassMemberKind.METHOD
            ):
                self.context.diagnostics.add(
                    (
                        f"class member '{class_.name}.{method_name}' "
                        "is not a method"
                    ),
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
                )
                return None
            hidden_group = self._hidden_method_candidates(
                class_,
                method_name,
                is_static=is_static,
            )
            if hidden_group:
                group = hidden_group
            else:
                missing_label = "static method" if is_static else "method"
                self.context.diagnostics.add(
                    (
                        f"class '{class_.name}' has no {missing_label} "
                        f"'{method_name}'"
                    ),
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
                )
                return None

        kind_candidates = tuple(
            candidate
            for candidate in group
            if candidate.is_static == is_static
        )
        if not kind_candidates:
            if is_static:
                self.context.diagnostics.add(
                    (
                        f"instance method '{class_.name}.{method_name}' "
                        "requires a receiver"
                    ),
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
                )
            else:
                self.context.diagnostics.add(
                    (
                        f"static method '{class_.name}.{method_name}' "
                        "must be called through the class"
                    ),
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
                )
            return None

        accessible_candidates = tuple(
            candidate
            for candidate in kind_candidates
            if self._member_is_accessible(candidate)
        )
        if not accessible_candidates:
            self.context.diagnostics.add(
                (
                    "class method "
                    f"'{self._member_display_name(kind_candidates[0])}' "
                    "is not accessible from this context"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            return None

        if len(accessible_candidates) == 1:
            return accessible_candidates[0], accessible_candidates

        exact_matches = tuple(
            candidate
            for candidate in accessible_candidates
            if self._method_arguments_exactly_match(candidate, arg_types)
        )
        if len(exact_matches) == 1:
            return exact_matches[0], accessible_candidates
        if len(exact_matches) > 1:
            available = ", ".join(
                self._format_method_signature(candidate)
                for candidate in accessible_candidates
            )
            self.context.diagnostics.add(
                (
                    f"call to '{class_.name}.{method_name}' is ambiguous; "
                    f"available overloads: {available}"
                ),
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
            )
            return None

        requested = self._format_requested_method_signature(
            method_name,
            arg_types,
            is_static=is_static,
        )
        available = ", ".join(
            self._format_method_signature(candidate)
            for candidate in accessible_candidates
        )
        self.context.diagnostics.add(
            (
                f"class '{class_.name}' has no exact overload for "
                f"'{requested}'; available overloads: {available}"
            ),
            node=node,
            code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
        )
        return None

    def _static_integer_literal_value(self, node: astx.AST) -> int | None:
        """
        title: Return a static integer literal value when present.
        parameters:
          node:
            type: astx.AST
        returns:
          type: int | None
        """
        if isinstance(
            node,
            (
                astx.LiteralInt8,
                astx.LiteralInt16,
                astx.LiteralInt32,
                astx.LiteralInt64,
                astx.LiteralUInt8,
                astx.LiteralUInt16,
                astx.LiteralUInt32,
                astx.LiteralUInt64,
                astx.LiteralUInt128,
            ),
        ):
            return int(node.value)
        return None

    def _validate_buffer_view_index_operation(
        self,
        *,
        node: astx.AST,
        base: astx.AST,
        indices: list[astx.AST],
        is_store: bool,
    ) -> astx.DataType | None:
        """
        title: Validate one low-level buffer view indexed access.
        parameters:
          node:
            type: astx.AST
          base:
            type: astx.AST
          indices:
            type: list[astx.AST]
          is_store:
            type: bool
        returns:
          type: astx.DataType | None
        """
        base_type = self._expr_type(base)
        if not isinstance(base_type, astx.BufferViewType):
            self.context.diagnostics.add(
                "buffer view indexing requires a BufferViewType base",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        metadata = self._static_buffer_view_metadata(base)
        if metadata is None:
            self.context.diagnostics.add(
                "buffer view indexing requires static descriptor metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif len(indices) != metadata.ndim:
            self.context.diagnostics.add(
                "buffer view indexing index count must match descriptor ndim",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif len(metadata.shape) == metadata.ndim:
            for axis, index in enumerate(indices):
                extent = metadata.shape[axis]
                static_index = self._static_integer_literal_value(index)
                if static_index is None:
                    continue
                if static_index < 0 or static_index >= extent:
                    self.context.diagnostics.add(
                        "buffer view index "
                        f"{axis} statically out of bounds for extent {extent}",
                        node=index,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )

        if (
            is_store
            and metadata is not None
            and buffer_view_is_readonly(metadata.flags)
        ):
            self.context.diagnostics.add(
                "cannot write through a readonly buffer view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        for index in indices:
            index_type = self._expr_type(index)
            if not is_integer_type(index_type):
                self.context.diagnostics.add(
                    "buffer view indices must be integer typed",
                    node=index,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )
                continue
            if bit_width(index_type) > bit_width(astx.Int64()):
                self.context.diagnostics.add(
                    "buffer view indices must fit 64-bit "
                    "descriptor stride arithmetic",
                    node=index,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )

        element_type = self._static_buffer_view_element_type(base)
        if element_type is None:
            self.context.diagnostics.add(
                "buffer view indexing requires a known element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
            return None
        if not (
            is_integer_type(element_type)
            or is_float_type(element_type)
            or is_boolean_type(element_type)
        ):
            self.context.diagnostics.add(
                "buffer view indexing requires a scalar element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
            return None
        self._semantic(node).extras[BUFFER_VIEW_ELEMENT_TYPE_EXTRA] = (
            element_type
        )
        return element_type

    def _validate_ndarray_index_operation(
        self,
        *,
        node: astx.AST,
        base: astx.AST,
        indices: list[astx.AST],
        is_store: bool,
    ) -> astx.DataType | None:
        """
        title: Validate one NDArray indexed access.
        parameters:
          node:
            type: astx.AST
          base:
            type: astx.AST
          indices:
            type: list[astx.AST]
          is_store:
            type: bool
        returns:
          type: astx.DataType | None
        """
        base_type = self._expr_type(base)
        if not isinstance(base_type, astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray indexing requires a NDArrayType base",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        layout = self._static_ndarray_layout(base)
        if layout is None:
            self.context.diagnostics.add(
                "ndarray indexing requires static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif len(indices) != layout.ndim:
            self.context.diagnostics.add(
                "ndarray indexing index count must match ndarray ndim",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        else:
            for axis, index in enumerate(indices):
                extent = layout.shape[axis]
                static_index = self._static_integer_literal_value(index)
                if static_index is None:
                    continue
                if static_index < 0 or static_index >= extent:
                    self.context.diagnostics.add(
                        "ndarray index "
                        f"{axis} statically out of bounds for extent {extent}",
                        node=index,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )

        flags = self._static_ndarray_flags(base)
        if is_store and flags is not None and buffer_view_is_readonly(flags):
            self.context.diagnostics.add(
                "cannot write through a readonly ndarray view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        for index in indices:
            index_type = self._expr_type(index)
            if not is_integer_type(index_type):
                self.context.diagnostics.add(
                    "ndarray indices must be integer typed",
                    node=index,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )
                continue
            if bit_width(index_type) > bit_width(astx.Int64()):
                self.context.diagnostics.add(
                    "ndarray indices must fit 64-bit stride arithmetic",
                    node=index,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )

        element_type = self._static_ndarray_element_type(base)
        if element_type is None:
            self.context.diagnostics.add(
                "ndarray indexing requires a known element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
            return None
        if ndarray_element_size_bytes(element_type) is None:
            self.context.diagnostics.add(
                "ndarray indexing requires a fixed-width numeric element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
            return None

        self._semantic(node).extras[NDARRAY_ELEMENT_TYPE_EXTRA] = element_type
        return element_type

    def _validate_buffer_lifetime_operation(
        self,
        *,
        node: astx.AST,
        view: astx.AST,
        operation: str,
    ) -> None:
        """
        title: Validate one explicit buffer lifetime helper operation.
        parameters:
          node:
            type: astx.AST
          view:
            type: astx.AST
          operation:
            type: str
        """
        metadata = self._static_buffer_view_metadata(view)
        if metadata is None:
            return
        ownership = buffer_view_ownership(metadata.flags)
        if ownership is BufferOwnership.BORROWED or metadata.owner.is_null:
            self.context.diagnostics.add(
                f"buffer {operation} requires an owned or external-owner view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

    def _validate_ndarray_lifetime_operation(
        self,
        *,
        node: astx.AST,
        view: astx.AST,
        operation: str,
    ) -> None:
        """
        title: Validate one explicit NDArray lifetime helper operation.
        parameters:
          node:
            type: astx.AST
          view:
            type: astx.AST
          operation:
            type: str
        """
        flags = self._static_ndarray_flags(view)
        if flags is None:
            return
        ownership = buffer_view_ownership(flags)
        if ownership is BufferOwnership.BORROWED:
            self.context.diagnostics.add(
                "ndarray "
                f"{operation} requires an owned or external-owner view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

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
    def visit(self, node: astx.VariableAssignment) -> None:
        """
        title: Visit VariableAssignment nodes.
        parameters:
          node:
            type: astx.VariableAssignment
        """
        self.visit(node.value)
        symbol = self.context.scopes.resolve(node.name)
        if symbol is None:
            self.context.diagnostics.add(
                f"cannot assign to unresolved name '{node.name}'",
                node=node,
                code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
            )
            return
        if not symbol.is_mutable:
            self.context.diagnostics.add(
                f"Cannot assign to '{node.name}': declared as constant",
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
            )
        if self._require_value_expression(
            node.value,
            context=f"Assignment to '{node.name}'",
        ):
            validate_assignment(
                self.context.diagnostics,
                target_name=node.name,
                target_type=symbol.type_,
                value_type=self._expr_type(node.value),
                node=node,
            )
        self._set_symbol(node, symbol)
        self._set_assignment(node, symbol)
        self._set_type(node, symbol.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.UnaryOp) -> None:
        """
        title: Visit UnaryOp nodes.
        parameters:
          node:
            type: astx.UnaryOp
        """
        self.visit(node.operand)
        if not self._require_value_expression(
            node.operand,
            context=f"Unary operator '{node.op_code}'",
        ):
            self._set_type(node, None)
            return
        operand_type = self._expr_type(node.operand)
        if (
            node.op_code == "!"
            and operand_type is not None
            and not is_boolean_type(operand_type)
        ):
            self.context.diagnostics.add(
                "unary operator '!' requires Boolean operand",
                node=node,
            )
        result_type = unary_result_type(node.op_code, operand_type)
        if node.op_code in {"++", "--"}:
            resolved_target = self._resolve_mutation_target(
                node.operand,
                node=node,
                action="mutate",
                invalid_message=(
                    "mutation target must be a variable or field"
                ),
            )
            if resolved_target is not None:
                target_symbol, _target_name, _target_type = resolved_target
                self._set_assignment(node, target_symbol)
        flags = normalize_flags(node, lhs_type=operand_type)
        self._set_flags(node, flags)
        self._set_operator(
            node,
            normalize_operator(
                node.op_code,
                result_type=result_type,
                lhs_type=operand_type,
                flags=flags,
            ),
        )
        self._set_type(node, result_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BinaryOp) -> None:
        """
        title: Visit BinaryOp nodes.
        parameters:
          node:
            type: astx.BinaryOp
        """
        self.visit(node.lhs)
        self.visit(node.rhs)
        lhs_type = self._expr_type(node.lhs)
        rhs_type = self._expr_type(node.rhs)
        flags = normalize_flags(node, lhs_type=lhs_type, rhs_type=rhs_type)
        self._set_flags(node, flags)
        specialized = specialize_binary_op(node)
        if specialized is not node:
            setattr(specialized, "semantic", self._semantic(node))
        self._semantic(node).extras[SPECIALIZED_BINARY_OP_EXTRA] = specialized

        if node.op_code == "=":
            resolved_target = self._resolve_mutation_target(
                node.lhs,
                node=node,
                action="assign to",
                invalid_message=(
                    "assignment target must be a variable or field"
                ),
            )
            if resolved_target is None:
                return
            assignment_symbol, target_name, target_type = resolved_target
            if self._require_value_expression(
                node.rhs,
                context=f"Assignment to '{target_name}'",
            ):
                validate_assignment(
                    self.context.diagnostics,
                    target_name=target_name,
                    target_type=target_type,
                    value_type=rhs_type,
                    node=node,
                )
            self._set_assignment(node, assignment_symbol)
            if isinstance(node.lhs, astx.Identifier):
                self._set_symbol(node.lhs, assignment_symbol)
            self._set_type(node, target_type)
            self._set_operator(
                node,
                normalize_operator(
                    node.op_code,
                    result_type=target_type,
                    lhs_type=target_type,
                    rhs_type=rhs_type,
                    flags=flags,
                ),
            )
            return

        lhs_has_value = self._require_value_expression(
            node.lhs,
            context=f"Operator '{node.op_code}'",
        )
        rhs_has_value = self._require_value_expression(
            node.rhs,
            context=f"Operator '{node.op_code}'",
        )
        if not (lhs_has_value and rhs_has_value):
            self._set_type(node, None)
            self._set_operator(
                node,
                normalize_operator(
                    node.op_code,
                    result_type=None,
                    lhs_type=lhs_type,
                    rhs_type=rhs_type,
                    flags=flags,
                ),
            )
            return

        if flags.fma and flags.fma_rhs is None:
            self.context.diagnostics.add(
                "FMA requires a third operand (fma_rhs)",
                node=node,
            )
        if flags.fma and flags.fma_rhs is not None:
            self.visit(flags.fma_rhs)

        if (
            node.op_code in {"&&", "and", "||", "or"}
            and lhs_type is not None
            and rhs_type is not None
            and not (is_boolean_type(lhs_type) and is_boolean_type(rhs_type))
        ):
            self.context.diagnostics.add(
                f"logical operator '{node.op_code}' requires Boolean operands",
                node=node,
            )

        if node.op_code in {"+", "-", "*", "/", "%"} and not (
            (is_numeric_type(lhs_type) and is_numeric_type(rhs_type))
            or (
                node.op_code == "+"
                and is_string_type(lhs_type)
                and is_string_type(rhs_type)
            )
        ):
            self.context.diagnostics.add(
                f"Invalid operator '{node.op_code}' for operand types",
                node=node,
            )

        result_type = binary_result_type(node.op_code, lhs_type, rhs_type)
        self._set_type(node, result_type)
        self._set_operator(
            node,
            normalize_operator(
                node.op_code,
                result_type=result_type,
                lhs_type=lhs_type,
                rhs_type=rhs_type,
                flags=flags,
            ),
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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.Cast) -> None:
        """
        title: Visit Cast nodes.
        parameters:
          node:
            type: astx.Cast
        """
        self.visit(node.value)
        if not self._require_value_expression(
            node.value,
            context="Cast",
        ):
            self._set_type(node, cast(astx.DataType | None, node.target_type))
            return
        source_type = self._expr_type(node.value)
        target_type = cast(astx.DataType | None, node.target_type)
        validate_cast(
            self.context.diagnostics,
            source_type=source_type,
            target_type=target_type,
            node=node,
        )
        self._set_type(node, target_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.PrintExpr) -> None:
        """
        title: Visit PrintExpr nodes.
        parameters:
          node:
            type: astx.PrintExpr
        """
        self.visit(node.message)
        if not self._require_value_expression(
            node.message,
            context="PrintExpr",
        ):
            self._set_type(node, astx.Int32())
            return
        message_type = self._expr_type(node.message)
        if not (
            is_string_type(message_type)
            or is_integer_type(message_type)
            or is_float_type(message_type)
            or is_boolean_type(message_type)
        ):
            self.context.diagnostics.add(
                "unsupported PrintExpr message type "
                f"{display_type_name(message_type)}",
                node=node,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ArrayInt32ArrayLength) -> None:
        """
        title: Visit ArrayInt32ArrayLength nodes.
        parameters:
          node:
            type: astx.ArrayInt32ArrayLength
        """
        for item in node.values:
            self.visit(item)
            if not is_integer_type(self._expr_type(item)):
                self.context.diagnostics.add(
                    "Array helper supports only integer expressions",
                    node=item,
                )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayLiteral) -> None:
        """
        title: Visit NDArrayLiteral nodes.
        parameters:
          node:
            type: astx.NDArrayLiteral
        """
        for item in node.values:
            self.visit(item)
            validate_assignment(
                self.context.diagnostics,
                target_name="ndarray element",
                target_type=node.element_type,
                value_type=self._expr_type(item),
                node=item,
            )

        shape = tuple(node.shape)
        element_size_bytes = ndarray_element_size_bytes(node.element_type)
        if element_size_bytes is None:
            if isinstance(node.element_type, astx.Boolean):
                self.context.diagnostics.add(
                    "bool ndarrays are not supported because bit-packed Arrow "
                    "values are not buffer-view compatible",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )
            else:
                self.context.diagnostics.add(
                    "ndarray literals require a fixed-width numeric element "
                    "type",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )

        if node.strides is None:
            if element_size_bytes is None or any(dim < 0 for dim in shape):
                strides = tuple(0 for _ in shape)
            else:
                strides = ndarray_default_strides(shape, element_size_bytes)
        else:
            strides = tuple(node.strides)

        layout = NDArrayLayout(
            shape=shape,
            strides=strides,
            offset_bytes=node.offset_bytes,
        )
        for error in validate_ndarray_layout(layout):
            self.context.diagnostics.add(
                error,
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        expected_value_count = ndarray_element_count(layout)
        if len(node.values) != expected_value_count:
            self.context.diagnostics.add(
                "ndarray literal value count must match the shape extent",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        if element_size_bytes is not None:
            bounds = ndarray_byte_bounds(layout)
            if bounds is not None:
                minimum, maximum = bounds
                storage_bytes = len(node.values) * element_size_bytes
                if minimum < 0 or maximum + element_size_bytes > storage_bytes:
                    self.context.diagnostics.add(
                        "ndarray literal layout exceeds compact backing "
                        "storage",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )

            flags = buffer_view_flags(
                BufferOwnership.EXTERNAL_OWNER,
                BufferMutability.READONLY,
                c_contiguous=ndarray_is_c_contiguous(
                    layout,
                    element_size_bytes,
                ),
                f_contiguous=ndarray_is_f_contiguous(
                    layout,
                    element_size_bytes,
                ),
            )
            self._semantic(node).extras[NDARRAY_FLAGS_EXTRA] = flags

        self._semantic(node).extras[NDARRAY_LAYOUT_EXTRA] = layout
        self._semantic(node).extras[NDARRAY_ELEMENT_TYPE_EXTRA] = (
            node.element_type
        )
        node.type_ = astx.NDArrayType(node.element_type)
        self._set_type(node, node.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayView) -> None:
        """
        title: Visit NDArrayView nodes.
        parameters:
          node:
            type: astx.NDArrayView
        """
        self.visit(node.base)
        base_type = self._expr_type(node.base)
        if not isinstance(base_type, astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray views require a NDArrayType base",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        base_layout = self._static_ndarray_layout(node.base)
        if base_layout is None:
            self.context.diagnostics.add(
                "ndarray views require static base layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        element_type = self._static_ndarray_element_type(node.base)
        if element_type is None:
            self.context.diagnostics.add(
                "ndarray views require a known element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        element_size_bytes = ndarray_element_size_bytes(element_type)
        if element_type is not None and element_size_bytes is None:
            self.context.diagnostics.add(
                "ndarray views require a fixed-width numeric element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        shape = tuple(node.shape)
        if node.strides is None:
            if (
                base_layout is None
                or element_size_bytes is None
                or any(dim < 0 for dim in shape)
            ):
                strides = tuple(0 for _ in shape)
            else:
                if not ndarray_is_c_contiguous(
                    base_layout,
                    element_size_bytes,
                ):
                    self.context.diagnostics.add(
                        "ndarray views without explicit strides require a "
                        "C-contiguous base",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )
                expected_count = 1
                for dim in shape:
                    expected_count *= dim
                if expected_count != ndarray_element_count(base_layout):
                    self.context.diagnostics.add(
                        "ndarray reshape views require the same element count "
                        "as the base",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )
                strides = ndarray_default_strides(shape, element_size_bytes)
        else:
            strides = tuple(node.strides)

        base_offset_bytes = (
            base_layout.offset_bytes if base_layout is not None else 0
        )
        layout = NDArrayLayout(
            shape=shape,
            strides=strides,
            offset_bytes=base_offset_bytes + node.offset_bytes,
        )
        for error in validate_ndarray_layout(layout):
            self.context.diagnostics.add(
                error,
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        if base_layout is not None and element_size_bytes is not None:
            base_bounds = ndarray_byte_bounds(base_layout)
            view_bounds = ndarray_byte_bounds(layout)
            if (
                base_bounds is not None
                and view_bounds is not None
                and (
                    view_bounds[0] < base_bounds[0]
                    or view_bounds[1] > base_bounds[1]
                )
            ):
                self.context.diagnostics.add(
                    "ndarray view exceeds base storage bounds",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )

        base_flags = self._static_ndarray_flags(node.base)
        if base_flags is None:
            flags = buffer_view_flags(
                BufferOwnership.EXTERNAL_OWNER,
                BufferMutability.READONLY,
                c_contiguous=(
                    False
                    if element_size_bytes is None
                    else ndarray_is_c_contiguous(layout, element_size_bytes)
                ),
                f_contiguous=(
                    False
                    if element_size_bytes is None
                    else ndarray_is_f_contiguous(layout, element_size_bytes)
                ),
            )
        else:
            ownership = (
                buffer_view_ownership(base_flags)
                or BufferOwnership.EXTERNAL_OWNER
            )
            mutability = (
                BufferMutability.READONLY
                if buffer_view_is_readonly(base_flags)
                else BufferMutability.WRITABLE
            )
            flags = buffer_view_flags(
                ownership,
                mutability,
                c_contiguous=(
                    False
                    if element_size_bytes is None
                    else ndarray_is_c_contiguous(layout, element_size_bytes)
                ),
                f_contiguous=(
                    False
                    if element_size_bytes is None
                    else ndarray_is_f_contiguous(layout, element_size_bytes)
                ),
            )
            if buffer_view_has_validity_bitmap(base_flags):
                flags |= BUFFER_FLAG_VALIDITY_BITMAP

        if element_type is not None:
            self._semantic(node).extras[NDARRAY_ELEMENT_TYPE_EXTRA] = (
                element_type
            )
            node.type_ = astx.NDArrayType(element_type)
        self._semantic(node).extras[NDARRAY_LAYOUT_EXTRA] = layout
        self._semantic(node).extras[NDARRAY_FLAGS_EXTRA] = flags
        self._set_type(node, node.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayIndex) -> None:
        """
        title: Visit NDArrayIndex nodes.
        parameters:
          node:
            type: astx.NDArrayIndex
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        element_type = self._validate_ndarray_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=False,
        )
        if element_type is not None:
            node.type_ = element_type
        self._set_type(node, element_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayStore) -> None:
        """
        title: Visit NDArrayStore nodes.
        parameters:
          node:
            type: astx.NDArrayStore
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        self.visit(node.value)
        element_type = self._validate_ndarray_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=True,
        )
        if element_type is not None:
            validate_assignment(
                self.context.diagnostics,
                target_name="ndarray element",
                target_type=element_type,
                value_type=self._expr_type(node.value),
                node=node,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayNDim) -> None:
        """
        title: Visit NDArrayNDim nodes.
        parameters:
          node:
            type: astx.NDArrayNDim
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray ndim requires a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayShape) -> None:
        """
        title: Visit NDArrayShape nodes.
        parameters:
          node:
            type: astx.NDArrayShape
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray shape queries require a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        layout = self._static_ndarray_layout(node.base)
        if layout is None:
            self.context.diagnostics.add(
                "ndarray shape queries require static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif node.axis < 0 or node.axis >= layout.ndim:
            self.context.diagnostics.add(
                "ndarray shape axis is out of bounds",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayStride) -> None:
        """
        title: Visit NDArrayStride nodes.
        parameters:
          node:
            type: astx.NDArrayStride
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray stride queries require a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        layout = self._static_ndarray_layout(node.base)
        if layout is None:
            self.context.diagnostics.add(
                "ndarray stride queries require static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif node.axis < 0 or node.axis >= layout.ndim:
            self.context.diagnostics.add(
                "ndarray stride axis is out of bounds",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayElementCount) -> None:
        """
        title: Visit NDArrayElementCount nodes.
        parameters:
          node:
            type: astx.NDArrayElementCount
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray element_count requires a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        if self._static_ndarray_layout(node.base) is None:
            self.context.diagnostics.add(
                "ndarray element_count requires static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayByteOffset) -> None:
        """
        title: Visit NDArrayByteOffset nodes.
        parameters:
          node:
            type: astx.NDArrayByteOffset
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        self._validate_ndarray_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=False,
        )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayRetain) -> None:
        """
        title: Visit NDArrayRetain nodes.
        parameters:
          node:
            type: astx.NDArrayRetain
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray retain requires a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._validate_ndarray_lifetime_operation(
            node=node,
            view=node.base,
            operation="retain",
        )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayRelease) -> None:
        """
        title: Visit NDArrayRelease nodes.
        parameters:
          node:
            type: astx.NDArrayRelease
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray release requires a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._validate_ndarray_lifetime_operation(
            node=node,
            view=node.base,
            operation="release",
        )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewDescriptor) -> None:
        """
        title: Visit BufferViewDescriptor nodes.
        parameters:
          node:
            type: astx.BufferViewDescriptor
        """
        for error in validate_buffer_view_metadata(node.metadata):
            self.context.diagnostics.add(error, node=node)
        self._semantic(node).extras[BUFFER_VIEW_METADATA_EXTRA] = node.metadata
        if node.type_.element_type is not None:
            self._semantic(node).extras[BUFFER_VIEW_ELEMENT_TYPE_EXTRA] = (
                node.type_.element_type
            )
        self._set_type(node, node.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewIndex) -> None:
        """
        title: Visit BufferViewIndex nodes.
        parameters:
          node:
            type: astx.BufferViewIndex
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        element_type = self._validate_buffer_view_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=False,
        )
        if element_type is not None:
            node.type_ = element_type
        self._set_type(node, element_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewStore) -> None:
        """
        title: Visit BufferViewStore nodes.
        parameters:
          node:
            type: astx.BufferViewStore
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        self.visit(node.value)
        element_type = self._validate_buffer_view_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=True,
        )
        if element_type is not None:
            validate_assignment(
                self.context.diagnostics,
                target_name="buffer view element",
                target_type=element_type,
                value_type=self._expr_type(node.value),
                node=node,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewWrite) -> None:
        """
        title: Visit BufferViewWrite nodes.
        parameters:
          node:
            type: astx.BufferViewWrite
        """
        if node.byte_offset < 0:
            self.context.diagnostics.add(
                "buffer view write byte_offset must be non-negative",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        self.visit(node.view)
        view_type = self._expr_type(node.view)
        if not isinstance(view_type, astx.BufferViewType):
            self.context.diagnostics.add(
                "buffer view write requires a BufferViewType view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        view_metadata = self._static_buffer_view_metadata(node.view)
        if view_metadata is not None and buffer_view_is_readonly(
            view_metadata.flags
        ):
            self.context.diagnostics.add(
                "cannot write through a readonly buffer view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        self.visit(node.value)
        value_type = self._expr_type(node.value)
        if (
            not is_integer_type(value_type)
            or bit_width(value_type) != RAW_BUFFER_BYTE_BITS
        ):
            self.context.diagnostics.add(
                "buffer view raw writes require an 8-bit integer value",
                node=node.value,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewRetain) -> None:
        """
        title: Visit BufferViewRetain nodes.
        parameters:
          node:
            type: astx.BufferViewRetain
        """
        self.visit(node.view)
        if not isinstance(self._expr_type(node.view), astx.BufferViewType):
            self.context.diagnostics.add(
                "buffer retain requires a BufferViewType view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._validate_buffer_lifetime_operation(
            node=node,
            view=node.view,
            operation="retain",
        )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewRelease) -> None:
        """
        title: Visit BufferViewRelease nodes.
        parameters:
          node:
            type: astx.BufferViewRelease
        """
        self.visit(node.view)
        if not isinstance(self._expr_type(node.view), astx.BufferViewType):
            self.context.diagnostics.add(
                "buffer release requires a BufferViewType view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._validate_buffer_lifetime_operation(
            node=node,
            view=node.view,
            operation="release",
        )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.AliasExpr) -> None:
        """
        title: Visit AliasExpr nodes.
        parameters:
          node:
            type: astx.AliasExpr
        """
        self._set_type(node, None)

    def _visit_temporal_literal(self, node: astx.AST) -> None:
        """
        title: Visit one temporal literal.
        parameters:
          node:
            type: astx.AST
        """
        try:
            literal_value = cast(str, getattr(node, "value"))
            parsed_value: object
            if isinstance(node, astx.LiteralTime):
                parsed_value = validate_literal_time(literal_value)
            elif isinstance(node, astx.LiteralTimestamp):
                parsed_value = validate_literal_timestamp(literal_value)
            else:
                parsed_value = validate_literal_datetime(literal_value)
            self._semantic(node).extras["parsed_value"] = parsed_value
        except ValueError as exc:
            self.context.diagnostics.add(str(exc), node=node)
        self._set_type(node, getattr(node, "type_", None))

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralTime) -> None:
        """
        title: Visit LiteralTime nodes.
        parameters:
          node:
            type: astx.LiteralTime
        """
        self._visit_temporal_literal(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralTimestamp) -> None:
        """
        title: Visit LiteralTimestamp nodes.
        parameters:
          node:
            type: astx.LiteralTimestamp
        """
        self._visit_temporal_literal(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralDateTime) -> None:
        """
        title: Visit LiteralDateTime nodes.
        parameters:
          node:
            type: astx.LiteralDateTime
        """
        self._visit_temporal_literal(node)

    def _visit_element_sequence_literal(self, node: astx.AST) -> None:
        """
        title: Visit one element-sequence literal.
        parameters:
          node:
            type: astx.AST
        """
        for element in cast(list[astx.AST], getattr(node, "elements")):
            self.visit(element)
        self._set_type(node, getattr(node, "type_", None))

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralList) -> None:
        """
        title: Visit LiteralList nodes.
        parameters:
          node:
            type: astx.LiteralList
        """
        self._visit_element_sequence_literal(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralTuple) -> None:
        """
        title: Visit LiteralTuple nodes.
        parameters:
          node:
            type: astx.LiteralTuple
        """
        self._visit_element_sequence_literal(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralSet) -> None:
        """
        title: Visit LiteralSet nodes.
        parameters:
          node:
            type: astx.LiteralSet
        """
        for element in node.elements:
            self.visit(element)
        if node.elements and not all(
            isinstance(element, astx.Literal) for element in node.elements
        ):
            self.context.diagnostics.add(
                "LiteralSet: only integer constants are "
                "currently supported for lowering",
                node=node,
            )
        self._set_type(
            node, cast(astx.DataType | None, getattr(node, "type_", None))
        )

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralDict) -> None:
        """
        title: Visit LiteralDict nodes.
        parameters:
          node:
            type: astx.LiteralDict
        """
        for key, value in node.elements.items():
            self.visit(key)
            self.visit(value)
        self._set_type(
            node, cast(astx.DataType | None, getattr(node, "type_", None))
        )

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ListCreate) -> None:
        """
        title: Visit ListCreate nodes.
        parameters:
          node:
            type: astx.ListCreate
        """
        self._set_type(node, node.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ListAppend) -> None:
        """
        title: Visit ListAppend nodes.
        parameters:
          node:
            type: astx.ListAppend
        """
        self.visit(node.base)
        self.visit(node.value)

        resolved_target = self._resolve_mutation_target(
            node.base,
            node=node,
            action="append to",
            invalid_message="list append target must be a variable or field",
        )
        if resolved_target is None:
            self._set_type(node, astx.Int32())
            return

        assignment_symbol, _target_name, target_type = resolved_target
        if not isinstance(target_type, astx.ListType):
            self.context.diagnostics.add(
                "list append requires a list target",
                node=node.base,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
            self._set_assignment(node, assignment_symbol)
            self._set_type(node, astx.Int32())
            return

        element_type = list_element_type(target_type)
        if element_type is None:
            self.context.diagnostics.add(
                "list append requires a single concrete list element type",
                node=node.base,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
            self._set_assignment(node, assignment_symbol)
            self._set_type(node, astx.Int32())
            return

        validate_assignment(
            self.context.diagnostics,
            target_name="list element",
            target_type=element_type,
            value_type=self._expr_type(node.value),
            node=node,
        )
        self._set_assignment(node, assignment_symbol)
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.SubscriptExpr) -> None:
        """
        title: Visit SubscriptExpr nodes.
        parameters:
          node:
            type: astx.SubscriptExpr
        """
        self.visit(node.value)
        if not isinstance(node.index, astx.LiteralNone):
            self.visit(node.index)
        value_type = self._expr_type(node.value)
        if isinstance(value_type, astx.ListType):
            if isinstance(node.index, astx.LiteralNone):
                self.context.diagnostics.add(
                    "list slicing is not supported",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
                self._set_type(node, None)
                return
            index_type = self._expr_type(node.index)
            if not is_integer_type(index_type):
                self.context.diagnostics.add(
                    "list indexing requires an integer index",
                    node=node.index,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            if not list_has_concrete_element_type(value_type):
                self.context.diagnostics.add(
                    "list indexing requires a single concrete list element "
                    "type",
                    node=node.value,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            self._set_type(node, list_element_type(value_type))
            return
        if isinstance(node.value, astx.LiteralDict):
            if not node.value.elements:
                self.context.diagnostics.add(
                    "SubscriptExpr: key lookup on empty dict",
                    node=node,
                )
            elif not isinstance(
                node.index,
                (
                    astx.LiteralInt8,
                    astx.LiteralInt16,
                    astx.LiteralInt32,
                    astx.LiteralInt64,
                    astx.LiteralUInt8,
                    astx.LiteralUInt16,
                    astx.LiteralUInt32,
                    astx.LiteralUInt64,
                    astx.LiteralFloat32,
                    astx.LiteralFloat64,
                    astx.Identifier,
                ),
            ):
                self.context.diagnostics.add(
                    "SubscriptExpr: only integer and floating-point "
                    "dict keys are supported",
                    node=node,
                )
        self._set_type(
            node,
            cast(
                astx.DataType | None,
                getattr(value_type, "value_type", None),
            ),
        )

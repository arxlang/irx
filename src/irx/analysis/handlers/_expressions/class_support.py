"""
title: Expression class-resolution helpers.
summary: >-
  Resolve class members, overload sets, and field metadata shared by the class-
  oriented expression visitors.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.handlers.class_helpers import (
    ClassMemberFormattingVisitorMixin,
)
from irx.analysis.resolved_nodes import (
    ClassMemberKind,
    SemanticClass,
    SemanticClassLayoutField,
    SemanticClassMember,
    SemanticClassStaticStorage,
    SemanticFunction,
)
from irx.analysis.types import display_type_name, same_type
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ExpressionClassSupportVisitorMixin(ClassMemberFormattingVisitorMixin):
    """
    title: Expression class-resolution helpers.
    """

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

    def _method_arguments_match_declared_defaults(
        self,
        member: SemanticClassMember,
        arg_types: list[astx.DataType | None],
    ) -> bool:
        """
        title: Return whether one method can accept omitted trailing defaults.
        parameters:
          member:
            type: SemanticClassMember
          arg_types:
            type: list[astx.DataType | None]
        returns:
          type: bool
        """
        declaration = member.declaration
        if member.signature is None or not isinstance(
            declaration, astx.FunctionDef
        ):
            return False
        provided_count = len(arg_types)
        fixed_param_count = len(member.signature.parameters)
        required_param_count = sum(
            1
            for argument in declaration.prototype.args.nodes
            if isinstance(argument.default, astx.Undefined)
        )
        if (
            provided_count < required_param_count
            or provided_count > fixed_param_count
        ):
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

        compatible_matches = tuple(
            candidate
            for candidate in accessible_candidates
            if self._method_arguments_match_declared_defaults(
                candidate,
                arg_types,
            )
        )
        if len(compatible_matches) == 1:
            return compatible_matches[0], accessible_candidates
        if len(compatible_matches) > 1:
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

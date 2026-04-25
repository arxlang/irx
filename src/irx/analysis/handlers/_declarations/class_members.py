# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=untyped-decorator

"""
title: Declaration class-member resolution helpers.
summary: >-
  Resolve declared and effective class members, then finish class-definition
  analysis from the normalized declaration metadata.
"""

from __future__ import annotations

from dataclasses import replace

from irx import astx
from irx.analysis.handlers._declarations.class_methods import (
    IMPLICIT_METHOD_RECEIVER_NAME,
    DeclarationClassMethodVisitorMixin,
)
from irx.analysis.handlers.base import SemanticAnalyzerCore
from irx.analysis.resolved_nodes import (
    ClassMemberKind,
    ClassMemberResolutionKind,
    SemanticClass,
    SemanticClassMember,
    SemanticClassMemberResolution,
)
from irx.analysis.validation import validate_assignment
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class DeclarationClassMemberVisitorMixin(DeclarationClassMethodVisitorMixin):
    """
    title: Declaration class-member resolution helpers.
    """

    def _resolve_declared_class_members(
        self,
        class_: SemanticClass,
        mro: tuple[SemanticClass, ...],
    ) -> tuple[SemanticClassMember, ...]:
        """
        title: Resolve the declared member set for one class.
        parameters:
          class_:
            type: SemanticClass
          mro:
            type: tuple[SemanticClass, Ellipsis]
        returns:
          type: tuple[SemanticClassMember, Ellipsis]
        """
        attribute_names: set[str] = set()
        method_groups: dict[str, list[SemanticClassMember]] = {}
        members: list[SemanticClassMember] = []
        ancestor_attributes = self._ancestor_declared_attributes(mro)
        ancestor_methods = self._ancestor_declared_method_groups(mro)

        for attribute in class_.declaration.attributes:
            if (
                attribute.name in attribute_names
                or attribute.name in method_groups
            ):
                self.context.diagnostics.add(
                    (
                        f"Class member '{attribute.name}' already defined "
                        f"in '{class_.name}'"
                    ),
                    node=attribute,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            attribute_names.add(attribute.name)
            self._resolve_declared_type(
                attribute.type_,
                node=attribute,
                unknown_message="Unknown attribute type '{name}'",
            )
            if attribute.value is not None and not isinstance(
                attribute.value,
                astx.Undefined,
            ):
                self.visit(attribute.value)
                if self._require_value_expression(
                    attribute.value,
                    context=(
                        f"Initializer for '{class_.name}.{attribute.name}'"
                    ),
                ):
                    validate_assignment(
                        self.context.diagnostics,
                        target_name=f"{class_.name}.{attribute.name}",
                        target_type=attribute.type_,
                        value_type=self._expr_type(attribute.value),
                        node=attribute,
                    )
            is_constant = attribute.mutability == astx.MutabilityKind.constant
            if is_constant and (
                attribute.value is None
                or isinstance(attribute.value, astx.Undefined)
            ):
                self.context.diagnostics.add(
                    (
                        "Constant attribute "
                        f"'{class_.name}.{attribute.name}' "
                        "requires an initializer"
                    ),
                    node=attribute,
                    code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
                )
            if self._attribute_is_static(attribute) and not (
                self._static_initializer_is_supported(attribute.value)
            ):
                self.context.diagnostics.add(
                    (
                        "Static attribute "
                        f"'{class_.name}.{attribute.name}' requires "
                        "a literal initializer or default construction"
                    ),
                    node=attribute,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            if (
                attribute.name in ancestor_attributes
                or attribute.name in ancestor_methods
            ):
                self.context.diagnostics.add(
                    (
                        f"Class '{class_.name}' cannot redeclare inherited "
                        f"member '{attribute.name}'"
                    ),
                    node=attribute,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            members.append(
                self.factory.make_class_member(
                    class_,
                    name=attribute.name,
                    kind=ClassMemberKind.ATTRIBUTE,
                    declaration=attribute,
                    visibility=attribute.visibility,
                    is_static=self._attribute_is_static(attribute),
                    is_constant=is_constant,
                    is_mutable=not is_constant,
                    type_=attribute.type_,
                )
            )

        for method in class_.declaration.methods:
            if method.name in attribute_names:
                self.context.diagnostics.add(
                    (
                        f"Class member '{method.name}' already defined in "
                        f"'{class_.name}'"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            for argument in method.prototype.args.nodes:
                self._resolve_declared_type(argument.type_, node=argument)
            self._resolve_declared_type(
                method.prototype.return_type,
                node=method,
            )
            signature = self._normalize_class_method_signature(class_, method)
            is_static = self._method_is_static(method)
            is_abstract = self._method_is_abstract(method)
            signature_key = self._method_signature_key(
                method.name,
                signature,
                is_static=is_static,
            )
            call_key = self._method_call_key_from_signature(
                method.name,
                signature,
                is_static=is_static,
            )
            if not is_static and any(
                argument.name == IMPLICIT_METHOD_RECEIVER_NAME
                for argument in method.prototype.args.nodes
            ):
                self.context.diagnostics.add(
                    (
                        f"Class method '{class_.name}.{method.name}' "
                        "cannot declare parameter '"
                        f"{IMPLICIT_METHOD_RECEIVER_NAME}'"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            if is_abstract and (
                method.prototype.visibility is astx.VisibilityKind.private
            ):
                self.context.diagnostics.add(
                    (
                        f"Class method '{class_.name}.{method.name}' "
                        "cannot be both abstract and private"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            if is_abstract and astx.is_template_node(method.prototype):
                self.context.diagnostics.add(
                    (
                        f"Class method '{class_.name}.{method.name}' "
                        "cannot be both abstract and templated"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            if is_abstract and len(method.body.nodes) > 0:
                self.context.diagnostics.add(
                    (
                        f"Abstract class method "
                        f"'{class_.name}.{method.name}' must not declare "
                        "a body"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            local_group = method_groups.setdefault(method.name, [])
            if any(
                existing.is_static != is_static for existing in local_group
            ):
                self.context.diagnostics.add(
                    (
                        f"Class method '{class_.name}.{method.name}' "
                        "cannot mix static and instance overloads"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            if any(
                self._member_call_key(existing) == call_key
                for existing in local_group
            ):
                if any(
                    existing.signature_key == signature_key
                    for existing in local_group
                ):
                    self.context.diagnostics.add(
                        (
                            f"Class method '{class_.name}.{method.name}' "
                            "already defines this exact signature"
                        ),
                        node=method,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                else:
                    self.context.diagnostics.add(
                        (
                            f"Class method '{class_.name}.{method.name}' "
                            "cannot overload only by return type"
                        ),
                        node=method,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                continue
            if method.name in ancestor_attributes:
                self.context.diagnostics.add(
                    (
                        f"Class method '{class_.name}.{method.name}' "
                        "cannot override inherited attribute "
                        f"'{method.name}'"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            inherited_methods = ancestor_methods.get(method.name, ())
            if any(
                inherited.is_static != is_static
                for inherited in inherited_methods
            ):
                self.context.diagnostics.add(
                    (
                        f"Class method '{class_.name}.{method.name}' "
                        "changes static/instance status across "
                        "inheritance"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            inherited_same_call = tuple(
                inherited
                for inherited in inherited_methods
                if self._member_call_key(inherited) == call_key
            )
            inherited_same_signature = tuple(
                inherited
                for inherited in inherited_same_call
                if inherited.signature_key == signature_key
            )
            if inherited_same_call and not inherited_same_signature:
                self.context.diagnostics.add(
                    (
                        f"Class method '{class_.name}.{method.name}' "
                        "cannot overload inherited methods only by "
                        "return type"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            if any(
                self._visibility_rank(method.prototype.visibility)
                < self._visibility_rank(inherited.visibility)
                for inherited in inherited_same_signature
            ):
                self.context.diagnostics.add(
                    (
                        f"Class method '{class_.name}.{method.name}' "
                        "cannot reduce visibility when overriding "
                        f"'{method.name}'"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            lowered_function = self._make_lowered_method_function(
                class_,
                method,
                signature,
                is_static=is_static,
                signature_key=signature_key,
            )
            member = self.factory.make_class_member(
                class_,
                name=method.name,
                kind=ClassMemberKind.METHOD,
                declaration=method,
                visibility=method.prototype.visibility,
                is_static=is_static,
                is_abstract=is_abstract,
                is_constant=True,
                is_mutable=False,
                signature=signature,
                signature_key=signature_key,
                overrides=(
                    inherited_same_signature[0].qualified_name
                    if inherited_same_signature
                    else None
                ),
                lowered_function=lowered_function,
            )
            self._set_class(method.prototype, class_)
            self._set_class(method, class_)
            self._set_function(method.prototype, lowered_function)
            self._set_function(method, lowered_function)
            self._set_type(method.prototype, None)
            self._set_type(method, None)
            local_group.append(member)
            members.append(member)

        return tuple(members)

    def _resolve_effective_class_members(
        self,
        class_: SemanticClass,
        mro: tuple[SemanticClass, ...],
        declared_member_table: dict[str, SemanticClassMember],
        declared_method_groups: dict[str, tuple[SemanticClassMember, ...]],
    ) -> tuple[
        dict[str, SemanticClassMember],
        dict[str, tuple[SemanticClassMember, ...]],
        dict[str, SemanticClassMemberResolution],
        dict[str, tuple[SemanticClassMemberResolution, ...]],
    ]:
        """
        title: Resolve the effective class members for one class.
        parameters:
          class_:
            type: SemanticClass
          mro:
            type: tuple[SemanticClass, Ellipsis]
          declared_member_table:
            type: dict[str, SemanticClassMember]
          declared_method_groups:
            type: dict[str, tuple[SemanticClassMember, Ellipsis]]
        returns:
          type: >-
            tuple[dict[str, SemanticClassMember], dict[str,
            tuple[SemanticClassMember, Ellipsis]], dict[str,
            SemanticClassMemberResolution], dict[str,
            tuple[SemanticClassMemberResolution, Ellipsis]]]
        """
        ancestor_attributes = self._ancestor_declared_attributes(mro)
        ancestor_methods = self._ancestor_declared_method_groups(mro)
        effective_member_table: dict[str, SemanticClassMember] = {}
        effective_method_groups: dict[
            str, tuple[SemanticClassMember, ...]
        ] = {}
        member_resolution: dict[str, SemanticClassMemberResolution] = {}
        method_resolution: dict[
            str, tuple[SemanticClassMemberResolution, ...]
        ] = {}

        for name, member in declared_member_table.items():
            if member.kind is not ClassMemberKind.ATTRIBUTE:
                continue
            inherited_candidates = ancestor_attributes.get(name, ())
            member_resolution[name] = SemanticClassMemberResolution(
                name=name,
                kind=ClassMemberResolutionKind.DECLARED,
                selected=member,
                candidates=(member, *inherited_candidates),
            )
            effective_member_table[name] = member

        for name, candidates in ancestor_attributes.items():
            if (
                name in effective_member_table
                or name in declared_method_groups
            ):
                continue
            if name in ancestor_methods:
                owner_names = ", ".join(
                    member.owner_name
                    for member in (*candidates, *ancestor_methods[name])
                )
                self.context.diagnostics.add(
                    (
                        f"Class '{class_.name}' inherits conflicting "
                        f"members named '{name}' from {owner_names}"
                    ),
                    node=class_.declaration,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            if len(candidates) != 1:
                owner_names = ", ".join(
                    member.owner_name for member in candidates
                )
                self.context.diagnostics.add(
                    (
                        f"Class '{class_.name}' inherits ambiguous "
                        f"attribute '{name}' from {owner_names}"
                    ),
                    node=class_.declaration,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            effective_member_table[name] = candidates[0]
            member_resolution[name] = SemanticClassMemberResolution(
                name=name,
                kind=ClassMemberResolutionKind.INHERITED,
                selected=candidates[0],
                candidates=candidates,
            )

        method_name_order = list(declared_method_groups)
        for name in ancestor_methods:
            if name not in method_name_order:
                method_name_order.append(name)

        for name in method_name_order:
            local_group = declared_method_groups.get(name, ())
            inherited_candidates = ancestor_methods.get(name, ())
            if not local_group and not inherited_candidates:
                continue
            if name in ancestor_attributes and not local_group:
                owner_names = ", ".join(
                    member.owner_name
                    for member in (
                        *ancestor_attributes[name],
                        *inherited_candidates,
                    )
                )
                self.context.diagnostics.add(
                    (
                        f"Class '{class_.name}' inherits conflicting "
                        f"members named '{name}' from {owner_names}"
                    ),
                    node=class_.declaration,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            staticness = {
                member.is_static
                for member in (*local_group, *inherited_candidates)
            }
            if len(staticness) > 1:
                self.context.diagnostics.add(
                    (
                        f"Class '{class_.name}' inherits incompatible "
                        f"static and instance methods named '{name}'"
                    ),
                    node=class_.declaration,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue

            inherited_by_call: dict[str, list[SemanticClassMember]] = {}
            inherited_call_order: list[str] = []
            for member in inherited_candidates:
                call_key = self._member_call_key(member)
                if call_key not in inherited_by_call:
                    inherited_call_order.append(call_key)
                    inherited_by_call[call_key] = []
                inherited_by_call[call_key].append(member)

            selected_group: list[SemanticClassMember] = []
            resolutions: list[SemanticClassMemberResolution] = []
            local_call_keys: set[str] = set()
            for member in local_group:
                call_key = self._member_call_key(member)
                local_call_keys.add(call_key)
                inherited_same_call = tuple(
                    inherited_by_call.get(call_key, ())
                )
                if inherited_same_call and any(
                    candidate.signature_key != member.signature_key
                    for candidate in inherited_same_call
                ):
                    self.context.diagnostics.add(
                        (
                            f"Class method '{class_.name}.{member.name}' "
                            "conflicts with inherited call signatures"
                        ),
                        node=member.declaration,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                    continue
                resolution_kind = (
                    ClassMemberResolutionKind.OVERRIDE
                    if inherited_same_call
                    else ClassMemberResolutionKind.DECLARED
                )
                resolutions.append(
                    SemanticClassMemberResolution(
                        name=name,
                        kind=resolution_kind,
                        selected=member,
                        candidates=(member, *inherited_same_call),
                        signature_key=member.signature_key,
                    )
                )
                selected_group.append(member)

            for call_key in inherited_call_order:
                if call_key in local_call_keys:
                    continue
                candidates = tuple(inherited_by_call[call_key])
                if (
                    len({candidate.signature_key for candidate in candidates})
                    > 1
                ):
                    owner_names = ", ".join(
                        candidate.owner_name for candidate in candidates
                    )
                    self.context.diagnostics.add(
                        (
                            f"Class '{class_.name}' inherits conflicting "
                            f"methods named '{name}' from {owner_names}"
                        ),
                        node=class_.declaration,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                    continue
                dominant = self._dominant_inherited_method(candidates)
                if dominant is None:
                    owner_names = ", ".join(
                        candidate.owner_name for candidate in candidates
                    )
                    self.context.diagnostics.add(
                        (
                            f"Class '{class_.name}' inherits conflicting "
                            f"methods named '{name}' from {owner_names}"
                        ),
                        node=class_.declaration,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                    continue
                resolutions.append(
                    SemanticClassMemberResolution(
                        name=name,
                        kind=ClassMemberResolutionKind.INHERITED,
                        selected=dominant,
                        candidates=candidates,
                        signature_key=dominant.signature_key,
                    )
                )
                selected_group.append(dominant)

            if not selected_group:
                continue
            effective_method_groups[name] = tuple(selected_group)
            method_resolution[name] = tuple(resolutions)
            if len(selected_group) == 1:
                effective_member_table[name] = selected_group[0]
                member_resolution[name] = resolutions[0]

        return (
            effective_member_table,
            effective_method_groups,
            member_resolution,
            method_resolution,
        )

    def _resolve_class_structure(
        self,
        class_: SemanticClass,
    ) -> SemanticClass:
        """
        title: Resolve one class declaration up to structural metadata.
        parameters:
          class_:
            type: SemanticClass
        returns:
          type: SemanticClass
        """
        current = self.context.get_class(class_.module_key, class_.name)
        if current is None:
            current = class_
            self.context.register_class(current)
        if current.is_structurally_resolved:
            return current

        active_stack = self._class_resolution_stack()
        if any(
            item.qualified_name == current.qualified_name
            for item in active_stack
        ):
            cycle_names = " -> ".join(
                [item.name for item in active_stack] + [current.name]
            )
            self.context.diagnostics.add(
                f"inheritance cycle is invalid: {cycle_names}",
                node=current.declaration,
                code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
            )
            return current

        active_stack.append(current)
        try:
            bases = self._resolve_class_bases(current)
            raw_mro = self._compute_class_mro(current, bases)
            mro = (
                current,
                *tuple(
                    self.context.get_class(item.module_key, item.name) or item
                    for item in raw_mro[1:]
                ),
            )
            declared_members = self._resolve_declared_class_members(
                current,
                mro,
            )
            declared_member_table, declared_method_groups = (
                self._declared_member_tables(declared_members)
            )
            updated = replace(
                current,
                bases=bases,
                declared_members=declared_members,
                declared_member_table=declared_member_table,
                declared_method_groups=declared_method_groups,
                inheritance_graph=tuple(
                    ancestor.qualified_name for ancestor in mro[1:]
                ),
                shared_ancestors=self._shared_class_ancestors(bases, mro),
                mro=(current, *mro[1:]),
                is_structurally_resolved=True,
            )
            updated = replace(updated, mro=(updated, *updated.mro[1:]))
            self.context.register_class(updated)
            self.bindings.bind_class(
                updated.name,
                updated,
                node=updated.declaration,
            )
            self._set_class(updated.declaration, updated)
            return updated
        finally:
            active_stack.pop()

    def _resolve_class_definition(
        self,
        class_: SemanticClass,
    ) -> SemanticClass:
        """
        title: Resolve one class declaration into normalized semantic metadata.
        parameters:
          class_:
            type: SemanticClass
        returns:
          type: SemanticClass
        """
        current = self.context.get_class(class_.module_key, class_.name)
        if current is None:
            current = class_
            self.context.register_class(current)
        if current.is_resolved:
            return current

        current = self._resolve_class_structure(current)
        self._finalize_registered_class_layouts()
        return (
            self.context.get_class(current.module_key, current.name) or current
        )

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ClassDefStmt) -> None:
        """
        title: Visit ClassDefStmt nodes.
        parameters:
          node:
            type: astx.ClassDefStmt
        """
        class_ = self.registry.register_class(node)
        self.bindings.bind_class(node.name, class_, node=node)
        with self.context.in_class(class_):
            class_ = self._resolve_class_definition(class_)
        with self.context.in_class(class_):
            for member in class_.declared_members:
                if member.kind is not ClassMemberKind.METHOD:
                    continue
                self._analyze_class_method_body(class_, member)
        self._set_class(node, class_)
        self._set_type(node, None)

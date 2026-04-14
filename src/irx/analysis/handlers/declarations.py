# mypy: disable-error-code=no-redef

"""
title: Declaration-oriented semantic visitors.
summary: >-
  Handle modules, functions, structs, and lexical declarations while delegating
  semantic entity creation and registration to shared infrastructure.
"""

from __future__ import annotations

from dataclasses import replace

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.module_symbols import (
    mangle_class_descriptor_name,
    mangle_class_dispatch_name,
    mangle_class_name,
    mangle_class_static_name,
    qualified_local_name,
)
from irx.analysis.resolved_nodes import (
    ClassHeaderFieldKind,
    ClassMemberKind,
    ClassMemberResolutionKind,
    ClassObjectRepresentationKind,
    FunctionSignature,
    SemanticClass,
    SemanticClassHeaderField,
    SemanticClassLayout,
    SemanticClassLayoutField,
    SemanticClassMember,
    SemanticClassMemberResolution,
    SemanticClassStaticStorage,
    SemanticFunction,
    SemanticStruct,
    SemanticStructField,
)
from irx.analysis.types import clone_type
from irx.analysis.validation import validate_assignment
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked

DIRECT_STRUCT_CYCLE_LENGTH = 2
MIN_SHARED_ANCESTOR_BASE_COUNT = 2
_CLASS_HEADER_LAYOUT: tuple[tuple[str, ClassHeaderFieldKind], ...] = (
    ("descriptor", ClassHeaderFieldKind.TYPE_DESCRIPTOR),
    ("dispatch", ClassHeaderFieldKind.DISPATCH_TABLE),
)


@typechecked
class DeclarationVisitorMixin(SemanticVisitorMixinBase):
    def _synchronize_function_signature(
        self,
        function: SemanticFunction,
        prototype: astx.FunctionPrototype,
        *,
        definition: astx.FunctionDef | None = None,
    ) -> SemanticFunction:
        """
        title: Synchronize one semantic function with resolved AST types.
        parameters:
          function:
            type: SemanticFunction
          prototype:
            type: astx.FunctionPrototype
          definition:
            type: astx.FunctionDef | None
        returns:
          type: SemanticFunction
        """
        signature = self.registry.normalize_function_signature(
            prototype,
            definition=definition,
        )
        if (
            function.prototype is not prototype
            and definition is None
            and not self.registry.signatures_match(
                function.signature, signature
            )
        ):
            return function
        if (
            definition is not None
            and function.definition is not None
            and function.definition is not definition
            and not self.registry.signatures_match(
                function.signature, signature
            )
        ):
            return function

        updated = replace(
            function,
            return_type=clone_type(signature.return_type),
            args=tuple(
                replace(
                    arg_symbol,
                    name=arg_node.name,
                    type_=clone_type(arg_node.type_),
                    qualified_name=qualified_local_name(
                        function.module_key,
                        arg_symbol.kind,
                        arg_node.name,
                        arg_symbol.symbol_id,
                    ),
                )
                for arg_node, arg_symbol in zip(
                    prototype.args.nodes,
                    function.args,
                )
            ),
            signature=signature,
            prototype=prototype,
            definition=(
                definition if definition is not None else function.definition
            ),
        )
        self.context.register_function(updated)
        return updated

    def _resolve_struct_fields(
        self,
        struct: SemanticStruct,
    ) -> SemanticStruct:
        """
        title: Resolve one struct's ordered field metadata.
        parameters:
          struct:
            type: SemanticStruct
        returns:
          type: SemanticStruct
        """
        seen: set[str] = set()
        fields: list[SemanticStructField] = []

        if len(list(struct.declaration.attributes)) == 0:
            self.context.diagnostics.add(
                f"Struct '{struct.name}' must declare at least one field",
                node=struct.declaration,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )

        for index, attr in enumerate(struct.declaration.attributes):
            if attr.name in seen:
                self.context.diagnostics.add(
                    f"Struct field '{attr.name}' already defined.",
                    node=attr,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
            seen.add(attr.name)
            self._resolve_declared_type(
                attr.type_,
                node=attr,
                unknown_message="Unknown field type '{name}'",
            )
            fields.append(
                SemanticStructField(
                    name=attr.name,
                    index=index,
                    type_=clone_type(attr.type_),
                    declaration=attr,
                )
            )

        updated = replace(
            struct,
            fields=tuple(fields),
            field_indices={field.name: field.index for field in fields},
        )
        self.context.register_struct(updated)
        self.bindings.bind_struct(
            updated.name,
            updated,
            node=updated.declaration,
        )
        self._set_struct(updated.declaration, updated)
        return updated

    def _find_struct_cycle(
        self,
        root: SemanticStruct,
        current: SemanticStruct,
        path: tuple[SemanticStruct, ...],
    ) -> tuple[SemanticStruct, ...] | None:
        """
        title: Find one by-value recursive struct cycle.
        parameters:
          root:
            type: SemanticStruct
          current:
            type: SemanticStruct
          path:
            type: tuple[SemanticStruct, Ellipsis]
        returns:
          type: tuple[SemanticStruct, Ellipsis] | None
        """
        seen = {struct.qualified_name for struct in path}
        for attr in current.declaration.attributes:
            field_struct = self._resolve_struct_from_type(
                attr.type_,
                node=attr,
                unknown_message="Unknown field type '{name}'",
            )
            if field_struct is None:
                continue
            if field_struct.qualified_name == root.qualified_name:
                return (*path, field_struct)
            if field_struct.qualified_name in seen:
                continue
            cycle = self._find_struct_cycle(
                root,
                field_struct,
                (*path, field_struct),
            )
            if cycle is not None:
                return cycle
        return None

    def _validate_struct_cycles(self, struct: SemanticStruct) -> None:
        """
        title: Reject by-value recursive struct layouts.
        parameters:
          struct:
            type: SemanticStruct
        """
        cycle = self._find_struct_cycle(struct, struct, (struct,))
        if cycle is None:
            return

        if len(cycle) == DIRECT_STRUCT_CYCLE_LENGTH:
            self.context.diagnostics.add(
                (
                    "direct by-value recursive struct "
                    f"'{struct.name}' is forbidden"
                ),
                node=struct.declaration,
            )
            return

        cycle_names = " -> ".join(item.name for item in cycle)
        self.context.diagnostics.add(
            f"mutual by-value recursive structs are forbidden: {cycle_names}",
            node=struct.declaration,
        )

    def _visibility_rank(
        self,
        visibility: astx.VisibilityKind,
    ) -> int:
        """
        title: Return one normalized visibility rank.
        parameters:
          visibility:
            type: astx.VisibilityKind
        returns:
          type: int
        """
        return {
            astx.VisibilityKind.private: 1,
            astx.VisibilityKind.protected: 2,
            astx.VisibilityKind.public: 3,
        }[visibility]

    def _class_resolution_stack(self) -> list[SemanticClass]:
        """
        title: Return the active class-resolution stack.
        returns:
          type: list[SemanticClass]
        """
        stack = getattr(self, "_active_class_resolution", None)
        if isinstance(stack, list):
            return stack
        created: list[SemanticClass] = []
        setattr(self, "_active_class_resolution", created)
        return created

    def _attribute_is_static(
        self,
        declaration: astx.VariableDeclaration,
    ) -> bool:
        """
        title: Return whether one class attribute uses static storage.
        parameters:
          declaration:
            type: astx.VariableDeclaration
        returns:
          type: bool
        """
        raw_value = getattr(declaration, "is_static", None)
        if isinstance(raw_value, bool):
            return raw_value
        return bool(declaration.scope == astx.ScopeKind.global_)

    def _method_is_static(
        self,
        declaration: astx.FunctionDef,
    ) -> bool:
        """
        title: Return whether one class method is static.
        parameters:
          declaration:
            type: astx.FunctionDef
        returns:
          type: bool
        """
        raw_value = getattr(declaration.prototype, "is_static", None)
        if isinstance(raw_value, bool):
            return raw_value
        return False

    def _resolve_class_bases(
        self,
        class_: SemanticClass,
    ) -> tuple[SemanticClass, ...]:
        """
        title: Resolve the direct base-class list for one class.
        parameters:
          class_:
            type: SemanticClass
        returns:
          type: tuple[SemanticClass, Ellipsis]
        """
        resolved: list[SemanticClass] = []
        seen: set[str] = set()
        active_stack = self._class_resolution_stack()
        active_identities = {item.qualified_name for item in active_stack}

        for base_type in class_.declaration.bases:
            base = self._resolve_class_from_type(
                base_type,
                node=base_type,
                unknown_message="Unknown base class '{name}'",
            )
            if base is None:
                continue
            if base.qualified_name == class_.qualified_name:
                self.context.diagnostics.add(
                    f"Class '{class_.name}' cannot inherit from itself",
                    node=base_type,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            if base.qualified_name in seen:
                self.context.diagnostics.add(
                    f"Class '{class_.name}' repeats base class '{base.name}'",
                    node=base_type,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            if base.qualified_name in active_identities:
                cycle_names = " -> ".join(
                    [item.name for item in active_stack] + [base.name]
                )
                self.context.diagnostics.add(
                    f"inheritance cycle is invalid: {cycle_names}",
                    node=base_type,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            seen.add(base.qualified_name)
            resolved.append(self._resolve_class_definition(base))

        return tuple(resolved)

    def _compute_class_mro(
        self,
        class_: SemanticClass,
        bases: tuple[SemanticClass, ...],
    ) -> tuple[SemanticClass, ...]:
        """
        title: Compute one deterministic class linearization.
        parameters:
          class_:
            type: SemanticClass
          bases:
            type: tuple[SemanticClass, Ellipsis]
        returns:
          type: tuple[SemanticClass, Ellipsis]
        """
        if not bases:
            return (class_,)

        sequences: list[list[SemanticClass]] = [
            list(base.mro or (base,)) for base in bases
        ] + [list(bases)]
        result: list[SemanticClass] = [class_]

        while True:
            sequences = [sequence for sequence in sequences if sequence]
            if not sequences:
                return tuple(result)

            candidate: SemanticClass | None = None
            for sequence in sequences:
                head = sequence[0]
                if any(
                    head.qualified_name == later.qualified_name
                    for other in sequences
                    for later in other[1:]
                ):
                    continue
                candidate = head
                break

            if candidate is None:
                base_names = ", ".join(base.name for base in bases)
                self.context.diagnostics.add(
                    (
                        f"Class '{class_.name}' has no consistent MRO "
                        f"for bases {base_names}"
                    ),
                    node=class_.declaration,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                return (class_, *bases)

            result.append(candidate)
            for sequence in sequences:
                if sequence and (
                    sequence[0].qualified_name == candidate.qualified_name
                ):
                    sequence.pop(0)

    def _shared_class_ancestors(
        self,
        bases: tuple[SemanticClass, ...],
        mro: tuple[SemanticClass, ...],
    ) -> tuple[SemanticClass, ...]:
        """
        title: Return ancestors reachable through more than one base lineage.
        parameters:
          bases:
            type: tuple[SemanticClass, Ellipsis]
          mro:
            type: tuple[SemanticClass, Ellipsis]
        returns:
          type: tuple[SemanticClass, Ellipsis]
        """
        if len(bases) < MIN_SHARED_ANCESTOR_BASE_COUNT:
            return ()

        counts: dict[str, int] = {}
        for base in bases:
            lineage_seen: set[str] = set()
            for ancestor in base.mro or (base,):
                if ancestor.qualified_name in lineage_seen:
                    continue
                lineage_seen.add(ancestor.qualified_name)
                counts[ancestor.qualified_name] = (
                    counts.get(ancestor.qualified_name, 0) + 1
                )

        return tuple(
            ancestor
            for ancestor in mro[1:]
            if counts.get(ancestor.qualified_name, 0) > 1
        )

    def _class_storage_ancestors(
        self,
        bases: tuple[SemanticClass, ...],
    ) -> tuple[SemanticClass, ...]:
        """
        title: Return the canonical ancestor storage order for one class.
        parameters:
          bases:
            type: tuple[SemanticClass, Ellipsis]
        returns:
          type: tuple[SemanticClass, Ellipsis]
        """
        ordered: list[SemanticClass] = []
        seen: set[str] = set()

        def visit(base: SemanticClass) -> None:
            """
            title: Visit one base lineage for canonical storage ordering.
            parameters:
              base:
                type: SemanticClass
            """
            for ancestor in base.bases:
                visit(ancestor)
            if base.qualified_name in seen:
                return
            seen.add(base.qualified_name)
            ordered.append(base)

        for base in bases:
            visit(base)
        return tuple(ordered)

    def _build_class_layout(
        self,
        class_: SemanticClass,
        bases: tuple[SemanticClass, ...],
        declared_members: tuple[SemanticClassMember, ...],
        effective_members: tuple[SemanticClassMember, ...],
    ) -> SemanticClassLayout:
        """
        title: Build one canonical low-level class layout.
        parameters:
          class_:
            type: SemanticClass
          bases:
            type: tuple[SemanticClass, Ellipsis]
          declared_members:
            type: tuple[SemanticClassMember, Ellipsis]
          effective_members:
            type: tuple[SemanticClassMember, Ellipsis]
        returns:
          type: SemanticClassLayout
        """
        header_fields = tuple(
            SemanticClassHeaderField(
                name=name,
                kind=kind,
                storage_index=index,
            )
            for index, (name, kind) in enumerate(_CLASS_HEADER_LAYOUT)
        )
        storage_ancestors = self._class_storage_ancestors(bases)
        instance_fields: list[SemanticClassLayoutField] = []
        field_slots: dict[str, SemanticClassLayoutField] = {}

        for storage_owner, owner_members in (
            *tuple(
                (
                    ancestor,
                    tuple(
                        member
                        for member in ancestor.declared_members
                        if member.kind is ClassMemberKind.ATTRIBUTE
                        and not member.is_static
                    ),
                )
                for ancestor in storage_ancestors
            ),
            (
                class_,
                tuple(
                    member
                    for member in declared_members
                    if member.kind is ClassMemberKind.ATTRIBUTE
                    and not member.is_static
                ),
            ),
        ):
            for member in owner_members:
                logical_index = len(instance_fields)
                slot = SemanticClassLayoutField(
                    member=member,
                    logical_index=logical_index,
                    storage_index=len(header_fields) + logical_index,
                    owner_name=storage_owner.name,
                    owner_qualified_name=storage_owner.qualified_name,
                )
                instance_fields.append(slot)
                field_slots[member.qualified_name] = slot

        static_fields = tuple(
            SemanticClassStaticStorage(
                member=member,
                global_name=mangle_class_static_name(
                    class_.module_key,
                    class_.name,
                    member.name,
                ),
                owner_name=class_.name,
                owner_qualified_name=class_.qualified_name,
            )
            for member in declared_members
            if member.kind is ClassMemberKind.ATTRIBUTE and member.is_static
        )
        static_storage = {
            storage.member.qualified_name: storage for storage in static_fields
        }
        inherited_static_storage: dict[str, SemanticClassStaticStorage] = {}
        for ancestor in storage_ancestors:
            layout = ancestor.layout
            if layout is None:
                continue
            inherited_static_storage.update(layout.static_storage)
        inherited_static_storage.update(static_storage)

        visible_field_slots: dict[str, SemanticClassLayoutField] = {}
        visible_static_storage: dict[str, SemanticClassStaticStorage] = {}
        for member in effective_members:
            if member.kind is not ClassMemberKind.ATTRIBUTE:
                continue
            if member.is_static:
                storage = inherited_static_storage.get(member.qualified_name)
                if storage is not None:
                    visible_static_storage[member.name] = storage
                continue
            visible_slot = field_slots.get(member.qualified_name)
            if visible_slot is not None:
                visible_field_slots[member.name] = visible_slot

        return SemanticClassLayout(
            llvm_name=mangle_class_name(class_.module_key, class_.name),
            object_representation=ClassObjectRepresentationKind.POINTER,
            descriptor_global_name=mangle_class_descriptor_name(
                class_.module_key,
                class_.name,
            ),
            dispatch_global_name=mangle_class_dispatch_name(
                class_.module_key,
                class_.name,
            ),
            header_fields=header_fields,
            instance_fields=tuple(instance_fields),
            field_slots=field_slots,
            visible_field_slots=visible_field_slots,
            static_fields=static_fields,
            static_storage=static_storage,
            visible_static_storage=visible_static_storage,
        )

    def _ancestor_declared_members(
        self,
        mro: tuple[SemanticClass, ...],
    ) -> dict[str, list[SemanticClassMember]]:
        """
        title: Collect visible ancestor member declarations in MRO order.
        parameters:
          mro:
            type: tuple[SemanticClass, Ellipsis]
        returns:
          type: dict[str, list[SemanticClassMember]]
        """
        members: dict[str, list[SemanticClassMember]] = {}
        for ancestor in mro[1:]:
            for name, member in ancestor.declared_member_table.items():
                if member.visibility is astx.VisibilityKind.private:
                    continue
                members.setdefault(name, []).append(member)
        return members

    def _normalize_class_method_signature(
        self,
        class_: SemanticClass,
        declaration: astx.FunctionDef,
    ) -> FunctionSignature:
        """
        title: Normalize one class-method signature.
        parameters:
          class_:
            type: SemanticClass
          declaration:
            type: astx.FunctionDef
        returns:
          type: FunctionSignature
        """
        signature = self.registry.normalize_function_signature(
            declaration.prototype,
            definition=declaration,
            validate_ffi=False,
            validate_main=False,
        )
        if signature.is_extern:
            self.context.diagnostics.add(
                (
                    f"Class method '{class_.name}.{declaration.name}' "
                    "cannot be extern"
                ),
                node=declaration,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
        if signature.is_variadic:
            self.context.diagnostics.add(
                (
                    f"Class method '{class_.name}.{declaration.name}' "
                    "must not be variadic"
                ),
                node=declaration,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
        return signature

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
        seen_names: set[str] = set()
        members: list[SemanticClassMember] = []
        ancestors = self._ancestor_declared_members(mro)

        for attribute in class_.declaration.attributes:
            if attribute.name in seen_names:
                self.context.diagnostics.add(
                    (
                        f"Class member '{attribute.name}' already defined "
                        f"in '{class_.name}'"
                    ),
                    node=attribute,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            seen_names.add(attribute.name)
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
            if attribute.name in ancestors:
                self.context.diagnostics.add(
                    (
                        f"Class '{class_.name}' cannot redeclare inherited "
                        f"member '{attribute.name}'"
                    ),
                    node=attribute,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
            member = self.factory.make_class_member(
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
            members.append(member)

        for method in class_.declaration.methods:
            if method.name in seen_names:
                self.context.diagnostics.add(
                    (
                        f"Class member '{method.name}' already defined in "
                        f"'{class_.name}'"
                    ),
                    node=method,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            seen_names.add(method.name)
            for argument in method.prototype.args.nodes:
                self._resolve_declared_type(argument.type_, node=argument)
            self._resolve_declared_type(
                method.prototype.return_type,
                node=method,
            )
            signature = self._normalize_class_method_signature(
                class_,
                method,
            )
            is_static = self._method_is_static(method)
            overrides: str | None = None
            inherited_members = ancestors.get(method.name, [])
            override_is_valid = True
            for inherited in inherited_members:
                if inherited.kind is not ClassMemberKind.METHOD:
                    self.context.diagnostics.add(
                        (
                            f"Class method '{class_.name}.{method.name}' "
                            "cannot override inherited attribute "
                            f"'{inherited.name}'"
                        ),
                        node=method,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                    override_is_valid = False
                    continue
                if inherited.is_static != is_static:
                    self.context.diagnostics.add(
                        (
                            f"Class method '{class_.name}.{method.name}' "
                            "changes static/instance status across "
                            "inheritance"
                        ),
                        node=method,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                    override_is_valid = False
                    continue
                if (
                    inherited.signature is None
                    or not self.registry.signatures_match(
                        inherited.signature,
                        signature,
                    )
                ):
                    self.context.diagnostics.add(
                        (
                            f"Class method '{class_.name}.{method.name}' "
                            "must match inherited signature exactly"
                        ),
                        node=method,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                    override_is_valid = False
                    continue
                if self._visibility_rank(
                    method.prototype.visibility
                ) < self._visibility_rank(
                    inherited.visibility,
                ):
                    self.context.diagnostics.add(
                        (
                            f"Class method '{class_.name}.{method.name}' "
                            "cannot reduce visibility when overriding "
                            f"'{inherited.name}'"
                        ),
                        node=method,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                    override_is_valid = False
            if override_is_valid and inherited_members:
                overrides = inherited_members[0].qualified_name
            member = self.factory.make_class_member(
                class_,
                name=method.name,
                kind=ClassMemberKind.METHOD,
                declaration=method,
                visibility=method.prototype.visibility,
                is_static=is_static,
                is_constant=True,
                is_mutable=False,
                signature=signature,
                overrides=overrides,
            )
            self._set_class(method.prototype, class_)
            self._set_class(method, class_)
            self._set_type(method.prototype, None)
            self._set_type(method, None)
            members.append(member)

        return tuple(members)

    def _resolve_effective_class_members(
        self,
        class_: SemanticClass,
        mro: tuple[SemanticClass, ...],
        declared_member_table: dict[str, SemanticClassMember],
    ) -> tuple[
        dict[str, SemanticClassMember],
        dict[str, SemanticClassMemberResolution],
    ]:
        """
        title: Resolve the effective member table for one class.
        parameters:
          class_:
            type: SemanticClass
          mro:
            type: tuple[SemanticClass, Ellipsis]
          declared_member_table:
            type: dict[str, SemanticClassMember]
        returns:
          type: >-
            tuple[dict[str, SemanticClassMember], dict[str,
            SemanticClassMemberResolution]]
        """
        effective_members = dict(declared_member_table)
        ancestor_members = self._ancestor_declared_members(mro)
        member_resolution: dict[str, SemanticClassMemberResolution] = {}

        for name, member in declared_member_table.items():
            inherited_candidates = tuple(ancestor_members.get(name, ()))
            resolution_kind = (
                ClassMemberResolutionKind.OVERRIDE
                if inherited_candidates
                else ClassMemberResolutionKind.DECLARED
            )
            member_resolution[name] = SemanticClassMemberResolution(
                name=name,
                kind=resolution_kind,
                selected=member,
                candidates=(member, *inherited_candidates),
            )

        for name, candidates in ancestor_members.items():
            if name in effective_members:
                continue
            primary = candidates[0]
            if len(candidates) == 1:
                effective_members[name] = primary
                member_resolution[name] = SemanticClassMemberResolution(
                    name=name,
                    kind=ClassMemberResolutionKind.INHERITED,
                    selected=primary,
                    candidates=tuple(candidates),
                )
                continue
            owner_names = ", ".join(member.owner_name for member in candidates)
            if any(
                member.kind is not primary.kind
                or member.is_static != primary.is_static
                for member in candidates[1:]
            ):
                self.context.diagnostics.add(
                    (
                        f"Class '{class_.name}' inherits conflicting "
                        f"members named '{name}' from {owner_names}"
                    ),
                    node=class_.declaration,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            if primary.kind is ClassMemberKind.ATTRIBUTE:
                self.context.diagnostics.add(
                    (
                        f"Class '{class_.name}' inherits ambiguous "
                        f"attribute '{name}' from {owner_names}"
                    ),
                    node=class_.declaration,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            if primary.signature is None or any(
                candidate.signature is None
                or not self.registry.signatures_match(
                    primary.signature,
                    candidate.signature,
                )
                for candidate in candidates[1:]
            ):
                self.context.diagnostics.add(
                    (
                        f"Class '{class_.name}' inherits conflicting "
                        f"methods named '{name}' from {owner_names}"
                    ),
                    node=class_.declaration,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            effective_members[name] = primary
            member_resolution[name] = SemanticClassMemberResolution(
                name=name,
                kind=ClassMemberResolutionKind.INHERITED,
                selected=primary,
                candidates=tuple(candidates),
            )

        return effective_members, member_resolution

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
            declared_member_table = {
                member.name: member for member in declared_members
            }
            member_table, member_resolution = (
                self._resolve_effective_class_members(
                    current,
                    mro,
                    declared_member_table,
                )
            )
            effective_members = tuple(member_table.values())
            shared_ancestors = self._shared_class_ancestors(bases, mro)
            layout = self._build_class_layout(
                current,
                bases,
                declared_members,
                effective_members,
            )
            updated = replace(
                current,
                bases=bases,
                declared_members=declared_members,
                declared_member_table=declared_member_table,
                member_table=member_table,
                member_resolution=member_resolution,
                instance_attributes=tuple(
                    member
                    for member in effective_members
                    if member.kind is ClassMemberKind.ATTRIBUTE
                    and not member.is_static
                ),
                static_attributes=tuple(
                    member
                    for member in effective_members
                    if member.kind is ClassMemberKind.ATTRIBUTE
                    and member.is_static
                ),
                instance_methods=tuple(
                    member
                    for member in effective_members
                    if member.kind is ClassMemberKind.METHOD
                    and not member.is_static
                ),
                static_methods=tuple(
                    member
                    for member in effective_members
                    if member.kind is ClassMemberKind.METHOD
                    and member.is_static
                ),
                inheritance_graph=tuple(
                    ancestor.qualified_name for ancestor in mro[1:]
                ),
                shared_ancestors=shared_ancestors,
                layout=layout,
                mro=(
                    current,
                    *tuple(
                        self.context.get_class(
                            ancestor.module_key, ancestor.name
                        )
                        or ancestor
                        for ancestor in mro[1:]
                    ),
                ),
                is_resolved=True,
            )
            updated = replace(updated, mro=(updated, *updated.mro[1:]))
            self.context.register_class(updated)
            self.bindings.bind_class(
                updated.name, updated, node=updated.declaration
            )
            self._set_class(updated.declaration, updated)
            return updated
        finally:
            active_stack.pop()

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, module: astx.Module) -> None:
        """
        title: Visit Module nodes.
        parameters:
          module:
            type: astx.Module
        """
        with self.context.in_module(module.name):
            self._visit_module(module, predeclared=False)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, block: astx.Block) -> None:
        """
        title: Visit Block nodes.
        parameters:
          block:
            type: astx.Block
        """
        self._set_type(block, None)
        self._predeclare_block_structs(block)
        for node in block.nodes:
            self.visit(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionPrototype) -> None:
        """
        title: Visit FunctionPrototype nodes.
        parameters:
          node:
            type: astx.FunctionPrototype
        """
        for arg in node.args.nodes:
            self._resolve_declared_type(arg.type_, node=arg)
        self._resolve_declared_type(node.return_type, node=node)
        function = self.registry.resolve_function(node.name)
        if function is None:
            function = self.registry.register_function(node)
        function = self._synchronize_function_signature(function, node)
        self.bindings.bind_function(node.name, function, node=node)
        self._set_function(node, function)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionDef) -> None:
        """
        title: Visit FunctionDef nodes.
        parameters:
          node:
            type: astx.FunctionDef
        """
        for arg in node.prototype.args.nodes:
            self._resolve_declared_type(arg.type_, node=arg)
        self._resolve_declared_type(node.prototype.return_type, node=node)
        function = self.registry.resolve_function(node.name)
        if function is None:
            function = self.registry.register_function(
                node.prototype,
                definition=node,
            )
        function = self._synchronize_function_signature(
            function,
            node.prototype,
            definition=node,
        )
        self.bindings.bind_function(node.name, function, node=node)
        self._set_function(node.prototype, function)
        self._set_function(node, function)
        with self.context.in_function(function):
            with self.context.scope("function"):
                for arg_node, arg_symbol in zip(
                    node.prototype.args.nodes,
                    function.args,
                ):
                    self.context.scopes.declare(arg_symbol)
                    self._set_symbol(arg_node, arg_symbol)
                    self._set_type(arg_node, arg_symbol.type_)
                self.visit(node.body)
        if not isinstance(
            function.return_type, astx.NoneType
        ) and not self._guarantees_return(node.body):
            self.context.diagnostics.add(
                f"Function '{node.name}' with return type "
                f"'{function.return_type}' is missing a return statement",
                node=node,
            )

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.VariableDeclaration) -> None:
        """
        title: Visit VariableDeclaration nodes.
        parameters:
          node:
            type: astx.VariableDeclaration
        """
        self._resolve_declared_type(node.type_, node=node)
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self.visit(node.value)
            if self._require_value_expression(
                node.value,
                context=f"Initializer for '{node.name}'",
            ):
                validate_assignment(
                    self.context.diagnostics,
                    target_name=node.name,
                    target_type=node.type_,
                    value_type=self._expr_type(node.value),
                    node=node,
                )
        symbol = self.registry.declare_local(
            node.name,
            node.type_,
            is_mutable=node.mutability != astx.MutabilityKind.constant,
            declaration=node,
        )
        self._set_symbol(node, symbol)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.InlineVariableDeclaration) -> None:
        """
        title: Visit InlineVariableDeclaration nodes.
        parameters:
          node:
            type: astx.InlineVariableDeclaration
        """
        self._resolve_declared_type(node.type_, node=node)
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self.visit(node.value)
            if self._require_value_expression(
                node.value,
                context=f"Initializer for '{node.name}'",
            ):
                validate_assignment(
                    self.context.diagnostics,
                    target_name=node.name,
                    target_type=node.type_,
                    value_type=self._expr_type(node.value),
                    node=node,
                )
        symbol = self.registry.declare_local(
            node.name,
            node.type_,
            is_mutable=node.mutability != astx.MutabilityKind.constant,
            declaration=node,
        )
        self._set_symbol(node, symbol)

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
        self._set_class(node, class_)
        self._set_type(node, None)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.StructDefStmt) -> None:
        """
        title: Visit StructDefStmt nodes.
        parameters:
          node:
            type: astx.StructDefStmt
        """
        struct = self.registry.register_struct(node)
        self.bindings.bind_struct(node.name, struct, node=node)
        struct = self._resolve_struct_fields(struct)
        self._set_struct(node, struct)
        self._validate_struct_cycles(struct)
        self._set_type(node, None)

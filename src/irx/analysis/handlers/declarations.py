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
    class_method_symbol_basename,
    mangle_class_descriptor_name,
    mangle_class_dispatch_name,
    mangle_class_name,
    mangle_class_static_name,
    qualified_class_method_name,
    qualified_local_name,
)
from irx.analysis.resolved_nodes import (
    ClassHeaderFieldKind,
    ClassInitializationSourceKind,
    ClassMemberKind,
    ClassMemberResolutionKind,
    ClassObjectRepresentationKind,
    FunctionSignature,
    ParameterSpec,
    SemanticClass,
    SemanticClassFieldInitializer,
    SemanticClassHeaderField,
    SemanticClassInitialization,
    SemanticClassLayout,
    SemanticClassLayoutField,
    SemanticClassMember,
    SemanticClassMemberResolution,
    SemanticClassMethodDispatch,
    SemanticClassStaticInitializer,
    SemanticClassStaticStorage,
    SemanticFunction,
    SemanticStruct,
    SemanticStructField,
    SemanticSymbol,
)
from irx.analysis.types import clone_type, display_type_name
from irx.analysis.validation import validate_assignment
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked

DIRECT_STRUCT_CYCLE_LENGTH = 2
MIN_SHARED_ANCESTOR_BASE_COUNT = 2
IMPLICIT_METHOD_RECEIVER_NAME = "self"
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

    def _initializer_source_kind(
        self,
        value: astx.AST | None,
    ) -> ClassInitializationSourceKind:
        """
        title: Return how one class attribute obtains its initial value.
        parameters:
          value:
            type: astx.AST | None
        returns:
          type: ClassInitializationSourceKind
        """
        if value is None or isinstance(value, astx.Undefined):
            return ClassInitializationSourceKind.DEFAULT
        return ClassInitializationSourceKind.DECLARATION

    def _static_initializer_is_supported(
        self,
        value: astx.AST | None,
    ) -> bool:
        """
        title: >-
          Return whether one static field initializer is deterministic now.
        parameters:
          value:
            type: astx.AST | None
        returns:
          type: bool
        """
        if value is None or isinstance(value, astx.Undefined):
            return True
        return isinstance(
            value,
            (
                astx.LiteralBoolean,
                astx.LiteralInt8,
                astx.LiteralInt16,
                astx.LiteralInt32,
                astx.LiteralInt64,
                astx.LiteralUInt8,
                astx.LiteralUInt16,
                astx.LiteralUInt32,
                astx.LiteralUInt64,
                astx.LiteralUInt128,
                astx.LiteralFloat16,
                astx.LiteralFloat32,
                astx.LiteralFloat64,
                astx.LiteralNone,
            ),
        )

    def _method_signature_key(
        self,
        method_name: str,
        signature: FunctionSignature,
        *,
        is_static: bool,
    ) -> str:
        """
        title: Return one canonical visible method signature key.
        parameters:
          method_name:
            type: str
          signature:
            type: FunctionSignature
          is_static:
            type: bool
        returns:
          type: str
        """
        dispatch_kind = "static" if is_static else "instance"
        parameters = ",".join(
            display_type_name(parameter.type_)
            for parameter in signature.parameters
        )
        variadic = "variadic" if signature.is_variadic else "fixed"
        return (
            f"{dispatch_kind}|{method_name}|"
            f"{signature.calling_convention.value}|{variadic}|"
            f"({parameters})->{display_type_name(signature.return_type)}"
        )

    def _method_call_key_from_signature(
        self,
        method_name: str,
        signature: FunctionSignature,
        *,
        is_static: bool,
    ) -> str:
        """
        title: Return one overload-selection key without the return type.
        parameters:
          method_name:
            type: str
          signature:
            type: FunctionSignature
          is_static:
            type: bool
        returns:
          type: str
        """
        dispatch_kind = "static" if is_static else "instance"
        parameters = ",".join(
            display_type_name(parameter.type_)
            for parameter in signature.parameters
        )
        variadic = "variadic" if signature.is_variadic else "fixed"
        return (
            f"{dispatch_kind}|{method_name}|"
            f"{signature.calling_convention.value}|{variadic}|"
            f"({parameters})"
        )

    def _member_call_key(self, member: SemanticClassMember) -> str:
        """
        title: Return one overload-selection key for a class method member.
        parameters:
          member:
            type: SemanticClassMember
        returns:
          type: str
        """
        if member.signature is None:
            raise TypeError("class method must carry a visible signature")
        return self._method_call_key_from_signature(
            member.name,
            member.signature,
            is_static=member.is_static,
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

    def _declared_member_tables(
        self,
        declared_members: tuple[SemanticClassMember, ...],
    ) -> tuple[
        dict[str, SemanticClassMember],
        dict[str, tuple[SemanticClassMember, ...]],
    ]:
        """
        title: Build the declared member tables for one class.
        parameters:
          declared_members:
            type: tuple[SemanticClassMember, Ellipsis]
        returns:
          type: >-
            tuple[dict[str, SemanticClassMember], dict[str,
            tuple[SemanticClassMember, Ellipsis]]]
        """
        declared_member_table: dict[str, SemanticClassMember] = {}
        declared_method_groups: dict[str, list[SemanticClassMember]] = {}

        for member in declared_members:
            if member.kind is ClassMemberKind.ATTRIBUTE:
                declared_member_table[member.name] = member
                continue
            declared_method_groups.setdefault(member.name, []).append(member)

        normalized_method_groups = {
            name: tuple(group)
            for name, group in declared_method_groups.items()
        }
        for name, group in normalized_method_groups.items():
            if len(group) == 1:
                declared_member_table[name] = group[0]

        return declared_member_table, normalized_method_groups

    def _ancestor_declared_attributes(
        self,
        mro: tuple[SemanticClass, ...],
    ) -> dict[str, tuple[SemanticClassMember, ...]]:
        """
        title: Collect visible ancestor attributes in MRO order.
        parameters:
          mro:
            type: tuple[SemanticClass, Ellipsis]
        returns:
          type: dict[str, tuple[SemanticClassMember, Ellipsis]]
        """
        members: dict[str, list[SemanticClassMember]] = {}
        for ancestor in mro[1:]:
            for member in ancestor.declared_members:
                if member.visibility is astx.VisibilityKind.private:
                    continue
                if member.kind is not ClassMemberKind.ATTRIBUTE:
                    continue
                members.setdefault(member.name, []).append(member)
        return {name: tuple(group) for name, group in members.items()}

    def _ancestor_declared_method_groups(
        self,
        mro: tuple[SemanticClass, ...],
    ) -> dict[str, tuple[SemanticClassMember, ...]]:
        """
        title: Collect visible ancestor method overload groups in MRO order.
        parameters:
          mro:
            type: tuple[SemanticClass, Ellipsis]
        returns:
          type: dict[str, tuple[SemanticClassMember, Ellipsis]]
        """
        groups: dict[str, list[SemanticClassMember]] = {}
        for ancestor in mro[1:]:
            for member in ancestor.declared_members:
                if member.visibility is astx.VisibilityKind.private:
                    continue
                if member.kind is not ClassMemberKind.METHOD:
                    continue
                groups.setdefault(member.name, []).append(member)
        return {name: tuple(group) for name, group in groups.items()}

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
            resolved.append(self._resolve_class_structure(base))

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
        return member.owner_qualified_name.partition("::class::")[0]

    def _member_dominates(
        self,
        dominant: SemanticClass,
        other: SemanticClass,
    ) -> bool:
        """
        title: Return whether one owner dominates another in inheritance.
        parameters:
          dominant:
            type: SemanticClass
          other:
            type: SemanticClass
        returns:
          type: bool
        """
        if dominant.qualified_name == other.qualified_name:
            return True
        return any(
            ancestor.qualified_name == other.qualified_name
            for ancestor in dominant.mro[1:]
        )

    def _dominant_inherited_method(
        self,
        candidates: tuple[SemanticClassMember, ...],
    ) -> SemanticClassMember | None:
        """
        title: Return the unique dominant inherited method, if any.
        parameters:
          candidates:
            type: tuple[SemanticClassMember, Ellipsis]
        returns:
          type: SemanticClassMember | None
        """
        if not candidates:
            return None
        selected = candidates[0]
        selected_owner = self.context.get_class(
            self._member_owner_module_key(selected),
            selected.owner_name,
        )
        if selected_owner is None:
            return None
        for candidate in candidates[1:]:
            owner = self.context.get_class(
                self._member_owner_module_key(candidate),
                candidate.owner_name,
            )
            if owner is None:
                return None
            if not self._member_dominates(selected_owner, owner):
                return None
        return selected

    def _build_class_layout(
        self,
        class_: SemanticClass,
        bases: tuple[SemanticClass, ...],
        declared_members: tuple[SemanticClassMember, ...],
        member_table: dict[str, SemanticClassMember],
        method_groups: dict[str, tuple[SemanticClassMember, ...]],
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
          member_table:
            type: dict[str, SemanticClassMember]
          method_groups:
            type: dict[str, tuple[SemanticClassMember, Ellipsis]]
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
        for name, member in member_table.items():
            if member.kind is not ClassMemberKind.ATTRIBUTE:
                continue
            if member.is_static:
                storage = inherited_static_storage.get(member.qualified_name)
                if storage is not None:
                    visible_static_storage[name] = storage
                continue
            visible_slot = field_slots.get(member.qualified_name)
            if visible_slot is not None:
                visible_field_slots[name] = visible_slot

        dispatch_slots: dict[int, SemanticClassMethodDispatch] = {}
        visible_method_slots: dict[str, SemanticClassMethodDispatch] = {}
        for group in method_groups.values():
            for member in group:
                if (
                    member.is_static
                    or member.dispatch_slot is None
                    or member.lowered_function is None
                    or member.signature_key is None
                ):
                    continue
                dispatch_entry = SemanticClassMethodDispatch(
                    member=member,
                    function=member.lowered_function,
                    slot_index=member.dispatch_slot,
                    owner_name=member.owner_name,
                    owner_qualified_name=member.owner_qualified_name,
                )
                dispatch_slots[dispatch_entry.slot_index] = dispatch_entry
                visible_method_slots[member.signature_key] = dispatch_entry

        dispatch_entries = tuple(
            dispatch_slots[index] for index in sorted(dispatch_slots)
        )
        dispatch_table_size = len(dispatch_entries)

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
            dispatch_entries=dispatch_entries,
            dispatch_slots=dispatch_slots,
            visible_method_slots=visible_method_slots,
            dispatch_table_size=dispatch_table_size,
            static_fields=static_fields,
            static_storage=static_storage,
            visible_static_storage=visible_static_storage,
        )

    def _build_class_initialization(
        self,
        class_: SemanticClass,
        layout: SemanticClassLayout,
    ) -> SemanticClassInitialization:
        """
        title: Build one canonical class initialization plan.
        parameters:
          class_:
            type: SemanticClass
          layout:
            type: SemanticClassLayout
        returns:
          type: SemanticClassInitialization
        """
        instance_initializers = tuple(
            SemanticClassFieldInitializer(
                field=field,
                source_kind=self._initializer_source_kind(
                    getattr(field.member.declaration, "value", None)
                ),
                value=(
                    None
                    if isinstance(
                        getattr(field.member.declaration, "value", None),
                        astx.Undefined,
                    )
                    else getattr(field.member.declaration, "value", None)
                ),
                owner_name=field.owner_name,
                owner_qualified_name=field.owner_qualified_name,
            )
            for field in layout.instance_fields
        )
        static_initializers = tuple(
            SemanticClassStaticInitializer(
                storage=storage,
                source_kind=self._initializer_source_kind(
                    getattr(storage.member.declaration, "value", None)
                ),
                value=(
                    None
                    if isinstance(
                        getattr(storage.member.declaration, "value", None),
                        astx.Undefined,
                    )
                    else getattr(storage.member.declaration, "value", None)
                ),
                owner_name=storage.owner_name,
                owner_qualified_name=storage.owner_qualified_name,
            )
            for storage in layout.static_fields
        )
        return SemanticClassInitialization(
            instance_initializers=instance_initializers,
            static_initializers=static_initializers,
        )

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

    def _make_method_receiver_type(
        self,
        class_: SemanticClass,
    ) -> astx.ClassType:
        """
        title: Return the canonical implicit receiver type for one class.
        parameters:
          class_:
            type: SemanticClass
        returns:
          type: astx.ClassType
        """
        return astx.ClassType(
            class_.name,
            resolved_name=class_.name,
            module_key=class_.module_key,
            qualified_name=class_.qualified_name,
            ancestor_qualified_names=tuple(
                ancestor.qualified_name for ancestor in class_.mro[1:]
            ),
        )

    def _make_visible_method_function(
        self,
        class_: SemanticClass,
        member: SemanticClassMember,
    ) -> SemanticFunction:
        """
        title: Build one visible-signature callable wrapper for a class method.
        parameters:
          class_:
            type: SemanticClass
          member:
            type: SemanticClassMember
        returns:
          type: SemanticFunction
        """
        declaration = member.declaration
        if (
            member.signature is None
            or member.lowered_function is None
            or not isinstance(declaration, astx.FunctionDef)
        ):
            raise TypeError("class method must have a lowered function")
        return SemanticFunction(
            symbol_id=member.lowered_function.symbol_id,
            name=f"{class_.name}.{member.name}",
            return_type=clone_type(member.signature.return_type),
            args=(),
            signature=member.signature,
            prototype=declaration.prototype,
            definition=declaration,
            module_key=class_.module_key,
            qualified_name=qualified_class_method_name(
                class_.module_key,
                class_.name,
                member.name,
                member.signature_key,
            ),
        )

    def _make_lowered_method_function(
        self,
        class_: SemanticClass,
        declaration: astx.FunctionDef,
        signature: FunctionSignature,
        *,
        is_static: bool,
        signature_key: str,
    ) -> SemanticFunction:
        """
        title: Build one lowered semantic function for a class method.
        parameters:
          class_:
            type: SemanticClass
          declaration:
            type: astx.FunctionDef
          signature:
            type: FunctionSignature
          is_static:
            type: bool
          signature_key:
            type: str
        returns:
          type: SemanticFunction
        """
        lowered_signature = signature
        user_args = tuple(
            self.factory.make_parameter_symbol(class_.module_key, argument)
            for argument in declaration.prototype.args.nodes
        )
        receiver_args: tuple[SemanticSymbol, ...] = ()
        if not is_static:
            receiver_type = self._make_method_receiver_type(class_)
            receiver_symbol = self.factory.make_variable_symbol(
                class_.module_key,
                IMPLICIT_METHOD_RECEIVER_NAME,
                receiver_type,
                is_mutable=False,
                declaration=declaration,
                kind="method_receiver",
            )
            receiver_spec = ParameterSpec(
                name=IMPLICIT_METHOD_RECEIVER_NAME,
                type_=clone_type(receiver_type),
            )
            lowered_signature = replace(
                signature,
                parameters=(receiver_spec, *signature.parameters),
                symbol_name=class_method_symbol_basename(
                    class_.name,
                    declaration.name,
                    signature_key,
                ),
                metadata={
                    **signature.metadata,
                    "class_name": class_.name,
                    "method_name": declaration.name,
                    "has_hidden_receiver": True,
                },
            )
            receiver_args = (receiver_symbol,)
        else:
            lowered_signature = replace(
                signature,
                symbol_name=class_method_symbol_basename(
                    class_.name,
                    declaration.name,
                    signature_key,
                ),
                metadata={
                    **signature.metadata,
                    "class_name": class_.name,
                    "method_name": declaration.name,
                    "has_hidden_receiver": False,
                },
            )
        semantic_args = (*receiver_args, *user_args)
        lowered_function = self.factory.make_function(
            class_.module_key,
            declaration.prototype,
            signature=lowered_signature,
            definition=declaration,
            args=semantic_args,
        )
        return replace(
            lowered_function,
            qualified_name=qualified_class_method_name(
                class_.module_key,
                class_.name,
                declaration.name,
                signature_key,
            ),
        )

    def _analyze_class_method_body(
        self,
        class_: SemanticClass,
        member: SemanticClassMember,
    ) -> None:
        """
        title: Analyze one lowered class method body.
        parameters:
          class_:
            type: SemanticClass
          member:
            type: SemanticClassMember
        """
        declaration = member.declaration
        function = member.lowered_function
        if not isinstance(declaration, astx.FunctionDef) or function is None:
            return

        hidden_parameter_count = len(function.args) - len(
            declaration.prototype.args.nodes
        )
        with self.context.in_function(function):
            with self.context.scope("method"):
                for idx, arg_symbol in enumerate(function.args):
                    self.context.scopes.declare(arg_symbol)
                    if idx < hidden_parameter_count:
                        continue
                    arg_node = declaration.prototype.args.nodes[
                        idx - hidden_parameter_count
                    ]
                    self._set_symbol(arg_node, arg_symbol)
                    self._set_type(arg_node, arg_symbol.type_)
                self.visit(declaration.body)
        if not isinstance(function.return_type, astx.NoneType) and not (
            self._guarantees_return(declaration.body)
        ):
            self.context.diagnostics.add(
                f"Function '{class_.name}.{member.name}' with return type "
                f"'{function.return_type}' is missing a return statement",
                node=declaration,
            )

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

    def _family_classes(
        self,
    ) -> tuple[SemanticClass, ...]:
        """
        title: Return all structurally resolved classes in the context.
        returns:
          type: tuple[SemanticClass, Ellipsis]
        """
        return tuple(
            class_
            for class_ in self.context.classes.values()
            if class_.is_structurally_resolved
        )

    def _class_family_components(
        self,
        classes: tuple[SemanticClass, ...],
    ) -> tuple[tuple[SemanticClass, ...], ...]:
        """
        title: Partition classes into weakly connected inheritance families.
        parameters:
          classes:
            type: tuple[SemanticClass, Ellipsis]
        returns:
          type: tuple[tuple[SemanticClass, Ellipsis], Ellipsis]
        """
        classes_by_name = {class_.qualified_name: class_ for class_ in classes}
        adjacency: dict[str, set[str]] = {
            class_.qualified_name: set() for class_ in classes
        }
        for class_ in classes:
            for base in class_.bases:
                if base.qualified_name not in classes_by_name:
                    continue
                adjacency[class_.qualified_name].add(base.qualified_name)
                adjacency[base.qualified_name].add(class_.qualified_name)

        seen: set[str] = set()
        components: list[tuple[SemanticClass, ...]] = []
        for class_ in sorted(classes, key=lambda item: item.qualified_name):
            if class_.qualified_name in seen:
                continue
            queue = [class_.qualified_name]
            component_names: list[str] = []
            seen.add(class_.qualified_name)
            while queue:
                current = queue.pop()
                component_names.append(current)
                for neighbor in sorted(adjacency[current]):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    queue.append(neighbor)
            components.append(
                tuple(
                    classes_by_name[name] for name in sorted(component_names)
                )
            )
        return tuple(components)

    def _topologically_sorted_class_family(
        self,
        family: tuple[SemanticClass, ...],
    ) -> tuple[SemanticClass, ...]:
        """
        title: >-
          Return one deterministic ancestor-before-descendant family order.
        parameters:
          family:
            type: tuple[SemanticClass, Ellipsis]
        returns:
          type: tuple[SemanticClass, Ellipsis]
        """
        family_names = {class_.qualified_name for class_ in family}
        indegree = {class_.qualified_name: 0 for class_ in family}
        children: dict[str, list[SemanticClass]] = {
            class_.qualified_name: [] for class_ in family
        }

        for class_ in family:
            for base in class_.bases:
                if base.qualified_name not in family_names:
                    continue
                indegree[class_.qualified_name] += 1
                children[base.qualified_name].append(class_)

        available = sorted(
            (
                class_
                for class_ in family
                if indegree[class_.qualified_name] == 0
            ),
            key=lambda item: item.qualified_name,
        )
        ordered: list[SemanticClass] = []
        while available:
            current = available.pop(0)
            ordered.append(current)
            for child in sorted(
                children[current.qualified_name],
                key=lambda item: item.qualified_name,
            ):
                indegree[child.qualified_name] -= 1
                if indegree[child.qualified_name] == 0:
                    available.append(child)
                    available.sort(key=lambda item: item.qualified_name)

        if len(ordered) != len(family):
            return tuple(sorted(family, key=lambda item: item.qualified_name))
        return tuple(ordered)

    def _assign_family_dispatch_slots(
        self,
        family: tuple[SemanticClass, ...],
    ) -> dict[str, int]:
        """
        title: Assign hierarchy-local dispatch slots for one class family.
        parameters:
          family:
            type: tuple[SemanticClass, Ellipsis]
        returns:
          type: dict[str, int]
        """
        slot_by_signature: dict[str, int] = {}
        member_slots: dict[str, int] = {}
        next_slot = 0

        for class_ in self._topologically_sorted_class_family(family):
            for member in class_.declared_members:
                if (
                    member.kind is not ClassMemberKind.METHOD
                    or member.is_static
                    or member.visibility is astx.VisibilityKind.private
                    or member.signature_key is None
                ):
                    continue
                slot_index = slot_by_signature.get(member.signature_key)
                if slot_index is None:
                    slot_index = next_slot
                    slot_by_signature[member.signature_key] = slot_index
                    next_slot += 1
                member_slots[member.qualified_name] = slot_index

        return member_slots

    def _finalize_class_family(
        self,
        family: tuple[SemanticClass, ...],
    ) -> None:
        """
        title: Compute effective members and layouts for one class family.
        parameters:
          family:
            type: tuple[SemanticClass, Ellipsis]
        """
        member_slots = self._assign_family_dispatch_slots(family)
        ordered_family = self._topologically_sorted_class_family(family)

        for structural_class in ordered_family:
            bases = tuple(
                self.context.get_class(base.module_key, base.name) or base
                for base in structural_class.bases
            )
            mro_ancestors = tuple(
                self.context.get_class(ancestor.module_key, ancestor.name)
                or ancestor
                for ancestor in structural_class.mro[1:]
            )
            declared_members = tuple(
                replace(
                    member,
                    dispatch_slot=member_slots.get(member.qualified_name),
                )
                if (
                    member.kind is ClassMemberKind.METHOD
                    and not member.is_static
                    and member.visibility is not astx.VisibilityKind.private
                )
                else replace(member, dispatch_slot=None)
                if member.kind is ClassMemberKind.METHOD
                else member
                for member in structural_class.declared_members
            )
            declared_member_table, declared_method_groups = (
                self._declared_member_tables(declared_members)
            )
            class_stub = replace(
                structural_class,
                bases=bases,
                declared_members=declared_members,
                declared_member_table=declared_member_table,
                declared_method_groups=declared_method_groups,
                mro=(structural_class, *mro_ancestors),
            )
            (
                member_table,
                method_groups,
                member_resolution,
                method_resolution,
            ) = self._resolve_effective_class_members(
                class_stub,
                class_stub.mro,
                declared_member_table,
                declared_method_groups,
            )
            layout = self._build_class_layout(
                class_stub,
                bases,
                declared_members,
                member_table,
                method_groups,
            )
            visible_methods = tuple(
                member for group in method_groups.values() for member in group
            )
            initialization = self._build_class_initialization(
                class_stub,
                layout,
            )
            updated = replace(
                class_stub,
                member_table=member_table,
                method_groups=method_groups,
                member_resolution=member_resolution,
                method_resolution=method_resolution,
                instance_attributes=tuple(
                    member
                    for member in member_table.values()
                    if member.kind is ClassMemberKind.ATTRIBUTE
                    and not member.is_static
                ),
                static_attributes=tuple(
                    member
                    for member in member_table.values()
                    if member.kind is ClassMemberKind.ATTRIBUTE
                    and member.is_static
                ),
                instance_methods=tuple(
                    member
                    for member in visible_methods
                    if not member.is_static
                ),
                static_methods=tuple(
                    member for member in visible_methods if member.is_static
                ),
                inheritance_graph=tuple(
                    ancestor.qualified_name for ancestor in mro_ancestors
                ),
                shared_ancestors=structural_class.shared_ancestors,
                layout=layout,
                initialization=initialization,
                mro=(structural_class, *mro_ancestors),
                is_resolved=True,
            )
            updated = replace(updated, mro=(updated, *mro_ancestors))
            self.context.register_class(updated)
            self.bindings.bind_class(
                updated.name,
                updated,
                node=updated.declaration,
            )
            self._set_class(updated.declaration, updated)

    def _finalize_registered_class_layouts(self) -> None:
        """
        title: Finalize all currently registered class families.
        """
        if self._class_resolution_stack():
            return
        for class_ in tuple(self.context.classes.values()):
            if class_.is_structurally_resolved:
                continue
            self._resolve_class_structure(class_)
        families = self._class_family_components(self._family_classes())
        for family in families:
            self._finalize_class_family(family)

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
        with self.context.in_class(class_):
            for member in class_.declared_members:
                if member.kind is not ClassMemberKind.METHOD:
                    continue
                self._analyze_class_method_body(class_, member)
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

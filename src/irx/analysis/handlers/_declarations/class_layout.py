# mypy: disable-error-code=no-redef
# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=untyped-decorator

"""
title: Declaration class-layout helpers.
summary: >-
  Build class layouts, initialization metadata, and family-level dispatch
  finalization structures during declaration analysis.
"""

from __future__ import annotations

from dataclasses import replace

from irx import astx
from irx.analysis.handlers.base import SemanticVisitorMixinBase
from irx.analysis.module_symbols import (
    mangle_class_descriptor_name,
    mangle_class_dispatch_name,
    mangle_class_name,
    mangle_class_static_name,
)
from irx.analysis.resolved_nodes import (
    ClassHeaderFieldKind,
    ClassMemberKind,
    ClassObjectRepresentationKind,
    SemanticClass,
    SemanticClassFieldInitializer,
    SemanticClassHeaderField,
    SemanticClassInitialization,
    SemanticClassLayout,
    SemanticClassLayoutField,
    SemanticClassMember,
    SemanticClassMethodDispatch,
    SemanticClassStaticInitializer,
    SemanticClassStaticStorage,
)
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked

_CLASS_HEADER_LAYOUT: tuple[tuple[str, ClassHeaderFieldKind], ...] = (
    ("descriptor", ClassHeaderFieldKind.TYPE_DESCRIPTOR),
    ("dispatch", ClassHeaderFieldKind.DISPATCH_TABLE),
)


@typechecked
class DeclarationClassLayoutVisitorMixin(SemanticVisitorMixinBase):
    """
    title: Declaration helpers for class layouts and family finalization
    """

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
        dispatch_slot_indices: set[int] = set()
        for group in method_groups.values():
            for member in group:
                if member.dispatch_slot is not None:
                    dispatch_slot_indices.add(member.dispatch_slot)
                if (
                    member.is_static
                    or member.is_abstract
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
        dispatch_table_size = (
            max(dispatch_slot_indices) + 1 if dispatch_slot_indices else 0
        )

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
                lowered_function = member.lowered_function
                if (
                    lowered_function is not None
                    and lowered_function.template_params
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
            abstract_methods = tuple(
                member for member in visible_methods if member.is_abstract
            )
            if abstract_methods and not self._class_is_declared_abstract(
                structural_class
            ):
                method_names = ", ".join(
                    f"{member.owner_name}.{member.name}"
                    for member in abstract_methods
                )
                self.context.diagnostics.add(
                    (
                        f"Class '{structural_class.name}' must be abstract "
                        "or implement abstract method"
                        f"{'s' if len(abstract_methods) != 1 else ''} "
                        f"{method_names}"
                    ),
                    node=structural_class.declaration,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
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
                abstract_methods=abstract_methods,
                inheritance_graph=tuple(
                    ancestor.qualified_name for ancestor in mro_ancestors
                ),
                shared_ancestors=structural_class.shared_ancestors,
                layout=layout,
                initialization=initialization,
                mro=(structural_class, *mro_ancestors),
                is_resolved=True,
                is_abstract=(
                    self._class_is_declared_abstract(structural_class)
                    or bool(abstract_methods)
                ),
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

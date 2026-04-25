"""
title: Sidecar semantic dataclasses attached to AST nodes.
summary: >-
  Define the semantic sidecar objects that analysis attaches to AST nodes and
  reuses during lowering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from public import public

from irx import astx
from irx.analysis.module_interfaces import ModuleKey
from irx.typecheck import typechecked


@public
@typechecked
@dataclass(frozen=True)
class SemanticSymbol:
    """
    title: Resolved symbol information.
    summary: >-
      Describe one resolved variable-like symbol, including its stable semantic
      id and declared type.
    attributes:
      symbol_id:
        type: str
      name:
        type: str
      type_:
        type: astx.DataType
      is_mutable:
        type: bool
      kind:
        type: str
      declaration:
        type: astx.AST | None
      module_key:
        type: ModuleKey
      qualified_name:
        type: str
    """

    symbol_id: str
    name: str
    type_: astx.DataType
    is_mutable: bool
    kind: str
    declaration: astx.AST | None = None
    module_key: ModuleKey = field(default_factory=lambda: "<unknown>")
    qualified_name: str = ""


@public
@typechecked
@dataclass(frozen=True)
class SemanticStruct:
    """
    title: Resolved struct information.
    summary: >-
      Describe one top-level struct declaration with module-aware identity.
    attributes:
      symbol_id:
        type: str
      name:
        type: str
      module_key:
        type: ModuleKey
      qualified_name:
        type: str
      declaration:
        type: astx.StructDefStmt
      fields:
        type: tuple[SemanticStructField, Ellipsis]
      field_indices:
        type: dict[str, int]
    """

    symbol_id: str
    name: str
    module_key: ModuleKey
    qualified_name: str
    declaration: astx.StructDefStmt
    fields: tuple["SemanticStructField", ...] = ()
    field_indices: dict[str, int] = field(default_factory=dict)


@public
@typechecked
@dataclass(frozen=True)
class SemanticStructField:
    """
    title: Resolved struct field information.
    summary: >-
      Describe one ordered field within a semantic struct, including its stable
      index and resolved field type.
    attributes:
      name:
        type: str
      index:
        type: int
      type_:
        type: astx.DataType
      declaration:
        type: astx.VariableDeclaration
    """

    name: str
    index: int
    type_: astx.DataType
    declaration: astx.VariableDeclaration


@public
@typechecked
class ParameterPassingKind(str, Enum):
    """
    title: Stable semantic parameter-passing modes.
    summary: >-
      Classify how one semantic parameter is passed across the callable
      boundary.
    """

    BY_VALUE = "by_value"


@public
@typechecked
class CallingConvention(str, Enum):
    """
    title: Stable semantic calling-convention classes.
    summary: >-
      Distinguish IRx-native callable semantics from C/native interop callables
      even when lowering currently emits the same LLVM calling convention.
    """

    IRX_DEFAULT = "irx_default"
    C = "c"


@public
@typechecked
class FFIAdmissibility(str, Enum):
    """
    title: Stable semantic FFI admissibility classes.
    summary: >-
      Distinguish regular IRx callables from callables that satisfy the public
      FFI contract.
    """

    INTERNAL_ONLY = "internal_only"
    PUBLIC = "public"


@public
@typechecked
class FFILinkStrategy(str, Enum):
    """
    title: Stable semantic native symbol-resolution strategies.
    summary: >-
      Describe whether an extern symbol relies only on the system linker or on
      one or more explicit runtime features.
    """

    SYSTEM_LINKER = "system_linker"
    RUNTIME_FEATURES = "runtime_features"


@public
@typechecked
class FFITypeClass(str, Enum):
    """
    title: Stable semantic FFI type classes.
    summary: >-
      Classify the narrow set of public ABI-safe value categories supported by
      IRx's current FFI contract.
    """

    BOOLEAN = "boolean"
    FLOAT = "float"
    INTEGER = "integer"
    OPAQUE_HANDLE = "opaque_handle"
    POINTER = "pointer"
    STRING = "string"
    STRUCT = "struct"
    VOID = "void"


@public
@typechecked
@dataclass(frozen=True)
class FFITypeInfo:
    """
    title: One canonical semantic FFI type description.
    summary: >-
      Describe how one semantically validated public FFI type participates in
      the stable ABI contract.
    attributes:
      classification:
        type: FFITypeClass
      display_name:
        type: str
      metadata:
        type: dict[str, Any]
    """

    classification: FFITypeClass
    display_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@public
@typechecked
@dataclass(frozen=True)
class FFICallableInfo:
    """
    title: Canonical public FFI callable metadata.
    summary: >-
      Record the validated FFI classification, type categories, symbol
      resolution strategy, and runtime-feature dependencies for one extern
      callable.
    attributes:
      admissibility:
        type: FFIAdmissibility
      parameters:
        type: tuple[FFITypeInfo, Ellipsis]
      return_type:
        type: FFITypeInfo
      required_runtime_features:
        type: tuple[str, Ellipsis]
      link_strategy:
        type: FFILinkStrategy
    """

    admissibility: FFIAdmissibility
    parameters: tuple[FFITypeInfo, ...]
    return_type: FFITypeInfo
    required_runtime_features: tuple[str, ...] = ()
    link_strategy: FFILinkStrategy = FFILinkStrategy.SYSTEM_LINKER


@public
@typechecked
@dataclass(frozen=True)
class ParameterSpec:
    """
    title: One canonical semantic parameter specification.
    summary: >-
      Describe one ordered callable parameter together with its declared type
      and passing policy, plus stable metadata such as default fingerprints.
    attributes:
      name:
        type: str
      type_:
        type: astx.DataType
      passing_kind:
        type: ParameterPassingKind
      metadata:
        type: dict[str, Any]
    """

    name: str
    type_: astx.DataType
    passing_kind: ParameterPassingKind = ParameterPassingKind.BY_VALUE
    metadata: dict[str, Any] = field(default_factory=dict)


@public
@typechecked
@dataclass(frozen=True)
class FunctionSignature:
    """
    title: Canonical semantic callable signature.
    summary: >-
      Normalize the stable callable contract that semantic analysis resolves
      and lowering consumes.
    attributes:
      name:
        type: str
      parameters:
        type: tuple[ParameterSpec, Ellipsis]
      return_type:
        type: astx.DataType
      calling_convention:
        type: CallingConvention
      is_variadic:
        type: bool
      is_extern:
        type: bool
      symbol_name:
        type: str
      required_runtime_features:
        type: tuple[str, Ellipsis]
      ffi:
        type: FFICallableInfo | None
      metadata:
        type: dict[str, Any]
    """

    name: str
    parameters: tuple[ParameterSpec, ...]
    return_type: astx.DataType
    calling_convention: CallingConvention
    is_variadic: bool = False
    is_extern: bool = False
    symbol_name: str = ""
    required_runtime_features: tuple[str, ...] = ()
    ffi: FFICallableInfo | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@public
@typechecked
class ClassMemberKind(str, Enum):
    """
    title: Stable semantic class-member categories.
    summary: Distinguish fields from methods inside one class member namespace.
    """

    ATTRIBUTE = "attribute"
    METHOD = "method"


@public
@typechecked
class ClassMemberResolutionKind(str, Enum):
    """
    title: Stable class-member resolution categories.
    summary: >-
      Distinguish members declared locally, members that override inherited
      declarations, and members inherited directly from the resolved MRO.
    """

    DECLARED = "declared"
    OVERRIDE = "override"
    INHERITED = "inherited"


@public
@typechecked
@dataclass(frozen=True)
class SemanticClassMember:
    """
    title: Canonical semantic class-member record.
    summary: >-
      Normalize one declared class member so later phases can reason about
      visibility, storage, mutability, and overrides without re-reading raw AST
      modifier fields.
    attributes:
      symbol_id:
        type: str
      name:
        type: str
      qualified_name:
        type: str
      owner_name:
        type: str
      owner_qualified_name:
        type: str
      kind:
        type: ClassMemberKind
      visibility:
        type: astx.VisibilityKind
      is_static:
        type: bool
      is_abstract:
        type: bool
      is_constant:
        type: bool
      is_mutable:
        type: bool
      declaration:
        type: astx.AST
      type_:
        type: astx.DataType | None
      signature:
        type: FunctionSignature | None
      signature_key:
        type: str | None
      overrides:
        type: str | None
      dispatch_slot:
        type: int | None
      lowered_function:
        type: SemanticFunction | None
    """

    symbol_id: str
    name: str
    qualified_name: str
    owner_name: str
    owner_qualified_name: str
    kind: ClassMemberKind
    visibility: astx.VisibilityKind
    is_static: bool
    is_abstract: bool
    is_constant: bool
    is_mutable: bool
    declaration: astx.AST
    type_: astx.DataType | None = None
    signature: FunctionSignature | None = None
    signature_key: str | None = None
    overrides: str | None = None
    dispatch_slot: int | None = None
    lowered_function: "SemanticFunction" | None = None


@public
@typechecked
@dataclass(frozen=True)
class SemanticClassMemberResolution:
    """
    title: Canonical class-member lookup resolution.
    summary: >-
      Record the ordered candidate set that one class member name considered
      and the member that won after deterministic inheritance resolution.
    attributes:
      name:
        type: str
      kind:
        type: ClassMemberResolutionKind
      selected:
        type: SemanticClassMember
      candidates:
        type: tuple[SemanticClassMember, Ellipsis]
      signature_key:
        type: str | None
    """

    name: str
    kind: ClassMemberResolutionKind
    selected: SemanticClassMember
    candidates: tuple[SemanticClassMember, ...] = ()
    signature_key: str | None = None


@public
@typechecked
class ClassObjectRepresentationKind(str, Enum):
    """
    title: Stable class object-representation categories.
    summary: >-
      Distinguish whether class values are modeled as pointers or values.
    """

    POINTER = "pointer"


@public
@typechecked
class MethodDispatchKind(str, Enum):
    """
    title: Stable method-dispatch categories.
    summary: >-
      Distinguish direct method calls from dispatch-table-driven instance
      calls.
    """

    DIRECT = "direct"
    INDIRECT = "indirect"


@public
@typechecked
class ClassHeaderFieldKind(str, Enum):
    """
    title: Stable class-header field categories.
    summary: >-
      Name the reserved object-header slots that later lowering and dispatch
      phases can populate without changing the class object layout.
    """

    TYPE_DESCRIPTOR = "type_descriptor"
    DISPATCH_TABLE = "dispatch_table"


@public
@typechecked
@dataclass(frozen=True)
class SemanticClassHeaderField:
    """
    title: One reserved class-object header slot.
    summary: >-
      Describe one hidden header entry that occupies a stable index in every
      class object representation.
    attributes:
      name:
        type: str
      kind:
        type: ClassHeaderFieldKind
      storage_index:
        type: int
    """

    name: str
    kind: ClassHeaderFieldKind
    storage_index: int


@public
@typechecked
@dataclass(frozen=True)
class SemanticClassLayoutField:
    """
    title: One resolved instance-field storage slot.
    summary: >-
      Record the stable storage position for one declared instance attribute in
      the flattened class-object layout.
    attributes:
      member:
        type: SemanticClassMember
      logical_index:
        type: int
      storage_index:
        type: int
      owner_name:
        type: str
      owner_qualified_name:
        type: str
    """

    member: SemanticClassMember
    logical_index: int
    storage_index: int
    owner_name: str
    owner_qualified_name: str


@public
@typechecked
@dataclass(frozen=True)
class SemanticClassStaticStorage:
    """
    title: One resolved static-member storage record.
    summary: >-
      Describe the module-global storage backing one declared static class
      attribute.
    attributes:
      member:
        type: SemanticClassMember
      global_name:
        type: str
      owner_name:
        type: str
      owner_qualified_name:
        type: str
    """

    member: SemanticClassMember
    global_name: str
    owner_name: str
    owner_qualified_name: str


@public
@typechecked
@dataclass(frozen=True)
class SemanticClassMethodDispatch:
    """
    title: One resolved class-method dispatch entry.
    summary: >-
      Record the stable dispatch slot and lowered function implementation for
      one visible instance method.
    attributes:
      member:
        type: SemanticClassMember
      function:
        type: SemanticFunction
      slot_index:
        type: int
      owner_name:
        type: str
      owner_qualified_name:
        type: str
    """

    member: SemanticClassMember
    function: "SemanticFunction"
    slot_index: int
    owner_name: str
    owner_qualified_name: str


@public
@typechecked
class ClassInitializationSourceKind(str, Enum):
    """
    title: Stable class-initialization source categories.
    summary: >-
      Distinguish declaration-provided field initializers from implicit default
      construction values.
    """

    DECLARATION = "declaration"
    DEFAULT = "default"


@public
@typechecked
@dataclass(frozen=True)
class SemanticClassFieldInitializer:
    """
    title: One resolved instance-field initialization step.
    summary: >-
      Record the ordered value source for one instance field during default
      class construction.
    attributes:
      field:
        type: SemanticClassLayoutField
      source_kind:
        type: ClassInitializationSourceKind
      value:
        type: astx.AST | None
      owner_name:
        type: str
      owner_qualified_name:
        type: str
    """

    field: SemanticClassLayoutField
    source_kind: ClassInitializationSourceKind
    value: astx.AST | None
    owner_name: str
    owner_qualified_name: str


@public
@typechecked
@dataclass(frozen=True)
class SemanticClassStaticInitializer:
    """
    title: One resolved static-field initialization step.
    summary: >-
      Record the deterministic module-global initializer source for one static
      class attribute.
    attributes:
      storage:
        type: SemanticClassStaticStorage
      source_kind:
        type: ClassInitializationSourceKind
      value:
        type: astx.AST | None
      owner_name:
        type: str
      owner_qualified_name:
        type: str
    """

    storage: SemanticClassStaticStorage
    source_kind: ClassInitializationSourceKind
    value: astx.AST | None
    owner_name: str
    owner_qualified_name: str


@public
@typechecked
@dataclass(frozen=True)
class SemanticClassInitialization:
    """
    title: Canonical class construction and initialization plan.
    summary: >-
      Normalize the ordered instance-field and static-field initialization plan
      that semantic analysis resolves for one class.
    attributes:
      instance_initializers:
        type: tuple[SemanticClassFieldInitializer, Ellipsis]
      static_initializers:
        type: tuple[SemanticClassStaticInitializer, Ellipsis]
    """

    instance_initializers: tuple[SemanticClassFieldInitializer, ...] = ()
    static_initializers: tuple[SemanticClassStaticInitializer, ...] = ()


@public
@typechecked
@dataclass(frozen=True)
class SemanticClassLayout:
    """
    title: Canonical class-object layout metadata.
    summary: >-
      Normalize the low-level object representation, hidden header slots,
      flattened instance-field storage, and static-global storage names for one
      class.
    attributes:
      llvm_name:
        type: str
      object_representation:
        type: ClassObjectRepresentationKind
      descriptor_global_name:
        type: str
      dispatch_global_name:
        type: str
      header_fields:
        type: tuple[SemanticClassHeaderField, Ellipsis]
      instance_fields:
        type: tuple[SemanticClassLayoutField, Ellipsis]
      field_slots:
        type: dict[str, SemanticClassLayoutField]
      visible_field_slots:
        type: dict[str, SemanticClassLayoutField]
      dispatch_entries:
        type: tuple[SemanticClassMethodDispatch, Ellipsis]
      dispatch_slots:
        type: dict[int, SemanticClassMethodDispatch]
      visible_method_slots:
        type: dict[str, SemanticClassMethodDispatch]
      dispatch_table_size:
        type: int
      static_fields:
        type: tuple[SemanticClassStaticStorage, Ellipsis]
      static_storage:
        type: dict[str, SemanticClassStaticStorage]
      visible_static_storage:
        type: dict[str, SemanticClassStaticStorage]
    """

    llvm_name: str
    object_representation: ClassObjectRepresentationKind
    descriptor_global_name: str
    dispatch_global_name: str
    header_fields: tuple[SemanticClassHeaderField, ...] = ()
    instance_fields: tuple[SemanticClassLayoutField, ...] = ()
    field_slots: dict[str, SemanticClassLayoutField] = field(
        default_factory=dict
    )
    visible_field_slots: dict[str, SemanticClassLayoutField] = field(
        default_factory=dict
    )
    dispatch_entries: tuple[SemanticClassMethodDispatch, ...] = ()
    dispatch_slots: dict[int, SemanticClassMethodDispatch] = field(
        default_factory=dict
    )
    visible_method_slots: dict[str, SemanticClassMethodDispatch] = field(
        default_factory=dict
    )
    dispatch_table_size: int = 0
    static_fields: tuple[SemanticClassStaticStorage, ...] = ()
    static_storage: dict[str, SemanticClassStaticStorage] = field(
        default_factory=dict
    )
    visible_static_storage: dict[str, SemanticClassStaticStorage] = field(
        default_factory=dict
    )


@public
@typechecked
@dataclass(frozen=True)
class SemanticClass:
    """
    title: Resolved class information.
    summary: >-
      Describe one top-level class declaration together with normalized bases,
      member tables, and deterministic inheritance metadata.
    attributes:
      symbol_id:
        type: str
      name:
        type: str
      module_key:
        type: ModuleKey
      qualified_name:
        type: str
      declaration:
        type: astx.ClassDefStmt
      bases:
        type: tuple[SemanticClass, Ellipsis]
      declared_members:
        type: tuple[SemanticClassMember, Ellipsis]
      declared_member_table:
        type: dict[str, SemanticClassMember]
      declared_method_groups:
        type: dict[str, tuple[SemanticClassMember, Ellipsis]]
      member_table:
        type: dict[str, SemanticClassMember]
      method_groups:
        type: dict[str, tuple[SemanticClassMember, Ellipsis]]
      member_resolution:
        type: dict[str, SemanticClassMemberResolution]
      method_resolution:
        type: dict[str, tuple[SemanticClassMemberResolution, Ellipsis]]
      instance_attributes:
        type: tuple[SemanticClassMember, Ellipsis]
      static_attributes:
        type: tuple[SemanticClassMember, Ellipsis]
      instance_methods:
        type: tuple[SemanticClassMember, Ellipsis]
      static_methods:
        type: tuple[SemanticClassMember, Ellipsis]
      abstract_methods:
        type: tuple[SemanticClassMember, Ellipsis]
      inheritance_graph:
        type: tuple[str, Ellipsis]
      shared_ancestors:
        type: tuple[SemanticClass, Ellipsis]
      layout:
        type: SemanticClassLayout | None
      initialization:
        type: SemanticClassInitialization | None
      mro:
        type: tuple[SemanticClass, Ellipsis]
      is_structurally_resolved:
        type: bool
      is_resolved:
        type: bool
      is_abstract:
        type: bool
    """

    symbol_id: str
    name: str
    module_key: ModuleKey
    qualified_name: str
    declaration: astx.ClassDefStmt
    bases: tuple["SemanticClass", ...] = ()
    declared_members: tuple[SemanticClassMember, ...] = ()
    declared_member_table: dict[str, SemanticClassMember] = field(
        default_factory=dict
    )
    declared_method_groups: dict[str, tuple[SemanticClassMember, ...]] = field(
        default_factory=dict
    )
    member_table: dict[str, SemanticClassMember] = field(default_factory=dict)
    method_groups: dict[str, tuple[SemanticClassMember, ...]] = field(
        default_factory=dict
    )
    member_resolution: dict[str, SemanticClassMemberResolution] = field(
        default_factory=dict
    )
    method_resolution: dict[str, tuple[SemanticClassMemberResolution, ...]] = (
        field(default_factory=dict)
    )
    instance_attributes: tuple[SemanticClassMember, ...] = ()
    static_attributes: tuple[SemanticClassMember, ...] = ()
    instance_methods: tuple[SemanticClassMember, ...] = ()
    static_methods: tuple[SemanticClassMember, ...] = ()
    abstract_methods: tuple[SemanticClassMember, ...] = ()
    inheritance_graph: tuple[str, ...] = ()
    shared_ancestors: tuple["SemanticClass", ...] = ()
    layout: SemanticClassLayout | None = None
    initialization: SemanticClassInitialization | None = None
    mro: tuple["SemanticClass", ...] = ()
    is_structurally_resolved: bool = False
    is_resolved: bool = False
    is_abstract: bool = False


@public
@typechecked
@dataclass(frozen=True)
class ImplicitConversion:
    """
    title: One semantically inserted implicit conversion.
    summary: >-
      Record one source-to-target type conversion that semantic analysis
      validated and lowering should honor directly.
    attributes:
      source_type:
        type: astx.DataType | None
      target_type:
        type: astx.DataType | None
    """

    source_type: astx.DataType | None
    target_type: astx.DataType | None


@public
@typechecked
@dataclass(frozen=True)
class TemplateArgumentBinding:
    """
    title: One concrete template argument binding.
    summary: >-
      Record the concrete type selected for one named template parameter in a
      specialization.
    attributes:
      name:
        type: str
      type_:
        type: astx.DataType
    """

    name: str
    type_: astx.DataType


@public
@typechecked
@dataclass(frozen=True)
class TemplateSpecializationKey:
    """
    title: Stable semantic template-specialization identity.
    summary: >-
      Identify one concrete specialization of a template function by the
      original semantic function name and the ordered concrete type names.
    attributes:
      qualified_name:
        type: str
      arg_type_names:
        type: tuple[str, Ellipsis]
    """

    qualified_name: str
    arg_type_names: tuple[str, ...]


@public
@typechecked
@dataclass(frozen=True)
class SemanticFunction:
    """
    title: Resolved function information.
    summary: >-
      Describe one top-level function declaration or definition together with
      its semantic identity, canonical signature, and argument symbols.
    attributes:
      symbol_id:
        type: str
      name:
        type: str
      return_type:
        type: astx.DataType
      args:
        type: tuple[SemanticSymbol, Ellipsis]
      signature:
        type: FunctionSignature
      prototype:
        type: astx.FunctionPrototype
      definition:
        type: astx.FunctionDef | None
      module_key:
        type: ModuleKey
      qualified_name:
        type: str
      template_params:
        type: tuple[astx.TemplateParam, Ellipsis]
      template_bindings:
        type: tuple[TemplateArgumentBinding, Ellipsis]
      template_definition:
        type: SemanticFunction | None
      specialization_key:
        type: TemplateSpecializationKey | None
      specializations:
        type: dict[TemplateSpecializationKey, SemanticFunction]
    """

    symbol_id: str
    name: str
    return_type: astx.DataType
    args: tuple[SemanticSymbol, ...]
    signature: FunctionSignature
    prototype: astx.FunctionPrototype
    definition: astx.FunctionDef | None = None
    module_key: ModuleKey = field(default_factory=lambda: "<unknown>")
    qualified_name: str = ""
    template_params: tuple[astx.TemplateParam, ...] = ()
    template_bindings: tuple[TemplateArgumentBinding, ...] = ()
    template_definition: "SemanticFunction" | None = None
    specialization_key: TemplateSpecializationKey | None = None
    specializations: dict[TemplateSpecializationKey, "SemanticFunction"] = (
        field(default_factory=dict, compare=False)
    )


@public
@typechecked
@dataclass(frozen=True)
class CallableResolution:
    """
    title: Resolved callable identity.
    summary: >-
      Point from one semantic site to the canonical callable identity and
      signature that analysis resolved.
    attributes:
      function:
        type: SemanticFunction
      signature:
        type: FunctionSignature
    """

    function: SemanticFunction
    signature: FunctionSignature


@public
@typechecked
@dataclass(frozen=True)
class CallResolution:
    """
    title: Resolved function-call semantics.
    summary: >-
      Capture the canonical callee, validated argument conversions, and result
      type for one call site.
    attributes:
      callee:
        type: CallableResolution
      signature:
        type: FunctionSignature
      resolved_argument_types:
        type: tuple[astx.DataType | None, Ellipsis]
      result_type:
        type: astx.DataType
      implicit_conversions:
        type: tuple[ImplicitConversion | None, Ellipsis]
    """

    callee: CallableResolution
    signature: FunctionSignature
    resolved_argument_types: tuple[astx.DataType | None, ...]
    result_type: astx.DataType
    implicit_conversions: tuple[ImplicitConversion | None, ...] = ()


@public
@typechecked
@dataclass(frozen=True)
class ReturnResolution:
    """
    title: Resolved return-statement semantics.
    summary: >-
      Capture how one return statement relates to the enclosing function
      signature and any implicit conversion that analysis inserted.
    attributes:
      callable:
        type: CallableResolution
      expected_type:
        type: astx.DataType
      value_type:
        type: astx.DataType | None
      returns_void:
        type: bool
      implicit_conversion:
        type: ImplicitConversion | None
    """

    callable: CallableResolution
    expected_type: astx.DataType
    value_type: astx.DataType | None
    returns_void: bool
    implicit_conversion: ImplicitConversion | None = None


@public
@typechecked
@dataclass(frozen=True)
class ResolvedGeneratorFunction:
    """
    title: Resolved generator-function semantics.
    summary: >-
      Describe one function body that suspends at yield sites and returns a
      first-class generator object from call sites.
    attributes:
      function:
        type: SemanticFunction
      yield_type:
        type: astx.DataType
      yield_nodes:
        type: tuple[astx.AST, Ellipsis]
    """

    function: "SemanticFunction"
    yield_type: astx.DataType
    yield_nodes: tuple[astx.AST, ...] = ()


@public
@typechecked
@dataclass(frozen=True)
class ResolvedYield:
    """
    title: Resolved yield-site semantics.
    summary: >-
      Capture how one yield statement or expression maps to its enclosing
      generator function and yielded element type.
    attributes:
      generator:
        type: ResolvedGeneratorFunction
      expected_type:
        type: astx.DataType
      value_type:
        type: astx.DataType | None
      site_index:
        type: int
      implicit_conversion:
        type: ImplicitConversion | None
    """

    generator: ResolvedGeneratorFunction
    expected_type: astx.DataType
    value_type: astx.DataType | None
    site_index: int
    implicit_conversion: ImplicitConversion | None = None


@public
@typechecked
@dataclass(frozen=True)
class SemanticModule:
    """
    title: Semantic identity for an imported module.
    summary: >-
      Represent a module binding that plain imports introduce into a module
      namespace.
    attributes:
      module_key:
        type: ModuleKey
      display_name:
        type: str | None
    """

    module_key: ModuleKey
    display_name: str | None = None


@public
@typechecked
@dataclass(frozen=True)
class SemanticBinding:
    """
    title: One visible top-level binding in a module namespace.
    summary: >-
      Normalize imported and local top-level names into one binding shape for
      module-visible lookup.
    attributes:
      kind:
        type: str
      module_key:
        type: ModuleKey
      qualified_name:
        type: str
      function:
        type: SemanticFunction | None
      struct:
        type: SemanticStruct | None
      class_:
        type: SemanticClass | None
      module:
        type: SemanticModule | None
    """

    kind: str
    module_key: ModuleKey
    qualified_name: str
    function: SemanticFunction | None = None
    struct: SemanticStruct | None = None
    class_: SemanticClass | None = None
    module: SemanticModule | None = None


@public
@typechecked
@dataclass(frozen=True)
class ResolvedImportBinding:
    """
    title: One resolved imported local binding.
    summary: >-
      Record how one imported local name maps back to its source-module
      declaration.
    attributes:
      local_name:
        type: str
      requested_name:
        type: str
      source_module_key:
        type: ModuleKey
      binding:
        type: SemanticBinding
    """

    local_name: str
    requested_name: str
    source_module_key: ModuleKey
    binding: SemanticBinding


@public
@typechecked
@dataclass(frozen=True)
class ResolvedModuleMemberAccess:
    """
    title: Resolved module-namespace member access.
    summary: >-
      Record which imported module namespace one member access targeted and the
      visible binding that namespace lookup selected.
    attributes:
      module:
        type: SemanticModule
      member_name:
        type: str
      binding:
        type: SemanticBinding
    """

    module: SemanticModule
    member_name: str
    binding: SemanticBinding


@public
@typechecked
@dataclass(frozen=True)
class SemanticFlags:
    """
    title: Normalized semantic flags.
    summary: >-
      Store normalized semantic modifiers such as unsigned and fast-math
      intent.
    attributes:
      unsigned:
        type: bool
      fast_math:
        type: bool
      fma:
        type: bool
      fma_rhs:
        type: astx.AST | None
    """

    unsigned: bool = False
    fast_math: bool = False
    fma: bool = False
    fma_rhs: astx.AST | None = None


@public
@typechecked
@dataclass(frozen=True)
class ResolvedOperator:
    """
    title: Normalized operator meaning.
    summary: >-
      Capture the normalized operator opcode, operand types, result type, and
      semantic flags for one expression.
    attributes:
      op_code:
        type: str
      result_type:
        type: astx.DataType | None
      lhs_type:
        type: astx.DataType | None
      rhs_type:
        type: astx.DataType | None
      flags:
        type: SemanticFlags
    """

    op_code: str
    result_type: astx.DataType | None = None
    lhs_type: astx.DataType | None = None
    rhs_type: astx.DataType | None = None
    flags: SemanticFlags = field(default_factory=SemanticFlags)


@public
@typechecked
@dataclass(frozen=True)
class ResolvedAssignment:
    """
    title: Resolved assignment target.
    summary: >-
      Point from an assignment-like node back to the resolved target symbol it
      mutates.
    attributes:
      target:
        type: SemanticSymbol
    """

    target: SemanticSymbol


@public
@typechecked
@dataclass(frozen=True)
class ResolvedFieldAccess:
    """
    title: Resolved field access metadata.
    summary: >-
      Point from a field-access node to its owning struct and stable field
      metadata.
    attributes:
      struct:
        type: SemanticStruct
      field:
        type: SemanticStructField
    """

    struct: SemanticStruct
    field: SemanticStructField


@public
@typechecked
@dataclass(frozen=True)
class ResolvedClassFieldAccess:
    """
    title: Resolved class-field access metadata.
    summary: >-
      Point from a class-attribute access node to the owning class member and
      stable flattened layout slot.
    attributes:
      class_:
        type: SemanticClass
      member:
        type: SemanticClassMember
      field:
        type: SemanticClassLayoutField
    """

    class_: SemanticClass
    member: SemanticClassMember
    field: SemanticClassLayoutField


@public
@typechecked
@dataclass(frozen=True)
class ResolvedBaseClassFieldAccess:
    """
    title: Resolved explicit base-class field access metadata.
    summary: >-
      Point from a base-qualified instance attribute read to the selected base
      view, concrete receiver class, and stable flattened layout slot.
    attributes:
      receiver_class:
        type: SemanticClass
      base_class:
        type: SemanticClass
      member:
        type: SemanticClassMember
      field:
        type: SemanticClassLayoutField
    """

    receiver_class: SemanticClass
    base_class: SemanticClass
    member: SemanticClassMember
    field: SemanticClassLayoutField


@public
@typechecked
@dataclass(frozen=True)
class ResolvedStaticClassFieldAccess:
    """
    title: Resolved static class-field access metadata.
    summary: >-
      Point from a class-qualified static attribute read to the selected class
      member and stable emitted storage metadata.
    attributes:
      class_:
        type: SemanticClass
      member:
        type: SemanticClassMember
      storage:
        type: SemanticClassStaticStorage
    """

    class_: SemanticClass
    member: SemanticClassMember
    storage: SemanticClassStaticStorage


@public
@typechecked
@dataclass(frozen=True)
class ResolvedClassConstruction:
    """
    title: Resolved class construction metadata.
    summary: >-
      Capture the analyzed class identity and ordered initialization plan for
      one default class construction expression.
    attributes:
      class_:
        type: SemanticClass
      initialization:
        type: SemanticClassInitialization
    """

    class_: SemanticClass
    initialization: SemanticClassInitialization


@public
@typechecked
@dataclass(frozen=True)
class ResolvedMethodCall:
    """
    title: Resolved class method call metadata.
    summary: >-
      Capture the resolved class member, lowered implementation, dispatch mode,
      and validated argument conversions for one method call site.
    attributes:
      class_:
        type: SemanticClass
      member:
        type: SemanticClassMember
      function:
        type: SemanticFunction
      overload_key:
        type: str
      dispatch_kind:
        type: MethodDispatchKind
      call:
        type: CallResolution
      candidates:
        type: tuple[SemanticClassMember, Ellipsis]
      receiver_type:
        type: astx.DataType | None
      receiver_class:
        type: SemanticClass | None
      slot_index:
        type: int | None
    """

    class_: SemanticClass
    member: SemanticClassMember
    function: SemanticFunction
    overload_key: str
    dispatch_kind: MethodDispatchKind
    call: CallResolution
    candidates: tuple[SemanticClassMember, ...] = ()
    receiver_type: astx.DataType | None = None
    receiver_class: SemanticClass | None = None
    slot_index: int | None = None


@public
@typechecked
@dataclass(frozen=True)
class ResolvedContextManager:
    """
    title: Resolved context-manager metadata.
    summary: >-
      Capture the manager class, resolved ``__enter__``/``__exit__`` methods,
      and optional target binding for one ``with`` statement.
    attributes:
      class_:
        type: SemanticClass
      manager_type:
        type: astx.DataType
      enter:
        type: ResolvedMethodCall
      exit:
        type: ResolvedMethodCall
      target_symbol:
        type: SemanticSymbol | None
    """

    class_: SemanticClass
    manager_type: astx.DataType
    enter: ResolvedMethodCall
    exit: ResolvedMethodCall
    target_symbol: SemanticSymbol | None = None


@public
@typechecked
class IterationKind(str, Enum):
    """
    title: Stable iterable adapter kinds.
    summary: >-
      Classify the semantic adapter that turns one iterable expression into a
      backend iteration plan.
    """

    LIST = "list"
    DICT_KEYS = "dict_keys"
    SET = "set"
    GENERATOR = "generator"
    RANGE = "range"
    CUSTOM = "custom"


@public
@typechecked
class IterationOrder(str, Enum):
    """
    title: Stable iterable ordering categories.
    summary: >-
      Describe the user-visible order guarantee, if any, exposed by one
      iterable adapter.
    """

    INDEX = "index"
    INSERTION = "insertion"
    STABLE = "stable"
    UNSPECIFIED = "unspecified"


@public
@typechecked
@dataclass(frozen=True)
class ResolvedIteration:
    """
    title: Resolved iterable capability.
    summary: >-
      Attach the semantic iteration plan that a for-in loop or comprehension
      should consume during backend lowering.
    attributes:
      iterable_node:
        type: astx.AST
      iterable_type:
        type: astx.DataType
      element_type:
        type: astx.DataType
      kind:
        type: IterationKind
      is_reiterable:
        type: bool
      order:
        type: IterationOrder
      target_symbol:
        type: SemanticSymbol | None
      extras:
        type: dict[str, Any]
    """

    iterable_node: astx.AST
    iterable_type: astx.DataType
    element_type: astx.DataType
    kind: IterationKind
    is_reiterable: bool = True
    order: IterationOrder = IterationOrder.UNSPECIFIED
    target_symbol: SemanticSymbol | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@public
@typechecked
class CollectionMethodKind(str, Enum):
    """
    title: Stable collection-method semantic categories.
    """

    LENGTH = "length"
    IS_EMPTY = "is_empty"
    CONTAINS = "contains"
    INDEX = "index"
    COUNT = "count"


@public
@typechecked
@dataclass(frozen=True)
class ResolvedCollectionMethod:
    """
    title: Resolved collection method capability.
    summary: >-
      Attach the semantic collection operation that lowering should consume
      without re-resolving receiver kind or result type.
    attributes:
      receiver_node:
        type: astx.AST
      receiver_type:
        type: astx.DataType
      method:
        type: CollectionMethodKind
      return_type:
        type: astx.DataType
      argument_types:
        type: tuple[astx.DataType, Ellipsis]
      mutates:
        type: bool
      extras:
        type: dict[str, Any]
    """

    receiver_node: astx.AST
    receiver_type: astx.DataType
    method: CollectionMethodKind
    return_type: astx.DataType
    argument_types: tuple[astx.DataType, ...] = ()
    mutates: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


@public
@typechecked
@dataclass
class SemanticInfo:
    """
    title: Sidecar semantic information stored on AST nodes.
    summary: >-
      Aggregate all semantic sidecar fields that analysis may attach to a
      single AST node.
    attributes:
      resolved_type:
        type: astx.DataType | None
      resolved_symbol:
        type: SemanticSymbol | None
      resolved_function:
        type: SemanticFunction | None
      resolved_callable:
        type: CallableResolution | None
      resolved_struct:
        type: SemanticStruct | None
      resolved_class:
        type: SemanticClass | None
      resolved_module:
        type: SemanticModule | None
      resolved_imports:
        type: tuple[ResolvedImportBinding, Ellipsis]
      resolved_call:
        type: CallResolution | None
      resolved_operator:
        type: ResolvedOperator | None
      resolved_assignment:
        type: ResolvedAssignment | None
      resolved_field_access:
        type: ResolvedFieldAccess | None
      resolved_module_member_access:
        type: ResolvedModuleMemberAccess | None
      resolved_class_field_access:
        type: ResolvedClassFieldAccess | None
      resolved_base_class_field_access:
        type: ResolvedBaseClassFieldAccess | None
      resolved_static_class_field_access:
        type: ResolvedStaticClassFieldAccess | None
      resolved_method_call:
        type: ResolvedMethodCall | None
      resolved_context_manager:
        type: ResolvedContextManager | None
      resolved_class_construction:
        type: ResolvedClassConstruction | None
      resolved_return:
        type: ReturnResolution | None
      resolved_generator_function:
        type: ResolvedGeneratorFunction | None
      resolved_yield:
        type: ResolvedYield | None
      resolved_iteration:
        type: ResolvedIteration | None
      resolved_collection_method:
        type: ResolvedCollectionMethod | None
      semantic_flags:
        type: SemanticFlags
      extras:
        type: dict[str, Any]
    """

    resolved_type: astx.DataType | None = None
    resolved_symbol: SemanticSymbol | None = None
    resolved_function: SemanticFunction | None = None
    resolved_callable: CallableResolution | None = None
    resolved_struct: SemanticStruct | None = None
    resolved_class: SemanticClass | None = None
    resolved_module: SemanticModule | None = None
    resolved_imports: tuple[ResolvedImportBinding, ...] = ()
    resolved_call: CallResolution | None = None
    resolved_operator: ResolvedOperator | None = None
    resolved_assignment: ResolvedAssignment | None = None
    resolved_field_access: ResolvedFieldAccess | None = None
    resolved_module_member_access: ResolvedModuleMemberAccess | None = None
    resolved_class_field_access: ResolvedClassFieldAccess | None = None
    resolved_base_class_field_access: ResolvedBaseClassFieldAccess | None = (
        None
    )
    resolved_static_class_field_access: (
        ResolvedStaticClassFieldAccess | None
    ) = None
    resolved_method_call: ResolvedMethodCall | None = None
    resolved_context_manager: ResolvedContextManager | None = None
    resolved_class_construction: ResolvedClassConstruction | None = None
    resolved_return: ReturnResolution | None = None
    resolved_generator_function: ResolvedGeneratorFunction | None = None
    resolved_yield: ResolvedYield | None = None
    resolved_iteration: ResolvedIteration | None = None
    resolved_collection_method: ResolvedCollectionMethod | None = None
    semantic_flags: SemanticFlags = field(default_factory=SemanticFlags)
    extras: dict[str, Any] = field(default_factory=dict)

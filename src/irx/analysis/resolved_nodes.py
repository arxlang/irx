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
@dataclass(frozen=True)
class ParameterSpec:
    """
    title: One canonical semantic parameter specification.
    summary: >-
      Describe one ordered callable parameter together with its declared type
      and passing policy.
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
    metadata: dict[str, Any] = field(default_factory=dict)


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
      module:
        type: SemanticModule | None
    """

    kind: str
    module_key: ModuleKey
    qualified_name: str
    function: SemanticFunction | None = None
    struct: SemanticStruct | None = None
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
      resolved_return:
        type: ReturnResolution | None
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
    resolved_module: SemanticModule | None = None
    resolved_imports: tuple[ResolvedImportBinding, ...] = ()
    resolved_call: CallResolution | None = None
    resolved_operator: ResolvedOperator | None = None
    resolved_assignment: ResolvedAssignment | None = None
    resolved_field_access: ResolvedFieldAccess | None = None
    resolved_return: ReturnResolution | None = None
    semantic_flags: SemanticFlags = field(default_factory=SemanticFlags)
    extras: dict[str, Any] = field(default_factory=dict)

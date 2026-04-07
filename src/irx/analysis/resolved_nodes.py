"""
title: Sidecar semantic dataclasses attached to AST nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from public import public

from irx import astx
from irx.analysis.module_interfaces import ModuleKey


@public
@dataclass(frozen=True)
class SemanticSymbol:
    """
    title: Resolved symbol information.
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
@dataclass(frozen=True)
class SemanticStruct:
    """
    title: Resolved struct information.
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
    """

    symbol_id: str
    name: str
    module_key: ModuleKey
    qualified_name: str
    declaration: astx.StructDefStmt


@public
@dataclass(frozen=True)
class SemanticFunction:
    """
    title: Resolved function information.
    attributes:
      symbol_id:
        type: str
      name:
        type: str
      return_type:
        type: astx.DataType
      args:
        type: tuple[SemanticSymbol, Ellipsis]
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
    prototype: astx.FunctionPrototype
    definition: astx.FunctionDef | None = None
    module_key: ModuleKey = field(default_factory=lambda: "<unknown>")
    qualified_name: str = ""


@public
@dataclass(frozen=True)
class SemanticModule:
    """
    title: Semantic identity for an imported module.
    attributes:
      module_key:
        type: ModuleKey
      display_name:
        type: str | None
    """

    module_key: ModuleKey
    display_name: str | None = None


@public
@dataclass(frozen=True)
class SemanticBinding:
    """
    title: One visible top-level binding in a module namespace.
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
@dataclass(frozen=True)
class ResolvedImportBinding:
    """
    title: One resolved imported local binding.
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
@dataclass(frozen=True)
class SemanticFlags:
    """
    title: Normalized semantic flags.
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
@dataclass(frozen=True)
class ResolvedOperator:
    """
    title: Normalized operator meaning.
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
@dataclass(frozen=True)
class ResolvedAssignment:
    """
    title: Resolved assignment target.
    attributes:
      target:
        type: SemanticSymbol
    """

    target: SemanticSymbol


@public
@dataclass
class SemanticInfo:
    """
    title: Sidecar semantic information stored on AST nodes.
    attributes:
      resolved_type:
        type: astx.DataType | None
      resolved_symbol:
        type: SemanticSymbol | None
      resolved_function:
        type: SemanticFunction | None
      resolved_struct:
        type: SemanticStruct | None
      resolved_module:
        type: SemanticModule | None
      resolved_imports:
        type: tuple[ResolvedImportBinding, Ellipsis]
      resolved_operator:
        type: ResolvedOperator | None
      resolved_assignment:
        type: ResolvedAssignment | None
      semantic_flags:
        type: SemanticFlags
      extras:
        type: dict[str, Any]
    """

    resolved_type: astx.DataType | None = None
    resolved_symbol: SemanticSymbol | None = None
    resolved_function: SemanticFunction | None = None
    resolved_struct: SemanticStruct | None = None
    resolved_module: SemanticModule | None = None
    resolved_imports: tuple[ResolvedImportBinding, ...] = ()
    resolved_operator: ResolvedOperator | None = None
    resolved_assignment: ResolvedAssignment | None = None
    semantic_flags: SemanticFlags = field(default_factory=SemanticFlags)
    extras: dict[str, Any] = field(default_factory=dict)

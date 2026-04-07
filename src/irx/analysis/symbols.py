"""
title: Symbol records for semantic analysis.
summary: >-
  Build the semantic symbol dataclasses that the analyzer stores in sidecars
  and registries.
"""

from __future__ import annotations

from dataclasses import replace

from public import public

from irx import astx
from irx.analysis.module_interfaces import ModuleKey
from irx.analysis.module_symbols import (
    qualified_function_name,
    qualified_local_name,
    qualified_struct_name,
)
from irx.analysis.resolved_nodes import (
    SemanticFunction,
    SemanticStruct,
    SemanticSymbol,
)


@public
def variable_symbol(
    symbol_id: str,
    module_key: ModuleKey,
    name: str,
    type_: astx.DataType,
    *,
    is_mutable: bool,
    declaration: astx.AST | None,
    kind: str = "variable",
) -> SemanticSymbol:
    """
    title: Create a variable-like symbol.
    summary: Create a variable-like symbol.
    parameters:
      symbol_id:
        type: str
      module_key:
        type: ModuleKey
      name:
        type: str
      type_:
        type: astx.DataType
      is_mutable:
        type: bool
      declaration:
        type: astx.AST | None
      kind:
        type: str
    returns:
      type: SemanticSymbol
    """
    return SemanticSymbol(
        symbol_id=symbol_id,
        name=name,
        type_=type_,
        is_mutable=is_mutable,
        kind=kind,
        declaration=declaration,
        module_key=module_key,
        qualified_name=qualified_local_name(module_key, kind, name, symbol_id),
    )


@public
def function_symbol(
    symbol_id: str,
    module_key: ModuleKey,
    prototype: astx.FunctionPrototype,
    args: tuple[SemanticSymbol, ...],
    *,
    definition: astx.FunctionDef | None = None,
) -> SemanticFunction:
    """
    title: Create a function symbol.
    summary: Create a function symbol.
    parameters:
      symbol_id:
        type: str
      module_key:
        type: ModuleKey
      prototype:
        type: astx.FunctionPrototype
      args:
        type: tuple[SemanticSymbol, Ellipsis]
      definition:
        type: astx.FunctionDef | None
    returns:
      type: SemanticFunction
    """
    return SemanticFunction(
        symbol_id=symbol_id,
        name=prototype.name,
        return_type=prototype.return_type,
        args=args,
        prototype=prototype,
        definition=definition,
        module_key=module_key,
        qualified_name=qualified_function_name(module_key, prototype.name),
    )


@public
def struct_symbol(
    symbol_id: str,
    module_key: ModuleKey,
    declaration: astx.StructDefStmt,
) -> SemanticStruct:
    """
    title: Create a struct symbol.
    summary: Create a struct symbol.
    parameters:
      symbol_id:
        type: str
      module_key:
        type: ModuleKey
      declaration:
        type: astx.StructDefStmt
    returns:
      type: SemanticStruct
    """
    return SemanticStruct(
        symbol_id=symbol_id,
        name=declaration.name,
        module_key=module_key,
        qualified_name=qualified_struct_name(module_key, declaration.name),
        declaration=declaration,
    )


@public
def with_definition(
    symbol: SemanticFunction, definition: astx.FunctionDef
) -> SemanticFunction:
    """
    title: Return a function symbol updated with its definition.
    summary: Return a function symbol updated with its definition.
    parameters:
      symbol:
        type: SemanticFunction
      definition:
        type: astx.FunctionDef
    returns:
      type: SemanticFunction
    """
    return replace(symbol, definition=definition)

"""
title: Symbol records for semantic analysis.
"""

from __future__ import annotations

from dataclasses import replace

from public import public

from irx import astx
from irx.analysis.resolved_nodes import SemanticFunction, SemanticSymbol


@public
def variable_symbol(
    symbol_id: str,
    name: str,
    type_: astx.DataType,
    *,
    is_mutable: bool,
    declaration: astx.AST | None,
    kind: str = "variable",
) -> SemanticSymbol:
    """
    title: Create a variable-like symbol.
    parameters:
      symbol_id:
        type: str
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
    )


@public
def function_symbol(
    symbol_id: str,
    prototype: astx.FunctionPrototype,
    args: tuple[SemanticSymbol, ...],
    *,
    definition: astx.FunctionDef | None = None,
) -> SemanticFunction:
    """
    title: Create a function symbol.
    parameters:
      symbol_id:
        type: str
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
    )


@public
def with_definition(
    symbol: SemanticFunction, definition: astx.FunctionDef
) -> SemanticFunction:
    """
    title: Return a function symbol updated with its definition.
    parameters:
      symbol:
        type: SemanticFunction
      definition:
        type: astx.FunctionDef
    returns:
      type: SemanticFunction
    """
    return replace(symbol, definition=definition)

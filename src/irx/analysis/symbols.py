"""
title: Variable-symbol helpers for semantic analysis.
summary: >-
  Build variable-like semantic symbols that need local-style qualified names.
"""

from __future__ import annotations

from public import public

from irx import astx
from irx.analysis.module_interfaces import ModuleKey
from irx.analysis.module_symbols import qualified_local_name
from irx.analysis.resolved_nodes import SemanticSymbol


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

"""
title: IRX-owned module namespace AST types.
"""

from __future__ import annotations

import astx

from astx.types import AnyType

from irx.typecheck import typechecked


@typechecked
class ModuleNamespaceType(AnyType):
    """
    title: Semantic-only module namespace type.
    summary: >-
      Represent one imported module namespace value during semantic analysis
      without modeling it as a runtime aggregate.
    attributes:
      module_key:
        type: str
      display_name:
        type: str | None
    """

    module_key: str
    display_name: str | None

    def __init__(
        self,
        module_key: str,
        *,
        display_name: str | None = None,
    ) -> None:
        """
        title: Initialize one module namespace type.
        parameters:
          module_key:
            type: str
          display_name:
            type: str | None
        """
        super().__init__()
        self.module_key = module_key
        self.display_name = display_name

    def __str__(self) -> str:
        """
        title: Render one module namespace type as text.
        returns:
          type: str
        """
        visible_name = self.display_name or self.module_key
        return f"ModuleNamespaceType[{visible_name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a module namespace type.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        visible_name = self.display_name or self.module_key
        key = f"MODULE-NAMESPACE[{visible_name}]"
        return self._prepare_struct(key, visible_name, simplified)


__all__ = ["ModuleNamespaceType"]

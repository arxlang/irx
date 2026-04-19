"""
title: IRX-owned namespace AST types.
"""

from __future__ import annotations

from enum import Enum

import astx

from astx.types import AnyType

from irx.typecheck import typechecked


@typechecked
class NamespaceKind(str, Enum):
    """
    title: Stable namespace categories.
    summary: >-
      Distinguish the semantic origin of one namespace-valued binding while
      keeping the user-facing expression model uniform.
    """

    MODULE = "module"
    PACKAGE = "package"
    LIBRARY = "library"


@typechecked
class NamespaceType(AnyType):
    """
    title: Semantic-only namespace type.
    summary: >-
      Represent one imported namespace value during semantic analysis and
      lowering without modeling it as a user-defined runtime aggregate.
    attributes:
      namespace_key:
        type: str
      namespace_kind:
        type: NamespaceKind
      display_name:
        type: str | None
    """

    namespace_key: str
    namespace_kind: NamespaceKind
    display_name: str | None

    def __init__(
        self,
        namespace_key: str,
        *,
        namespace_kind: NamespaceKind = NamespaceKind.MODULE,
        display_name: str | None = None,
    ) -> None:
        """
        title: Initialize one namespace type.
        parameters:
          namespace_key:
            type: str
          namespace_kind:
            type: NamespaceKind
          display_name:
            type: str | None
        """
        super().__init__()
        self.namespace_key = namespace_key
        self.namespace_kind = namespace_kind
        self.display_name = display_name

    @property
    def module_key(self) -> str:
        """
        title: Return the legacy module-key alias for this namespace.
        returns:
          type: str
        """
        return self.namespace_key

    def __str__(self) -> str:
        """
        title: Render one namespace type as text.
        returns:
          type: str
        """
        visible_name = self.display_name or self.namespace_key
        return f"NamespaceType[{self.namespace_kind.value}:{visible_name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a namespace type.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        visible_name = self.display_name or self.namespace_key
        key = f"NAMESPACE[{self.namespace_kind.value}:{visible_name}]"
        return self._prepare_struct(key, visible_name, simplified)


ModuleNamespaceType = NamespaceType

__all__ = [
    "ModuleNamespaceType",
    "NamespaceKind",
    "NamespaceType",
]

"""
title: IRX-owned struct AST nodes.
"""

from __future__ import annotations

import astx

from astx.types import AnyType

from irx.typecheck import typechecked


@typechecked
class StructType(AnyType):
    """
    title: Named struct type reference.
    attributes:
      name:
        type: str
      resolved_name:
        type: str | None
      module_key:
        type: str | None
      qualified_name:
        type: str | None
    """

    name: str
    resolved_name: str | None
    module_key: str | None
    qualified_name: str | None

    def __init__(
        self,
        name: str,
        *,
        resolved_name: str | None = None,
        module_key: str | None = None,
        qualified_name: str | None = None,
    ) -> None:
        """
        title: Initialize one named struct type reference.
        parameters:
          name:
            type: str
          resolved_name:
            type: str | None
          module_key:
            type: str | None
          qualified_name:
            type: str | None
        """
        super().__init__()
        self.name = name
        self.resolved_name = resolved_name
        self.module_key = module_key
        self.qualified_name = qualified_name

    def __str__(self) -> str:
        """
        title: Render one struct type reference as text.
        returns:
          type: str
        """
        return f"StructType[{self.name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a struct type reference.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"STRUCT-TYPE[{self.name}]"
        value = self.qualified_name or self.name
        return self._prepare_struct(key, value, simplified)


@typechecked
class FieldAccess(astx.DataType):
    """
    title: Field access expression.
    attributes:
      value:
        type: astx.AST
      field_name:
        type: str
      type_:
        type: AnyType
    """

    value: astx.AST
    field_name: str
    type_: AnyType

    def __init__(self, value: astx.AST, field_name: str) -> None:
        """
        title: Initialize one field access expression.
        parameters:
          value:
            type: astx.AST
          field_name:
            type: str
        """
        super().__init__()
        self.value = value
        self.field_name = field_name
        self.type_ = AnyType()

    def __str__(self) -> str:
        """
        title: Render one field access expression as text.
        returns:
          type: str
        """
        return f"FieldAccess[{self.field_name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a field access expression.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"FIELD-ACCESS[{self.field_name}]"
        value = self.value.get_struct(simplified)
        return self._prepare_struct(key, value, simplified)


__all__ = ["FieldAccess", "StructType"]

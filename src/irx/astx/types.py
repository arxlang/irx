"""
title: IRx-owned AST type helpers.
summary: >-
  Provide semantic-facing type nodes that extend the upstream ASTx type model
  without coupling them to template metadata helpers.
"""

from __future__ import annotations

from typing import Iterable, cast

import astx

from astx.types import AnyType

from irx.typecheck import typechecked


@typechecked
class UnionType(AnyType):
    """
    title: Finite compile-time union type domain.
    summary: >-
      Represent one union of concrete type references that semantic analysis
      may enumerate as a finite type domain.
    attributes:
      members:
        type: tuple[astx.DataType, Ellipsis]
      alias_name:
        type: str | None
    """

    members: tuple[astx.DataType, ...]
    alias_name: str | None

    def __init__(
        self,
        members: Iterable[astx.DataType],
        *,
        alias_name: str | None = None,
    ) -> None:
        """
        title: Initialize one finite union type.
        parameters:
          members:
            type: Iterable[astx.DataType]
          alias_name:
            type: str | None
        """
        super().__init__()
        self.members = tuple(members)
        self.alias_name = alias_name

    def __str__(self) -> str:
        """
        title: Render one finite union type as text.
        returns:
          type: str
        """
        if self.alias_name is not None:
            return self.alias_name
        return " | ".join(type(member).__name__ for member in self.members)

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a finite union type.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"UNION[{self.alias_name or id(self)}]"
        value = cast(
            astx.base.DataTypesStruct,
            [member.get_struct(simplified) for member in self.members],
        )
        return self._prepare_struct(key, value, simplified)


@typechecked
class TemplateTypeVar(AnyType):
    """
    title: Semantic-only template type variable.
    summary: >-
      Represent one unresolved template type parameter inside function
      signatures or local declared types before specialization.
    attributes:
      name:
        type: str
      bound:
        type: astx.DataType
    """

    name: str
    bound: astx.DataType

    def __init__(self, name: str, *, bound: astx.DataType) -> None:
        """
        title: Initialize one template type variable.
        parameters:
          name:
            type: str
          bound:
            type: astx.DataType
        """
        super().__init__()
        self.name = name
        self.bound = bound

    def __str__(self) -> str:
        """
        title: Render one template type variable as text.
        returns:
          type: str
        """
        return self.name

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a template type variable.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"TEMPLATE_TYPE_VAR[{self.name}]"
        value = cast(
            astx.base.DataTypesStruct,
            {
                "name": self.name,
                "bound": self.bound.get_struct(simplified),
            },
        )
        return self._prepare_struct(key, value, simplified)


@typechecked
class GeneratorType(AnyType):
    """
    title: First-class generator object type.
    summary: >-
      Represent one stateful generator value that yields values of one
      statically known element type when iterated.
    attributes:
      yield_type:
        type: astx.DataType
    """

    yield_type: astx.DataType

    def __init__(self, yield_type: astx.DataType) -> None:
        """
        title: Initialize one generator type.
        parameters:
          yield_type:
            type: astx.DataType
        """
        super().__init__()
        self.yield_type = yield_type

    def __str__(self) -> str:
        """
        title: Render one generator type as text.
        returns:
          type: str
        """
        return f"GeneratorType[{self.yield_type}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a generator type.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        return self._prepare_struct(
            "GENERATOR-TYPE",
            cast(
                astx.base.ReprStruct,
                {"yield_type": self.yield_type.get_struct(simplified)},
            ),
            simplified,
        )


__all__ = ["GeneratorType", "TemplateTypeVar", "UnionType"]

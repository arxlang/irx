"""
title: IRX-owned class AST nodes.
summary: >-
  Provide explicit class-type references and class-definition nodes that host
  compilers can use before higher-level surface syntax exists in Arx.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import astx

from astx.types import AnyType

from irx.typecheck import typechecked


@typechecked
class ClassType(AnyType):
    """
    title: Named class type reference.
    attributes:
      name:
        type: str
      resolved_name:
        type: str | None
      module_key:
        type: str | None
      qualified_name:
        type: str | None
      ancestor_qualified_names:
        type: tuple[str, Ellipsis]
    """

    name: str
    resolved_name: str | None
    module_key: str | None
    qualified_name: str | None
    ancestor_qualified_names: tuple[str, ...]

    def __init__(
        self,
        name: str,
        *,
        resolved_name: str | None = None,
        module_key: str | None = None,
        qualified_name: str | None = None,
        ancestor_qualified_names: tuple[str, ...] = (),
    ) -> None:
        """
        title: Initialize one named class type reference.
        parameters:
          name:
            type: str
          resolved_name:
            type: str | None
          module_key:
            type: str | None
          qualified_name:
            type: str | None
          ancestor_qualified_names:
            type: tuple[str, Ellipsis]
        """
        super().__init__()
        self.name = name
        self.resolved_name = resolved_name
        self.module_key = module_key
        self.qualified_name = qualified_name
        self.ancestor_qualified_names = ancestor_qualified_names

    def __str__(self) -> str:
        """
        title: Render one class type reference as text.
        returns:
          type: str
        """
        return f"ClassType[{self.name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a class type reference.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"CLASS-TYPE[{self.name}]"
        value = self.qualified_name or self.name
        return self._prepare_struct(key, value, simplified)


@typechecked
class ClassDefStmt(astx.StructDeclStmt):
    """
    title: Explicit class definition statement.
    summary: >-
      Represent one top-level class declaration with explicit base-class
      references and inherited struct-like member containers.
    attributes:
      bases:
        type: astx.ASTNodes[ClassType]
      kind:
        type: astx.ASTKind
    """

    bases: astx.ASTNodes[ClassType]
    kind: astx.ASTKind

    def __init__(
        self,
        name: str,
        *,
        bases: Iterable[ClassType] | astx.ASTNodes[ClassType] = (),
        attributes: (
            Iterable[astx.VariableDeclaration]
            | astx.ASTNodes[astx.VariableDeclaration]
        ) = (),
        decorators: Iterable[astx.Expr] | astx.ASTNodes[astx.Expr] = (),
        methods: Iterable[astx.FunctionDef]
        | astx.ASTNodes[astx.FunctionDef] = (),
        visibility: astx.VisibilityKind = astx.VisibilityKind.public,
        loc: astx.SourceLocation = astx.base.NO_SOURCE_LOCATION,
        parent: astx.ASTNodes[astx.AST] | None = None,
    ) -> None:
        """
        title: Initialize one class definition statement.
        parameters:
          name:
            type: str
          bases:
            type: Iterable[ClassType] | astx.ASTNodes[ClassType]
          attributes:
            type: >-
              Iterable[astx.VariableDeclaration] |
              astx.ASTNodes[astx.VariableDeclaration]
          decorators:
            type: Iterable[astx.Expr] | astx.ASTNodes[astx.Expr]
          methods:
            type: Iterable[astx.FunctionDef] | astx.ASTNodes[astx.FunctionDef]
          visibility:
            type: astx.VisibilityKind
          loc:
            type: astx.SourceLocation
          parent:
            type: astx.ASTNodes[astx.AST] | None
        """
        super().__init__(
            name=name,
            attributes=attributes,
            decorators=decorators,
            methods=methods,
            visibility=visibility,
            loc=loc,
            parent=parent,
        )
        if isinstance(bases, astx.ASTNodes):
            self.bases = bases
        else:
            self.bases = astx.ASTNodes[ClassType]()
            for base in bases:
                self.bases.append(base)
        self.kind = astx.ASTKind.StructDefStmtKind

    def __str__(self) -> str:
        """
        title: Render one class definition as text.
        returns:
          type: str
        """
        decorators_str = "".join(
            f"@{decorator}\n" for decorator in self.decorators
        )
        visibility_str = (
            self.visibility.name.lower()
            if self.visibility != astx.VisibilityKind.public
            else ""
        )
        bases_str = ""
        if self.bases:
            bases_str = "(" + ", ".join(base.name for base in self.bases) + ")"
        class_header = f"{visibility_str} class {self.name}{bases_str}".strip()
        member_lines = [str(attribute) for attribute in self.attributes] + [
            str(method) for method in self.methods
        ]
        members_str = "\n    ".join(member_lines)
        return f"{decorators_str}{class_header} {{\n    {members_str}\n}}"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a class definition.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        vis = dict(zip(("public", "private", "protected"), ("+", "-", "#")))
        key = f"CLASS-DEF[{vis[self.visibility.name]}{self.name}]"

        bases_dict: astx.base.ReprStruct = {}
        if self.bases:
            bases_dict = {"bases": self.bases.get_struct(simplified)}

        decorators_dict: astx.base.ReprStruct = {}
        if self.decorators:
            decorators_dict = {
                "decorators": self.decorators.get_struct(simplified)
            }

        attrs_dict: astx.base.ReprStruct = {}
        if self.attributes:
            attrs_dict = {"attributes": self.attributes.get_struct(simplified)}

        methods_dict: astx.base.ReprStruct = {}
        if self.methods:
            methods_dict = {"methods": self.methods.get_struct(simplified)}

        value = {
            **cast(dict[str, astx.base.ReprStruct], bases_dict),
            **cast(dict[str, astx.base.ReprStruct], decorators_dict),
            **cast(dict[str, astx.base.ReprStruct], attrs_dict),
            **cast(dict[str, astx.base.ReprStruct], methods_dict),
        }
        return self._prepare_struct(
            key,
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class ClassConstruct(astx.DataType):
    """
    title: Low-level default class construction expression.
    attributes:
      class_name:
        type: str
      type_:
        type: AnyType
    """

    class_name: str
    type_: AnyType

    def __init__(
        self,
        class_name: str,
    ) -> None:
        """
        title: Initialize one default class construction expression.
        parameters:
          class_name:
            type: str
        """
        super().__init__()
        self.class_name = class_name
        self.type_ = AnyType()

    def __str__(self) -> str:
        """
        title: Render one class construction expression as text.
        returns:
          type: str
        """
        return f"ClassConstruct[{self.class_name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for class construction.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"CLASS-CONSTRUCT[{self.class_name}]"
        return self._prepare_struct(key, self.class_name, simplified)


@typechecked
class MethodCall(astx.DataType):
    """
    title: Instance method call expression.
    attributes:
      receiver:
        type: astx.AST
      method_name:
        type: str
      args:
        type: tuple[astx.DataType, Ellipsis]
      type_:
        type: AnyType
    """

    receiver: astx.AST
    method_name: str
    args: tuple[astx.DataType, ...]
    type_: AnyType

    def __init__(
        self,
        receiver: astx.AST,
        method_name: str,
        args: Iterable[astx.DataType],
    ) -> None:
        """
        title: Initialize one instance method call expression.
        parameters:
          receiver:
            type: astx.AST
          method_name:
            type: str
          args:
            type: Iterable[astx.DataType]
        """
        super().__init__()
        self.receiver = receiver
        self.method_name = method_name
        self.args = tuple(args)
        self.type_ = AnyType()

    def __str__(self) -> str:
        """
        title: Render one instance method call expression as text.
        returns:
          type: str
        """
        return f"MethodCall[{self.method_name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for an instance method call.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"METHOD-CALL[{self.method_name}]"
        arg_nodes = astx.ASTNodes[astx.DataType]("args")
        for arg in self.args:
            arg_nodes.append(arg)
        value = {
            "receiver": self.receiver.get_struct(simplified),
            "args": arg_nodes.get_struct(simplified),
        }
        return self._prepare_struct(
            key,
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class StaticMethodCall(astx.DataType):
    """
    title: Class-scoped static method call expression.
    attributes:
      class_name:
        type: str
      method_name:
        type: str
      args:
        type: tuple[astx.DataType, Ellipsis]
      type_:
        type: AnyType
    """

    class_name: str
    method_name: str
    args: tuple[astx.DataType, ...]
    type_: AnyType

    def __init__(
        self,
        class_name: str,
        method_name: str,
        args: Iterable[astx.DataType],
    ) -> None:
        """
        title: Initialize one static method call expression.
        parameters:
          class_name:
            type: str
          method_name:
            type: str
          args:
            type: Iterable[astx.DataType]
        """
        super().__init__()
        self.class_name = class_name
        self.method_name = method_name
        self.args = tuple(args)
        self.type_ = AnyType()

    def __str__(self) -> str:
        """
        title: Render one static method call expression as text.
        returns:
          type: str
        """
        return f"StaticMethodCall[{self.class_name}.{self.method_name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a static method call.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"STATIC-METHOD-CALL[{self.class_name}.{self.method_name}]"
        arg_nodes = astx.ASTNodes[astx.DataType]("args")
        for arg in self.args:
            arg_nodes.append(arg)
        value = {
            "args": arg_nodes.get_struct(simplified),
        }
        return self._prepare_struct(
            key,
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class StaticFieldAccess(astx.DataType):
    """
    title: Class-scoped static field access expression.
    attributes:
      class_name:
        type: str
      field_name:
        type: str
      type_:
        type: AnyType
    """

    class_name: str
    field_name: str
    type_: AnyType

    def __init__(
        self,
        class_name: str,
        field_name: str,
    ) -> None:
        """
        title: Initialize one static field access expression.
        parameters:
          class_name:
            type: str
          field_name:
            type: str
        """
        super().__init__()
        self.class_name = class_name
        self.field_name = field_name
        self.type_ = AnyType()

    def __str__(self) -> str:
        """
        title: Render one static field access expression as text.
        returns:
          type: str
        """
        return f"StaticFieldAccess[{self.class_name}.{self.field_name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a static field access.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"STATIC-FIELD-ACCESS[{self.class_name}.{self.field_name}]"
        return self._prepare_struct(
            key,
            f"{self.class_name}.{self.field_name}",
            simplified,
        )


@typechecked
class BaseFieldAccess(astx.DataType):
    """
    title: Explicit base-qualified instance field access expression.
    attributes:
      receiver:
        type: astx.AST
      base_class_name:
        type: str
      field_name:
        type: str
      type_:
        type: AnyType
    """

    receiver: astx.AST
    base_class_name: str
    field_name: str
    type_: AnyType

    def __init__(
        self,
        receiver: astx.AST,
        base_class_name: str,
        field_name: str,
    ) -> None:
        """
        title: Initialize one base-qualified field access expression.
        parameters:
          receiver:
            type: astx.AST
          base_class_name:
            type: str
          field_name:
            type: str
        """
        super().__init__()
        self.receiver = receiver
        self.base_class_name = base_class_name
        self.field_name = field_name
        self.type_ = AnyType()

    def __str__(self) -> str:
        """
        title: Render one base-qualified field access expression as text.
        returns:
          type: str
        """
        return f"BaseFieldAccess[{self.base_class_name}.{self.field_name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a base-qualified field access.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"BASE-FIELD-ACCESS[{self.base_class_name}.{self.field_name}]"
        value = {
            "receiver": self.receiver.get_struct(simplified),
        }
        return self._prepare_struct(
            key,
            cast(astx.base.ReprStruct, value),
            simplified,
        )


@typechecked
class BaseMethodCall(astx.DataType):
    """
    title: Explicit base-qualified instance method call expression.
    attributes:
      receiver:
        type: astx.AST
      base_class_name:
        type: str
      method_name:
        type: str
      args:
        type: tuple[astx.DataType, Ellipsis]
      type_:
        type: AnyType
    """

    receiver: astx.AST
    base_class_name: str
    method_name: str
    args: tuple[astx.DataType, ...]
    type_: AnyType

    def __init__(
        self,
        receiver: astx.AST,
        base_class_name: str,
        method_name: str,
        args: Iterable[astx.DataType],
    ) -> None:
        """
        title: Initialize one base-qualified method call expression.
        parameters:
          receiver:
            type: astx.AST
          base_class_name:
            type: str
          method_name:
            type: str
          args:
            type: Iterable[astx.DataType]
        """
        super().__init__()
        self.receiver = receiver
        self.base_class_name = base_class_name
        self.method_name = method_name
        self.args = tuple(args)
        self.type_ = AnyType()

    def __str__(self) -> str:
        """
        title: Render one base-qualified method call expression as text.
        returns:
          type: str
        """
        return f"BaseMethodCall[{self.base_class_name}.{self.method_name}]"

    def get_struct(self, simplified: bool = False) -> astx.base.ReprStruct:
        """
        title: Build one repr structure for a base-qualified method call.
        parameters:
          simplified:
            type: bool
        returns:
          type: astx.base.ReprStruct
        """
        key = f"BASE-METHOD-CALL[{self.base_class_name}.{self.method_name}]"
        arg_nodes = astx.ASTNodes[astx.DataType]("args")
        for arg in self.args:
            arg_nodes.append(arg)
        value = {
            "receiver": self.receiver.get_struct(simplified),
            "args": arg_nodes.get_struct(simplified),
        }
        return self._prepare_struct(
            key,
            cast(astx.base.ReprStruct, value),
            simplified,
        )


__all__ = [
    "BaseFieldAccess",
    "BaseMethodCall",
    "ClassConstruct",
    "ClassDefStmt",
    "ClassType",
    "MethodCall",
    "StaticFieldAccess",
    "StaticMethodCall",
]

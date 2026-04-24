"""
title: IRx-owned context-manager AST nodes.
summary: >-
  Provide a small facade node for Python-like ``with`` statements until
  upstream ASTx exposes a canonical context-manager construct.
"""

from __future__ import annotations

from typing import cast

import astx

from astx.base import NO_SOURCE_LOCATION, ReprStruct, SourceLocation

from irx.typecheck import typechecked


@typechecked
class WithStmt(astx.StatementType):
    """
    title: Internal context-manager statement node.
    summary: >-
      Model source forms equivalent to ``with manager as target: body`` while
      keeping the manager expression, optional target, and body as regular ASTx
      child nodes.
    attributes:
      manager:
        type: astx.AST
      body:
        type: astx.Block
      target:
        type: astx.AST | None
    """

    manager: astx.AST
    body: astx.Block
    target: astx.AST | None

    def __init__(
        self,
        manager: astx.AST,
        body: astx.Block,
        target: astx.AST | None = None,
        loc: SourceLocation = NO_SOURCE_LOCATION,
        parent: astx.ASTNodes | None = None,
    ) -> None:
        """
        title: Initialize one with statement.
        parameters:
          manager:
            type: astx.AST
          body:
            type: astx.Block
          target:
            type: astx.AST | None
          loc:
            type: SourceLocation
          parent:
            type: astx.ASTNodes | None
        """
        super().__init__(loc=loc, parent=parent)
        self.manager = manager
        self.body = body
        self.target = target

    def __str__(self) -> str:
        """
        title: Return a compact display string.
        returns:
          type: str
        """
        return "WithStmt"

    def get_struct(self, simplified: bool = False) -> ReprStruct:
        """
        title: Return the structured representation.
        parameters:
          simplified:
            type: bool
        returns:
          type: ReprStruct
        """
        key = f"WITH-STMT[{id(self)}]" if simplified else "WITH-STMT"
        target = (
            {"target": self.target.get_struct(simplified)}
            if self.target is not None
            else {}
        )
        value = cast(
            ReprStruct,
            {
                "manager": self.manager.get_struct(simplified),
                **target,
                "body": self.body.get_struct(simplified),
            },
        )
        return self._prepare_struct(key, value, simplified)


__all__ = ["WithStmt"]

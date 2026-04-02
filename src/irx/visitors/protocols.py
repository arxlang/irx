"""
title: Shared typing protocols for IRx visitors.
"""

from __future__ import annotations

from typing import Protocol

from irx import astx


class BaseVisitorProtocol(Protocol):
    """
    title: Minimal typing contract shared by IRx visitors.
    """

    def visit(self, _node: astx.AST) -> None: ...

    def visit_child(self, _node: astx.AST) -> None: ...

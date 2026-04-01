"""
title: Shared typing protocols for IRx visitors.
"""

from __future__ import annotations

from typing import Protocol

import astx


class BaseVisitorProtocol(Protocol):
    """
    title: Minimal typing contract shared by IRx visitors.
    """

    def visit(self, _node: astx.AST) -> None: ...

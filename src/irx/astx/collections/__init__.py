"""
title: IRX-owned collection helper AST nodes.
summary: >-
  Group IRX-specific collection-oriented AST helpers behind one stable package
  boundary while re-exporting the current public node names.
"""

from __future__ import annotations

from irx.astx.collections.common import (
    CollectionContains as CollectionContains,
)
from irx.astx.collections.common import CollectionCount as CollectionCount
from irx.astx.collections.common import CollectionIndex as CollectionIndex
from irx.astx.collections.common import (
    CollectionIsEmpty as CollectionIsEmpty,
)
from irx.astx.collections.common import CollectionLength as CollectionLength
from irx.astx.collections.list import ListAppend as ListAppend
from irx.astx.collections.list import ListCreate as ListCreate
from irx.astx.collections.list import ListIndex as ListIndex
from irx.astx.collections.list import ListLength as ListLength

__all__ = (
    "CollectionContains",
    "CollectionCount",
    "CollectionIndex",
    "CollectionIsEmpty",
    "CollectionLength",
    "ListAppend",
    "ListCreate",
    "ListIndex",
    "ListLength",
)

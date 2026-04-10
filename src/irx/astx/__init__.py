"""
title: IRX-owned AST facade built on top of ASTx.
"""

from __future__ import annotations

from typing import Any

import astx as _upstream_astx

from irx.astx.arrow import ArrowInt32ArrayLength as ArrowInt32ArrayLength
from irx.astx.binary_op import (
    SPECIALIZED_BINARY_OP_EXTRA as SPECIALIZED_BINARY_OP_EXTRA,
)
from irx.astx.binary_op import (
    AddBinOp as AddBinOp,
)
from irx.astx.binary_op import (
    AssignmentBinOp as AssignmentBinOp,
)
from irx.astx.binary_op import (
    BitAndBinOp as BitAndBinOp,
)
from irx.astx.binary_op import (
    BitOrBinOp as BitOrBinOp,
)
from irx.astx.binary_op import (
    BitXorBinOp as BitXorBinOp,
)
from irx.astx.binary_op import (
    DivBinOp as DivBinOp,
)
from irx.astx.binary_op import (
    EqBinOp as EqBinOp,
)
from irx.astx.binary_op import (
    GeBinOp as GeBinOp,
)
from irx.astx.binary_op import (
    GtBinOp as GtBinOp,
)
from irx.astx.binary_op import (
    LeBinOp as LeBinOp,
)
from irx.astx.binary_op import (
    LogicalAndBinOp as LogicalAndBinOp,
)
from irx.astx.binary_op import (
    LogicalOrBinOp as LogicalOrBinOp,
)
from irx.astx.binary_op import (
    LtBinOp as LtBinOp,
)
from irx.astx.binary_op import (
    ModBinOp as ModBinOp,
)
from irx.astx.binary_op import (
    MulBinOp as MulBinOp,
)
from irx.astx.binary_op import (
    NeBinOp as NeBinOp,
)
from irx.astx.binary_op import (
    SubBinOp as SubBinOp,
)
from irx.astx.binary_op import (
    binary_op_type_for_opcode as binary_op_type_for_opcode,
)
from irx.astx.binary_op import (
    specialize_binary_op as specialize_binary_op,
)
from irx.astx.buffer import BufferOwnerType as BufferOwnerType
from irx.astx.buffer import BufferViewDescriptor as BufferViewDescriptor
from irx.astx.buffer import BufferViewRelease as BufferViewRelease
from irx.astx.buffer import BufferViewRetain as BufferViewRetain
from irx.astx.buffer import BufferViewType as BufferViewType
from irx.astx.buffer import BufferViewWrite as BufferViewWrite
from irx.astx.structs import FieldAccess as FieldAccess
from irx.astx.structs import StructType as StructType
from irx.astx.system import Cast as Cast
from irx.astx.system import PrintExpr as PrintExpr
from irx.typecheck import typechecked

__all__ = (
    "SPECIALIZED_BINARY_OP_EXTRA",
    "AddBinOp",
    "ArrowInt32ArrayLength",
    "AssignmentBinOp",
    "BitAndBinOp",
    "BitOrBinOp",
    "BitXorBinOp",
    "BufferOwnerType",
    "BufferViewDescriptor",
    "BufferViewRelease",
    "BufferViewRetain",
    "BufferViewType",
    "BufferViewWrite",
    "Cast",
    "DivBinOp",
    "EqBinOp",
    "FieldAccess",
    "GeBinOp",
    "GtBinOp",
    "LeBinOp",
    "LogicalAndBinOp",
    "LogicalOrBinOp",
    "LtBinOp",
    "ModBinOp",
    "MulBinOp",
    "NeBinOp",
    "PrintExpr",
    "StructType",
    "SubBinOp",
    "binary_op_type_for_opcode",
    "specialize_binary_op",
)


def __getattr__(name: str) -> Any:
    """
    title: Forward unknown attributes to the upstream astx package.
    parameters:
      name:
        type: str
    returns:
      type: Any
    """
    return getattr(_upstream_astx, name)


def __dir__() -> list[str]:
    """
    title: Return the visible attributes from the facade module.
    returns:
      type: list[str]
    """
    return sorted(set(dir(_upstream_astx)) | set(__all__))


__getattr__ = typechecked(__getattr__)
__dir__ = typechecked(__dir__)

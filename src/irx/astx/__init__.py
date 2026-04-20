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
from irx.astx.buffer import BufferViewIndex as BufferViewIndex
from irx.astx.buffer import BufferViewRelease as BufferViewRelease
from irx.astx.buffer import BufferViewRetain as BufferViewRetain
from irx.astx.buffer import BufferViewStore as BufferViewStore
from irx.astx.buffer import BufferViewType as BufferViewType
from irx.astx.buffer import BufferViewWrite as BufferViewWrite
from irx.astx.classes import BaseFieldAccess as BaseFieldAccess
from irx.astx.classes import BaseMethodCall as BaseMethodCall
from irx.astx.classes import ClassConstruct as ClassConstruct
from irx.astx.classes import ClassDefStmt as ClassDefStmt
from irx.astx.classes import ClassType as ClassType
from irx.astx.classes import MethodCall as MethodCall
from irx.astx.classes import StaticFieldAccess as StaticFieldAccess
from irx.astx.classes import StaticMethodCall as StaticMethodCall
from irx.astx.ffi import OpaqueHandleType as OpaqueHandleType
from irx.astx.ffi import PointerType as PointerType
from irx.astx.modules import ModuleNamespaceType as ModuleNamespaceType
from irx.astx.modules import NamespaceKind as NamespaceKind
from irx.astx.modules import NamespaceType as NamespaceType
from irx.astx.structs import FieldAccess as FieldAccess
from irx.astx.structs import StructType as StructType
from irx.astx.system import AssertStmt as AssertStmt
from irx.astx.system import Cast as Cast
from irx.astx.system import PrintExpr as PrintExpr
from irx.astx.templates import TemplateParam as TemplateParam
from irx.astx.templates import TemplateTypeVar as TemplateTypeVar
from irx.astx.templates import UnionType as UnionType
from irx.astx.templates import (
    add_generated_template_node as add_generated_template_node,
)
from irx.astx.templates import (
    generated_template_nodes as generated_template_nodes,
)
from irx.astx.templates import get_template_args as get_template_args
from irx.astx.templates import get_template_params as get_template_params
from irx.astx.templates import is_template_node as is_template_node
from irx.astx.templates import (
    is_template_specialization as is_template_specialization,
)
from irx.astx.templates import (
    mark_template_specialization as mark_template_specialization,
)
from irx.astx.templates import set_template_args as set_template_args
from irx.astx.templates import set_template_params as set_template_params
from irx.astx.templates import (
    template_specialization_name as template_specialization_name,
)
from irx.typecheck import typechecked

__all__ = (
    "SPECIALIZED_BINARY_OP_EXTRA",
    "AddBinOp",
    "ArrowInt32ArrayLength",
    "AssertStmt",
    "AssignmentBinOp",
    "BaseFieldAccess",
    "BaseMethodCall",
    "BitAndBinOp",
    "BitOrBinOp",
    "BitXorBinOp",
    "BufferOwnerType",
    "BufferViewDescriptor",
    "BufferViewIndex",
    "BufferViewRelease",
    "BufferViewRetain",
    "BufferViewStore",
    "BufferViewType",
    "BufferViewWrite",
    "Cast",
    "ClassConstruct",
    "ClassDefStmt",
    "ClassType",
    "DivBinOp",
    "EqBinOp",
    "FieldAccess",
    "GeBinOp",
    "GtBinOp",
    "LeBinOp",
    "LogicalAndBinOp",
    "LogicalOrBinOp",
    "LtBinOp",
    "MethodCall",
    "ModBinOp",
    "ModuleNamespaceType",
    "MulBinOp",
    "NamespaceKind",
    "NamespaceType",
    "NeBinOp",
    "OpaqueHandleType",
    "PointerType",
    "PrintExpr",
    "StaticFieldAccess",
    "StaticMethodCall",
    "StructType",
    "SubBinOp",
    "TemplateParam",
    "TemplateTypeVar",
    "UnionType",
    "add_generated_template_node",
    "binary_op_type_for_opcode",
    "generated_template_nodes",
    "get_template_args",
    "get_template_params",
    "is_template_node",
    "is_template_specialization",
    "mark_template_specialization",
    "set_template_args",
    "set_template_params",
    "specialize_binary_op",
    "template_specialization_name",
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

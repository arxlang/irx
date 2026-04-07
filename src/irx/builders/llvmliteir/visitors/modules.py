# mypy: disable-error-code=no-redef

"""
title: Module-level visitor mixins for llvmliteir.
"""

from llvmlite import ir

from irx import astx
from irx.builders.llvmliteir.core import (
    VisitorCore,
    semantic_struct_key,
    semantic_struct_name,
)
from irx.builders.llvmliteir.protocols import VisitorMixinBase


class ModuleVisitorMixin(VisitorMixinBase):
    @VisitorCore.visit.dispatch
    def visit(self, node: astx.Module) -> None:
        """
        title: Visit Module nodes.
        parameters:
          node:
            type: astx.Module
        """
        self._translate_modules([node])

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.StructDefStmt) -> None:
        """
        title: Visit StructDefStmt nodes.
        parameters:
          node:
            type: astx.StructDefStmt
        """
        struct_key = semantic_struct_key(node, node.name)
        existing = self.llvm_structs_by_qualified_name.get(struct_key)
        if existing is not None:
            self.struct_types[struct_key] = existing
            return

        llvm_name = semantic_struct_name(node, node.name)
        struct_type = self._llvm.module.context.get_identified_type(llvm_name)
        if not struct_type.is_opaque:
            raise ValueError(f"Struct '{node.name}' already defined.")

        field_types: list[ir.Type] = []
        for attr in node.attributes:
            type_str = attr.type_.__class__.__name__.lower()
            field_types.append(self._llvm.get_data_type(type_str))

        struct_type.set_body(*field_types)
        self.struct_types[struct_key] = struct_type
        self.llvm_structs_by_qualified_name[struct_key] = struct_type

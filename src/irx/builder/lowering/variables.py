# mypy: disable-error-code=no-redef

"""
title: Variable visitor mixins for llvmliteir.
"""

from typing import Any, cast

from llvmlite import ir

from irx import astx
from irx.builder.core import (
    VisitorCore,
    semantic_assignment_key,
    semantic_symbol_key,
)
from irx.builder.diagnostics import raise_lowering_error
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class VariableVisitorMixin(VisitorMixinBase):
    @VisitorCore.visit.dispatch
    def visit(self, expr: astx.VariableAssignment) -> None:
        """
        title: Visit VariableAssignment nodes.
        parameters:
          expr:
            type: astx.VariableAssignment
        """
        var_name = expr.name
        var_key = semantic_assignment_key(expr, var_name)

        if var_key in self.const_vars:
            raise Exception(
                f"Cannot assign to '{var_name}': declared as constant"
            )

        self.visit_child(expr.value)
        llvm_value = safe_pop(self.result_stack)
        if llvm_value is None:
            raise Exception("codegen: Invalid value in VariableAssignment.")
        llvm_value = self._cast_ast_value(
            llvm_value,
            source_type=self._resolved_ast_type(expr.value),
            target_type=self._resolved_ast_type(expr),
        )

        llvm_var = self.named_values.get(var_key)
        if not llvm_var:
            raise Exception(
                f"Identifier '{var_name}' not found in the named values."
            )

        self._llvm.ir_builder.store(llvm_value, llvm_var)
        self.result_stack.append(llvm_value)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.Identifier) -> None:
        """
        title: Visit Identifier nodes.
        parameters:
          node:
            type: astx.Identifier
        """
        symbol_key = semantic_symbol_key(node, node.name)
        expr_var = self.named_values.get(symbol_key)
        if expr_var:
            result = self._llvm.ir_builder.load(expr_var, node.name)
            self.result_stack.append(result)
            return

        namespace_value = self._namespace_value(node)
        if namespace_value is not None:
            self.result_stack.append(namespace_value)
            return

        raise Exception(f"Unknown variable name: {node.name}")

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.FieldAccess) -> None:
        """
        title: Visit FieldAccess nodes.
        parameters:
          node:
            type: astx.FieldAccess
        """
        namespace_value = self._namespace_value(node)
        if namespace_value is not None:
            self.visit_child(node.value)
            _ = safe_pop(self.result_stack)
            self.result_stack.append(namespace_value)
            return

        resolved_module_member_access = getattr(
            getattr(node, "semantic", None),
            "resolved_module_member_access",
            None,
        )
        if resolved_module_member_access is not None:
            raise_lowering_error(
                "module namespace member references are lowerable only when "
                "they resolve to nested namespaces or are used in call "
                "position",
                code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
                node=node,
                hint=(
                    "use namespace.member(...) for callable members, or "
                    "return/bind a namespace-valued member instead"
                ),
            )

        if isinstance(node.value, astx.FieldAccess):
            parent_ptr = self._field_address(node.value)
            parent_value = self._llvm.ir_builder.load(
                parent_ptr,
                f"{node.field_name}_parent",
            )
            resolved = getattr(
                getattr(node, "semantic", None),
                "resolved_field_access",
                None,
            )
            if resolved is None:
                raise Exception("codegen: unresolved field access.")
            result = self._llvm.ir_builder.extract_value(
                parent_value,
                resolved.field.index,
                node.field_name,
            )
            self.result_stack.append(result)
            return

        field_ptr = self._field_address(node)
        result = self._llvm.ir_builder.load(field_ptr, node.field_name)
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.BaseFieldAccess) -> None:
        """
        title: Visit BaseFieldAccess nodes.
        parameters:
          node:
            type: astx.BaseFieldAccess
        """
        field_ptr = self._base_class_field_address(node)
        result = self._llvm.ir_builder.load(field_ptr, node.field_name)
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.StaticFieldAccess) -> None:
        """
        title: Visit StaticFieldAccess nodes.
        parameters:
          node:
            type: astx.StaticFieldAccess
        """
        field_ptr = self._static_class_field_address(node)
        result = self._llvm.ir_builder.load(field_ptr, node.field_name)
        self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.VariableDeclaration) -> None:
        """
        title: Visit VariableDeclaration nodes.
        parameters:
          node:
            type: astx.VariableDeclaration
        """
        symbol_key = semantic_symbol_key(node, node.name)
        existing_storage = self.named_values.get(symbol_key)
        if existing_storage and self._current_generator_frame_ptr is None:
            raise Exception(f"Identifier already declared: {node.name}")

        type_str = node.type_.__class__.__name__.lower()
        llvm_type = self._llvm_type_for_ast_type(node.type_)
        if llvm_type is None:
            raise Exception(
                f"codegen: Unknown LLVM type for variable '{node.name}'."
            )
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self.visit_child(node.value)
            init_val = safe_pop(self.result_stack)
            if init_val is None:
                raise Exception("Initializer code generation failed.")
            init_val = self._cast_ast_value(
                init_val,
                source_type=self._resolved_ast_type(node.value),
                target_type=node.type_,
            )

            if type_str == "string":
                alloca = self.create_entry_block_alloca(
                    node.name, "stringascii"
                )
            elif existing_storage is not None:
                alloca = existing_storage
            else:
                alloca = self.create_entry_block_alloca(node.name, llvm_type)
            self._llvm.ir_builder.store(init_val, alloca)
        else:
            if type_str == "string":
                empty_str_type = ir.ArrayType(self._llvm.INT8_TYPE, 1)
                empty_str_global = ir.GlobalVariable(
                    self._llvm.module,
                    empty_str_type,
                    name=f"empty_str_{node.name}",
                )
                empty_str_global.linkage = "internal"
                empty_str_global.global_constant = True
                empty_str_global.initializer = ir.Constant(
                    empty_str_type, bytearray(b"\0")
                )
                init_val = self._llvm.ir_builder.gep(
                    empty_str_global,
                    [
                        ir.Constant(ir.IntType(32), 0),
                        ir.Constant(ir.IntType(32), 0),
                    ],
                    inbounds=True,
                )
                alloca = (
                    existing_storage
                    if existing_storage is not None
                    else self.create_entry_block_alloca(
                        node.name, "stringascii"
                    )
                )
            elif isinstance(node.type_, astx.StructType):
                init_val = ir.Constant(llvm_type, None)
                alloca = (
                    existing_storage
                    if existing_storage is not None
                    else self.create_entry_block_alloca(node.name, llvm_type)
                )
            elif isinstance(node.type_, astx.ListType):
                init_val = cast(
                    ir.Constant,
                    cast(Any, self)._empty_list_value_for_type(node.type_),
                )
                alloca = (
                    existing_storage
                    if existing_storage is not None
                    else self.create_entry_block_alloca(node.name, llvm_type)
                )
            elif isinstance(node.type_, astx.ClassType):
                init_val = ir.Constant(llvm_type, None)
                alloca = (
                    existing_storage
                    if existing_storage is not None
                    else self.create_entry_block_alloca(node.name, llvm_type)
                )
            elif isinstance(node.type_, astx.GeneratorType):
                init_val = ir.Constant(llvm_type, None)
                alloca = (
                    existing_storage
                    if existing_storage is not None
                    else self.create_entry_block_alloca(node.name, llvm_type)
                )
            elif "float" in type_str:
                init_val = ir.Constant(self._llvm.get_data_type(type_str), 0.0)
                alloca = (
                    existing_storage
                    if existing_storage is not None
                    else self.create_entry_block_alloca(node.name, llvm_type)
                )
            else:
                init_val = ir.Constant(self._llvm.get_data_type(type_str), 0)
                alloca = (
                    existing_storage
                    if existing_storage is not None
                    else self.create_entry_block_alloca(node.name, llvm_type)
                )

            self._llvm.ir_builder.store(init_val, alloca)

        if node.mutability == astx.MutabilityKind.constant:
            self.const_vars.add(symbol_key)
        self.named_values[symbol_key] = alloca

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.InlineVariableDeclaration) -> None:
        """
        title: Visit InlineVariableDeclaration nodes.
        parameters:
          node:
            type: astx.InlineVariableDeclaration
        """
        symbol_key = semantic_symbol_key(node, node.name)
        existing_storage = self.named_values.get(symbol_key)
        if existing_storage and self._current_generator_frame_ptr is None:
            raise Exception(f"Identifier already declared: {node.name}")

        type_str = node.type_.__class__.__name__.lower()
        llvm_type = self._llvm_type_for_ast_type(node.type_)
        if llvm_type is None:
            raise Exception(
                "codegen: Unknown LLVM type for inline variable "
                f"'{node.name}'."
            )
        if node.value is not None:
            self.visit_child(node.value)
            init_val = safe_pop(self.result_stack)
            if init_val is None:
                raise Exception("Initializer code generation failed.")
            init_val = self._cast_ast_value(
                init_val,
                source_type=self._resolved_ast_type(node.value),
                target_type=node.type_,
            )
        elif isinstance(node.type_, astx.StructType):
            init_val = ir.Constant(llvm_type, None)
        elif isinstance(node.type_, astx.ListType):
            init_val = cast(
                ir.Constant,
                cast(Any, self)._empty_list_value_for_type(node.type_),
            )
        elif isinstance(node.type_, astx.ClassType):
            init_val = ir.Constant(llvm_type, None)
        elif isinstance(node.type_, astx.GeneratorType):
            init_val = ir.Constant(llvm_type, None)
        elif "float" in type_str:
            init_val = ir.Constant(self._llvm.get_data_type(type_str), 0.0)
        else:
            init_val = ir.Constant(self._llvm.get_data_type(type_str), 0)

        if type_str == "string":
            alloca = self.create_entry_block_alloca(node.name, "stringascii")
        elif existing_storage is not None:
            alloca = existing_storage
        else:
            alloca = self.create_entry_block_alloca(node.name, llvm_type)

        self._llvm.ir_builder.store(init_val, alloca)
        if node.mutability == astx.MutabilityKind.constant:
            self.const_vars.add(symbol_key)
        self.named_values[symbol_key] = alloca
        self.result_stack.append(init_val)

# mypy: disable-error-code=no-redef

"""
title: Declaration-oriented semantic visitors.
summary: >-
  Handle modules, functions, structs, and lexical declarations while delegating
  semantic entity creation and registration to shared infrastructure.
"""

from __future__ import annotations

from dataclasses import replace

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.module_symbols import qualified_local_name
from irx.analysis.resolved_nodes import (
    SemanticFunction,
    SemanticStruct,
    SemanticStructField,
)
from irx.analysis.types import clone_type
from irx.analysis.validation import validate_assignment
from irx.typecheck import typechecked

DIRECT_STRUCT_CYCLE_LENGTH = 2


@typechecked
class DeclarationVisitorMixin(SemanticVisitorMixinBase):
    def _synchronize_function_signature(
        self,
        function: SemanticFunction,
        prototype: astx.FunctionPrototype,
        *,
        definition: astx.FunctionDef | None = None,
    ) -> SemanticFunction:
        """
        title: Synchronize one semantic function with resolved AST types.
        parameters:
          function:
            type: SemanticFunction
          prototype:
            type: astx.FunctionPrototype
          definition:
            type: astx.FunctionDef | None
        returns:
          type: SemanticFunction
        """
        signature = self.registry.normalize_function_signature(
            prototype,
            definition=definition,
        )
        if (
            function.prototype is not prototype
            and definition is None
            and not self.registry.signatures_match(
                function.signature, signature
            )
        ):
            return function
        if (
            definition is not None
            and function.definition is not None
            and function.definition is not definition
            and not self.registry.signatures_match(
                function.signature, signature
            )
        ):
            return function

        updated = replace(
            function,
            return_type=clone_type(signature.return_type),
            args=tuple(
                replace(
                    arg_symbol,
                    name=arg_node.name,
                    type_=clone_type(arg_node.type_),
                    qualified_name=qualified_local_name(
                        function.module_key,
                        arg_symbol.kind,
                        arg_node.name,
                        arg_symbol.symbol_id,
                    ),
                )
                for arg_node, arg_symbol in zip(
                    prototype.args.nodes,
                    function.args,
                )
            ),
            signature=signature,
            prototype=prototype,
            definition=(
                definition if definition is not None else function.definition
            ),
        )
        self.context.register_function(updated)
        return updated

    def _resolve_struct_fields(
        self,
        struct: SemanticStruct,
    ) -> SemanticStruct:
        """
        title: Resolve one struct's ordered field metadata.
        parameters:
          struct:
            type: SemanticStruct
        returns:
          type: SemanticStruct
        """
        seen: set[str] = set()
        fields: list[SemanticStructField] = []

        if len(list(struct.declaration.attributes)) == 0:
            self.context.diagnostics.add(
                f"Struct '{struct.name}' must declare at least one field",
                node=struct.declaration,
            )

        for index, attr in enumerate(struct.declaration.attributes):
            if attr.name in seen:
                self.context.diagnostics.add(
                    f"Struct field '{attr.name}' already defined.",
                    node=attr,
                )
            seen.add(attr.name)
            self._resolve_declared_type(
                attr.type_,
                node=attr,
                unknown_message="Unknown field type '{name}'",
            )
            fields.append(
                SemanticStructField(
                    name=attr.name,
                    index=index,
                    type_=clone_type(attr.type_),
                    declaration=attr,
                )
            )

        updated = replace(
            struct,
            fields=tuple(fields),
            field_indices={field.name: field.index for field in fields},
        )
        self.context.register_struct(updated)
        self.bindings.bind_struct(
            updated.name,
            updated,
            node=updated.declaration,
        )
        self._set_struct(updated.declaration, updated)
        return updated

    def _find_struct_cycle(
        self,
        root: SemanticStruct,
        current: SemanticStruct,
        path: tuple[SemanticStruct, ...],
    ) -> tuple[SemanticStruct, ...] | None:
        """
        title: Find one by-value recursive struct cycle.
        parameters:
          root:
            type: SemanticStruct
          current:
            type: SemanticStruct
          path:
            type: tuple[SemanticStruct, Ellipsis]
        returns:
          type: tuple[SemanticStruct, Ellipsis] | None
        """
        seen = {struct.qualified_name for struct in path}
        for attr in current.declaration.attributes:
            field_struct = self._resolve_struct_from_type(
                attr.type_,
                node=attr,
                unknown_message="Unknown field type '{name}'",
            )
            if field_struct is None:
                continue
            if field_struct.qualified_name == root.qualified_name:
                return (*path, field_struct)
            if field_struct.qualified_name in seen:
                continue
            cycle = self._find_struct_cycle(
                root,
                field_struct,
                (*path, field_struct),
            )
            if cycle is not None:
                return cycle
        return None

    def _validate_struct_cycles(self, struct: SemanticStruct) -> None:
        """
        title: Reject by-value recursive struct layouts.
        parameters:
          struct:
            type: SemanticStruct
        """
        cycle = self._find_struct_cycle(struct, struct, (struct,))
        if cycle is None:
            return

        if len(cycle) == DIRECT_STRUCT_CYCLE_LENGTH:
            self.context.diagnostics.add(
                (
                    "direct by-value recursive struct "
                    f"'{struct.name}' is forbidden"
                ),
                node=struct.declaration,
            )
            return

        cycle_names = " -> ".join(item.name for item in cycle)
        self.context.diagnostics.add(
            f"mutual by-value recursive structs are forbidden: {cycle_names}",
            node=struct.declaration,
        )

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, module: astx.Module) -> None:
        """
        title: Visit Module nodes.
        parameters:
          module:
            type: astx.Module
        """
        with self.context.in_module(module.name):
            self._visit_module(module, predeclared=False)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, block: astx.Block) -> None:
        """
        title: Visit Block nodes.
        parameters:
          block:
            type: astx.Block
        """
        self._set_type(block, None)
        self._predeclare_block_structs(block)
        for node in block.nodes:
            self.visit(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionPrototype) -> None:
        """
        title: Visit FunctionPrototype nodes.
        parameters:
          node:
            type: astx.FunctionPrototype
        """
        for arg in node.args.nodes:
            self._resolve_declared_type(arg.type_, node=arg)
        self._resolve_declared_type(node.return_type, node=node)
        function = self.registry.resolve_function(node.name)
        if function is None:
            function = self.registry.register_function(node)
        function = self._synchronize_function_signature(function, node)
        self.bindings.bind_function(node.name, function, node=node)
        self._set_function(node, function)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionDef) -> None:
        """
        title: Visit FunctionDef nodes.
        parameters:
          node:
            type: astx.FunctionDef
        """
        for arg in node.prototype.args.nodes:
            self._resolve_declared_type(arg.type_, node=arg)
        self._resolve_declared_type(node.prototype.return_type, node=node)
        function = self.registry.resolve_function(node.name)
        if function is None:
            function = self.registry.register_function(
                node.prototype,
                definition=node,
            )
        function = self._synchronize_function_signature(
            function,
            node.prototype,
            definition=node,
        )
        self.bindings.bind_function(node.name, function, node=node)
        self._set_function(node.prototype, function)
        self._set_function(node, function)
        with self.context.in_function(function):
            with self.context.scope("function"):
                for arg_node, arg_symbol in zip(
                    node.prototype.args.nodes,
                    function.args,
                ):
                    self.context.scopes.declare(arg_symbol)
                    self._set_symbol(arg_node, arg_symbol)
                    self._set_type(arg_node, arg_symbol.type_)
                self.visit(node.body)
        if not isinstance(
            function.return_type, astx.NoneType
        ) and not self._guarantees_return(node.body):
            self.context.diagnostics.add(
                f"Function '{node.name}' with return type "
                f"'{function.return_type}' is missing a return statement",
                node=node,
            )

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.VariableDeclaration) -> None:
        """
        title: Visit VariableDeclaration nodes.
        parameters:
          node:
            type: astx.VariableDeclaration
        """
        self._resolve_declared_type(node.type_, node=node)
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self.visit(node.value)
            if self._require_value_expression(
                node.value,
                context=f"Initializer for '{node.name}'",
            ):
                validate_assignment(
                    self.context.diagnostics,
                    target_name=node.name,
                    target_type=node.type_,
                    value_type=self._expr_type(node.value),
                    node=node,
                )
        symbol = self.registry.declare_local(
            node.name,
            node.type_,
            is_mutable=node.mutability != astx.MutabilityKind.constant,
            declaration=node,
        )
        self._set_symbol(node, symbol)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.InlineVariableDeclaration) -> None:
        """
        title: Visit InlineVariableDeclaration nodes.
        parameters:
          node:
            type: astx.InlineVariableDeclaration
        """
        self._resolve_declared_type(node.type_, node=node)
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self.visit(node.value)
            if self._require_value_expression(
                node.value,
                context=f"Initializer for '{node.name}'",
            ):
                validate_assignment(
                    self.context.diagnostics,
                    target_name=node.name,
                    target_type=node.type_,
                    value_type=self._expr_type(node.value),
                    node=node,
                )
        symbol = self.registry.declare_local(
            node.name,
            node.type_,
            is_mutable=node.mutability != astx.MutabilityKind.constant,
            declaration=node,
        )
        self._set_symbol(node, symbol)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.StructDefStmt) -> None:
        """
        title: Visit StructDefStmt nodes.
        parameters:
          node:
            type: astx.StructDefStmt
        """
        struct = self.registry.register_struct(node)
        self.bindings.bind_struct(node.name, struct, node=node)
        struct = self._resolve_struct_fields(struct)
        self._set_struct(node, struct)
        self._validate_struct_cycles(struct)
        self._set_type(node, None)

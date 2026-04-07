# mypy: disable-error-code=no-redef

"""
title: Public semantic-analysis entry points.
summary: >-
  Implement the main semantic-analysis visitor plus the public single- and
  multi-module analysis entry points.
"""

from __future__ import annotations

from typing import cast

from plum import dispatch
from public import public

from irx import astx
from irx.analysis.context import SemanticContext
from irx.analysis.module_interfaces import (
    ImportResolver,
    ModuleKey,
    ParsedModule,
)
from irx.analysis.normalization import normalize_flags, normalize_operator
from irx.analysis.resolved_nodes import (
    ResolvedAssignment,
    ResolvedImportBinding,
    ResolvedOperator,
    SemanticBinding,
    SemanticFlags,
    SemanticFunction,
    SemanticInfo,
    SemanticModule,
    SemanticStruct,
    SemanticSymbol,
)
from irx.analysis.session import CompilationSession
from irx.analysis.symbols import (
    function_symbol,
    struct_symbol,
    variable_symbol,
    with_definition,
)
from irx.analysis.types import (
    clone_type,
    is_assignable,
    is_boolean_type,
    is_float_type,
    is_integer_type,
    is_numeric_type,
    is_string_type,
)
from irx.analysis.typing import binary_result_type, unary_result_type
from irx.analysis.validation import (
    validate_assignment,
    validate_call,
    validate_cast,
    validate_literal_datetime,
    validate_literal_time,
    validate_literal_timestamp,
)
from irx.astx.binary_op import (
    SPECIALIZED_BINARY_OP_EXTRA,
    specialize_binary_op,
)
from irx.base.visitors.base import BaseVisitor


class SemanticAnalyzer(BaseVisitor):
    """
    title: Walk the AST and attach node.semantic information.
    summary: >-
      Perform semantic analysis over AST nodes, attaching resolved sidecars and
      recording diagnostics as it goes.
    attributes:
      context:
        type: SemanticContext
      session:
        type: CompilationSession | None
      _visible_bindings:
        type: dict[ModuleKey, dict[str, SemanticBinding]]
    """

    context: SemanticContext
    session: CompilationSession | None
    _visible_bindings: dict[ModuleKey, dict[str, SemanticBinding]]

    def __init__(
        self,
        *,
        context: SemanticContext | None = None,
        session: CompilationSession | None = None,
    ) -> None:
        """
        title: Initialize SemanticAnalyzer.
        summary: >-
          Build an analyzer around either a fresh semantic context or a shared
          compilation session.
        parameters:
          context:
            type: SemanticContext | None
          session:
            type: CompilationSession | None
        """
        self.session = session
        self.context = context or SemanticContext()
        if session is not None:
            self.context.diagnostics = session.diagnostics
        self._visible_bindings = (
            session.visible_bindings if session is not None else {}
        )

    def analyze(self, node: astx.AST) -> astx.AST:
        """
        title: Analyze one AST root.
        summary: >-
          Run semantic analysis for a single AST root and raise if any
          diagnostics were recorded.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.AST
        """
        if isinstance(node, astx.Module):
            parsed_module = ParsedModule(node.name, node)
            self.analyze_parsed_module(parsed_module, predeclared=False)
        else:
            with self.context.scope("module"):
                self.visit(node)
        self.context.diagnostics.raise_if_errors()
        return node

    def analyze_parsed_module(
        self,
        parsed_module: ParsedModule,
        *,
        predeclared: bool,
    ) -> astx.Module:
        """
        title: Analyze one parsed module with a known module key.
        summary: >-
          Analyze one host-provided parsed module while preserving its
          externally-assigned module identity.
        parameters:
          parsed_module:
            type: ParsedModule
          predeclared:
            type: bool
        returns:
          type: astx.Module
        """
        with self.context.in_module(parsed_module.key):
            self._visit_module(parsed_module.ast, predeclared=predeclared)
        return parsed_module.ast

    def _semantic(self, node: astx.AST) -> SemanticInfo:
        """
        title: Semantic.
        summary: >-
          Return the semantic sidecar for a node, creating it on demand when
          analysis touches the node for the first time.
        parameters:
          node:
            type: astx.AST
        returns:
          type: SemanticInfo
        """
        info = cast(SemanticInfo | None, getattr(node, "semantic", None))
        if info is None or not isinstance(info, SemanticInfo):
            info = SemanticInfo()
            setattr(node, "semantic", info)
        return info

    def _set_type(
        self, node: astx.AST, type_: astx.DataType | None
    ) -> astx.DataType | None:
        """
        title: Set type.
        summary: >-
          Attach a resolved semantic type to a node and mirror it back to
          `node.type_` when that attribute exists.
        parameters:
          node:
            type: astx.AST
          type_:
            type: astx.DataType | None
        returns:
          type: astx.DataType | None
        """
        info = self._semantic(node)
        info.resolved_type = type_
        if type_ is not None and hasattr(node, "type_"):
            try:
                setattr(node, "type_", clone_type(type_))
            except Exception:
                pass
        return type_

    def _set_symbol(
        self, node: astx.AST, symbol: SemanticSymbol | None
    ) -> SemanticSymbol | None:
        """
        title: Set symbol.
        summary: >-
          Attach a resolved variable-like symbol to a node and propagate its
          type into the node sidecar.
        parameters:
          node:
            type: astx.AST
          symbol:
            type: SemanticSymbol | None
        returns:
          type: SemanticSymbol | None
        """
        info = self._semantic(node)
        info.resolved_symbol = symbol
        if symbol is not None:
            self._set_type(node, symbol.type_)
        return symbol

    def _set_function(
        self, node: astx.AST, function: SemanticFunction | None
    ) -> SemanticFunction | None:
        """
        title: Set function.
        summary: >-
          Attach a resolved function to a node and propagate the function's
          return type into the node sidecar.
        parameters:
          node:
            type: astx.AST
          function:
            type: SemanticFunction | None
        returns:
          type: SemanticFunction | None
        """
        info = self._semantic(node)
        info.resolved_function = function
        if function is not None:
            self._set_type(node, function.return_type)
        return function

    def _set_struct(
        self,
        node: astx.AST,
        struct: SemanticStruct | None,
    ) -> SemanticStruct | None:
        """
        title: Set struct.
        summary: Attach a resolved struct declaration to a node sidecar.
        parameters:
          node:
            type: astx.AST
          struct:
            type: SemanticStruct | None
        returns:
          type: SemanticStruct | None
        """
        info = self._semantic(node)
        info.resolved_struct = struct
        return struct

    def _set_module(
        self,
        node: astx.AST,
        module: SemanticModule | None,
    ) -> SemanticModule | None:
        """
        title: Set module.
        summary: Attach an imported module identity to a node sidecar.
        parameters:
          node:
            type: astx.AST
          module:
            type: SemanticModule | None
        returns:
          type: SemanticModule | None
        """
        info = self._semantic(node)
        info.resolved_module = module
        return module

    def _set_imports(
        self,
        node: astx.AST,
        imports: tuple[ResolvedImportBinding, ...],
    ) -> None:
        """
        title: Set resolved imports.
        summary: >-
          Store the resolved import-binding list produced for one import
          statement.
        parameters:
          node:
            type: astx.AST
          imports:
            type: tuple[ResolvedImportBinding, Ellipsis]
        """
        self._semantic(node).resolved_imports = imports

    def _set_flags(self, node: astx.AST, flags: SemanticFlags) -> None:
        """
        title: Set flags.
        summary: >-
          Attach normalized semantic flags such as unsigned and fast-math
          behavior to a node.
        parameters:
          node:
            type: astx.AST
          flags:
            type: SemanticFlags
        """
        info = self._semantic(node)
        info.semantic_flags = flags

    def _set_operator(
        self,
        node: astx.AST,
        operator: ResolvedOperator | None,
    ) -> None:
        """
        title: Set operator.
        summary: >-
          Attach normalized operator meaning to a node after semantic
          normalization.
        parameters:
          node:
            type: astx.AST
          operator:
            type: ResolvedOperator | None
        """
        info = self._semantic(node)
        info.resolved_operator = operator

    def _set_assignment(
        self, node: astx.AST, symbol: SemanticSymbol | None
    ) -> None:
        """
        title: Set assignment.
        summary: Record which resolved symbol an assignment-like node mutates.
        parameters:
          node:
            type: astx.AST
          symbol:
            type: SemanticSymbol | None
        """
        info = self._semantic(node)
        if symbol is None:
            info.resolved_assignment = None
            return
        info.resolved_assignment = ResolvedAssignment(symbol)

    def _current_module_key(self) -> ModuleKey:
        """
        title: Return the current analysis module key.
        summary: >-
          Return the active module identity, falling back to a synthetic root
          key outside module-scoped analysis.
        returns:
          type: ModuleKey
        """
        return self.context.current_module_key or "<root>"

    def _current_visible_bindings(self) -> dict[str, SemanticBinding]:
        """
        title: Return the current module visible binding table.
        summary: >-
          Return the per-module binding table used to resolve imported and
          local top-level names.
        returns:
          type: dict[str, SemanticBinding]
        """
        module_key = self._current_module_key()
        return self._visible_bindings.setdefault(module_key, {})

    def _function_binding(self, function: SemanticFunction) -> SemanticBinding:
        """
        title: Return a visible binding for a function.
        summary: >-
          Wrap a semantic function in the normalized module-visible binding
          shape.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: SemanticBinding
        """
        return SemanticBinding(
            kind="function",
            module_key=function.module_key,
            qualified_name=function.qualified_name,
            function=function,
        )

    def _struct_binding(self, struct: SemanticStruct) -> SemanticBinding:
        """
        title: Return a visible binding for a struct.
        summary: >-
          Wrap a semantic struct in the normalized module-visible binding
          shape.
        parameters:
          struct:
            type: SemanticStruct
        returns:
          type: SemanticBinding
        """
        return SemanticBinding(
            kind="struct",
            module_key=struct.module_key,
            qualified_name=struct.qualified_name,
            struct=struct,
        )

    def _module_binding(self, module: SemanticModule) -> SemanticBinding:
        """
        title: Return a visible binding for a module.
        summary: >-
          Wrap an imported module identity in the normalized module-visible
          binding shape.
        parameters:
          module:
            type: SemanticModule
        returns:
          type: SemanticBinding
        """
        return SemanticBinding(
            kind="module",
            module_key=module.module_key,
            qualified_name=str(module.module_key),
            module=module,
        )

    def _binding_conflicts(
        self,
        existing: SemanticBinding,
        new_binding: SemanticBinding,
    ) -> bool:
        """
        title: Return True when two visible bindings disagree.
        summary: >-
          Decide whether a newly introduced visible binding conflicts with an
          existing local top-level name.
        parameters:
          existing:
            type: SemanticBinding
          new_binding:
            type: SemanticBinding
        returns:
          type: bool
        """
        return (
            existing.kind != new_binding.kind
            or existing.qualified_name != new_binding.qualified_name
        )

    def _bind_visible_name(
        self,
        local_name: str,
        binding: SemanticBinding,
        *,
        node: astx.AST,
    ) -> None:
        """
        title: Bind one name in the current module namespace.
        summary: Bind one name in the current module namespace.
        parameters:
          local_name:
            type: str
          binding:
            type: SemanticBinding
          node:
            type: astx.AST
        """
        bindings = self._current_visible_bindings()
        existing = bindings.get(local_name)
        if existing is not None and self._binding_conflicts(existing, binding):
            self.context.diagnostics.add(
                f"Conflicting binding for '{local_name}'",
                node=node,
            )
            return
        bindings[local_name] = binding

    def _resolve_visible_name(
        self,
        name: str,
        *,
        module_key: ModuleKey | None = None,
    ) -> SemanticBinding | None:
        """
        title: Resolve one visible module binding.
        summary: Resolve one visible module binding.
        parameters:
          name:
            type: str
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticBinding | None
        """
        current_module_key = module_key or self._current_module_key()
        return self._visible_bindings.get(current_module_key, {}).get(name)

    def _imports_supported_here(self, node: astx.AST) -> bool:
        """
        title: Return True when import handling is available for this node.
        summary: Return True when import handling is available for this node.
        parameters:
          node:
            type: astx.AST
        returns:
          type: bool
        """
        if self.session is None:
            self.context.diagnostics.add(
                "Import statements require analyze_modules(...) with a "
                "host-provided resolver.",
                node=node,
            )
            return False
        current_scope = self.context.scopes.current
        if current_scope is None or current_scope.kind != "module":
            self.context.diagnostics.add(
                "Import statements are supported only at module top level "
                "in this MVP.",
                node=node,
            )
            return False
        return True

    def _expr_type(self, node: astx.AST | None) -> astx.DataType | None:
        """
        title: Expr type.
        summary: Expr type.
        parameters:
          node:
            type: astx.AST | None
        returns:
          type: astx.DataType | None
        """
        if node is None:
            return None
        info = cast(SemanticInfo | None, getattr(node, "semantic", None))
        if info is not None and info.resolved_type is not None:
            return info.resolved_type
        return getattr(node, "type_", None)

    def _declare_symbol(
        self,
        name: str,
        type_: astx.DataType,
        *,
        is_mutable: bool,
        declaration: astx.AST,
        kind: str = "variable",
    ) -> SemanticSymbol:
        """
        title: Declare symbol.
        summary: Declare symbol.
        parameters:
          name:
            type: str
          type_:
            type: astx.DataType
          is_mutable:
            type: bool
          declaration:
            type: astx.AST
          kind:
            type: str
        returns:
          type: SemanticSymbol
        """
        module_key = self._current_module_key()
        symbol = variable_symbol(
            self.context.next_symbol_id(kind),
            module_key,
            name,
            clone_type(type_),
            is_mutable=is_mutable,
            declaration=declaration,
            kind=kind,
        )
        if not self.context.scopes.declare(symbol):
            self.context.diagnostics.add(
                f"Identifier already declared: {name}",
                node=declaration,
            )
        return symbol

    def _register_function(
        self,
        prototype: astx.FunctionPrototype,
        *,
        definition: astx.FunctionDef | None = None,
    ) -> SemanticFunction:
        """
        title: Register function.
        summary: Register function.
        parameters:
          prototype:
            type: astx.FunctionPrototype
          definition:
            type: astx.FunctionDef | None
        returns:
          type: SemanticFunction
        """
        module_key = self._current_module_key()
        existing = self.context.get_function(module_key, prototype.name)
        if existing is not None:
            if definition is not None and existing.definition is not None:
                self.context.diagnostics.add(
                    f"Function '{prototype.name}' already defined",
                    node=definition,
                )
            if definition is not None:
                updated = with_definition(existing, definition)
                self.context.register_function(updated)
                self._bind_visible_name(
                    prototype.name,
                    self._function_binding(updated),
                    node=definition,
                )
                return updated
            return existing

        args = tuple(
            variable_symbol(
                self.context.next_symbol_id("arg"),
                module_key,
                arg.name,
                clone_type(arg.type_),
                is_mutable=True,
                declaration=arg,
                kind="argument",
            )
            for arg in prototype.args.nodes
        )
        function = function_symbol(
            self.context.next_symbol_id("fn"),
            module_key,
            prototype,
            args,
            definition=definition,
        )
        self.context.register_function(function)
        self._bind_visible_name(
            prototype.name,
            self._function_binding(function),
            node=definition or prototype,
        )
        return function

    def _register_struct(
        self,
        node: astx.StructDefStmt,
    ) -> SemanticStruct:
        """
        title: Register struct.
        summary: Register struct.
        parameters:
          node:
            type: astx.StructDefStmt
        returns:
          type: SemanticStruct
        """
        module_key = self._current_module_key()
        existing = self.context.get_struct(module_key, node.name)
        if existing is not None:
            if existing.declaration is not node:
                self.context.diagnostics.add(
                    f"Struct '{node.name}' already defined.",
                    node=node,
                )
            return existing

        struct = struct_symbol(
            self.context.next_symbol_id("struct"),
            module_key,
            node,
        )
        self.context.register_struct(struct)
        self._bind_visible_name(
            node.name,
            self._struct_binding(struct),
            node=node,
        )
        return struct

    def _predeclare_module_members(self, module: astx.Module) -> None:
        """
        title: Predeclare module members.
        summary: Predeclare module members.
        parameters:
          module:
            type: astx.Module
        """
        for node in module.nodes:
            if isinstance(node, astx.FunctionPrototype):
                self._set_function(node, self._register_function(node))
            elif isinstance(node, astx.FunctionDef):
                function = self._register_function(
                    node.prototype, definition=node
                )
                self._set_function(node.prototype, function)
                self._set_function(node, function)
            elif isinstance(node, astx.StructDefStmt):
                self._set_struct(node, self._register_struct(node))

    def _visit_module(
        self,
        module: astx.Module,
        *,
        predeclared: bool,
    ) -> None:
        """
        title: Visit one module with optional predeclaration.
        summary: Visit one module with optional predeclaration.
        parameters:
          module:
            type: astx.Module
          predeclared:
            type: bool
        """
        self._set_type(module, None)
        with self.context.scope("module"):
            if not predeclared:
                self._predeclare_module_members(module)
            for node in module.nodes:
                self.visit(node)

    @dispatch
    def visit(self, node: astx.AST) -> None:
        """
        title: Visit AST nodes.
        summary: Visit AST nodes.
        parameters:
          node:
            type: astx.AST
        """
        self._visit_plain_typed_node(node)

    @dispatch
    def visit(self, module: astx.Module) -> None:
        """
        title: Visit Module nodes.
        summary: Visit Module nodes.
        parameters:
          module:
            type: astx.Module
        """
        with self.context.in_module(module.name):
            self._visit_module(module, predeclared=False)

    @dispatch
    def visit(self, block: astx.Block) -> None:
        """
        title: Visit Block nodes.
        summary: Visit Block nodes.
        parameters:
          block:
            type: astx.Block
        """
        self._set_type(block, None)
        for node in block.nodes:
            self.visit(node)

    def _visit_plain_typed_node(self, node: astx.AST) -> None:
        """
        title: Visit plain typed node.
        summary: Visit plain typed node.
        parameters:
          node:
            type: astx.AST
        """
        self._set_type(node, getattr(node, "type_", None))

    @dispatch
    def visit(self, node: astx.FunctionPrototype) -> None:
        """
        title: Visit FunctionPrototype nodes.
        summary: Visit FunctionPrototype nodes.
        parameters:
          node:
            type: astx.FunctionPrototype
        """
        function = self.context.get_function(
            self._current_module_key(),
            node.name,
        )
        if function is None:
            function = self._register_function(node)
        self._set_function(node, function)

    @dispatch
    def visit(self, node: astx.FunctionDef) -> None:
        """
        title: Visit FunctionDef nodes.
        summary: Visit FunctionDef nodes.
        parameters:
          node:
            type: astx.FunctionDef
        """
        function = self.context.get_function(
            self._current_module_key(),
            node.name,
        )
        if function is None:
            function = self._register_function(node.prototype, definition=node)
        self._set_function(node.prototype, function)
        self._set_function(node, function)
        with self.context.in_function(function):
            with self.context.scope("function"):
                for arg_node, arg_symbol in zip(
                    node.prototype.args.nodes, function.args
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

    @dispatch
    def visit(self, node: astx.VariableDeclaration) -> None:
        """
        title: Visit VariableDeclaration nodes.
        summary: Visit VariableDeclaration nodes.
        parameters:
          node:
            type: astx.VariableDeclaration
        """
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self.visit(node.value)
            validate_assignment(
                self.context.diagnostics,
                target_name=node.name,
                target_type=node.type_,
                value_type=self._expr_type(node.value),
                node=node,
            )
        symbol = self._declare_symbol(
            node.name,
            node.type_,
            is_mutable=node.mutability != astx.MutabilityKind.constant,
            declaration=node,
        )
        self._set_symbol(node, symbol)

    @dispatch
    def visit(self, node: astx.InlineVariableDeclaration) -> None:
        """
        title: Visit InlineVariableDeclaration nodes.
        summary: Visit InlineVariableDeclaration nodes.
        parameters:
          node:
            type: astx.InlineVariableDeclaration
        """
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self.visit(node.value)
            validate_assignment(
                self.context.diagnostics,
                target_name=node.name,
                target_type=node.type_,
                value_type=self._expr_type(node.value),
                node=node,
            )
        symbol = self._declare_symbol(
            node.name,
            node.type_,
            is_mutable=node.mutability != astx.MutabilityKind.constant,
            declaration=node,
        )
        self._set_symbol(node, symbol)

    @dispatch
    def visit(self, node: astx.Identifier) -> None:
        """
        title: Visit Identifier nodes.
        summary: Visit Identifier nodes.
        parameters:
          node:
            type: astx.Identifier
        """
        symbol = self.context.scopes.resolve(node.name)
        if symbol is None:
            self.context.diagnostics.add(
                f"Unknown variable name: {node.name}",
                node=node,
            )
            self._set_type(
                node, cast(astx.DataType | None, getattr(node, "type_", None))
            )
            return
        self._set_symbol(node, symbol)

    @dispatch
    def visit(self, node: astx.VariableAssignment) -> None:
        """
        title: Visit VariableAssignment nodes.
        summary: Visit VariableAssignment nodes.
        parameters:
          node:
            type: astx.VariableAssignment
        """
        self.visit(node.value)
        symbol = self.context.scopes.resolve(node.name)
        if symbol is None:
            self.context.diagnostics.add(
                f"Identifier '{node.name}' not found in the named values.",
                node=node,
            )
            return
        if not symbol.is_mutable:
            self.context.diagnostics.add(
                f"Cannot assign to '{node.name}': declared as constant",
                node=node,
            )
        validate_assignment(
            self.context.diagnostics,
            target_name=node.name,
            target_type=symbol.type_,
            value_type=self._expr_type(node.value),
            node=node,
        )
        self._set_symbol(node, symbol)
        self._set_assignment(node, symbol)
        self._set_type(node, symbol.type_)

    @dispatch
    def visit(self, node: astx.UnaryOp) -> None:
        """
        title: Visit UnaryOp nodes.
        summary: Visit UnaryOp nodes.
        parameters:
          node:
            type: astx.UnaryOp
        """
        self.visit(node.operand)
        operand_type = self._expr_type(node.operand)
        result_type = unary_result_type(node.op_code, operand_type)
        if node.op_code in {"++", "--"} and isinstance(
            node.operand, astx.Identifier
        ):
            symbol = cast(
                SemanticInfo, getattr(node.operand, "semantic", SemanticInfo())
            ).resolved_symbol
            if symbol is not None and not symbol.is_mutable:
                self.context.diagnostics.add(
                    "Cannot mutate "
                    f"'{node.operand.name}': declared as constant",
                    node=node,
                )
        flags = normalize_flags(node, lhs_type=operand_type)
        self._set_flags(node, flags)
        self._set_operator(
            node,
            normalize_operator(
                node.op_code,
                result_type=result_type,
                lhs_type=operand_type,
                flags=flags,
            ),
        )
        self._set_type(node, result_type)

    @dispatch
    def visit(self, node: astx.BinaryOp) -> None:
        """
        title: Visit BinaryOp nodes.
        summary: Visit BinaryOp nodes.
        parameters:
          node:
            type: astx.BinaryOp
        """
        self.visit(node.lhs)
        self.visit(node.rhs)
        lhs_type = self._expr_type(node.lhs)
        rhs_type = self._expr_type(node.rhs)
        flags = normalize_flags(node, lhs_type=lhs_type, rhs_type=rhs_type)
        self._set_flags(node, flags)
        specialized = specialize_binary_op(node)
        if specialized is not node:
            setattr(specialized, "semantic", self._semantic(node))
        self._semantic(node).extras[SPECIALIZED_BINARY_OP_EXTRA] = specialized

        if node.op_code == "=":
            if not isinstance(node.lhs, astx.Identifier):
                self.context.diagnostics.add(
                    "destination of '=' must be a variable",
                    node=node,
                )
                return
            symbol = self.context.scopes.resolve(node.lhs.name)
            if symbol is None:
                self.context.diagnostics.add(
                    "codegen: Invalid lhs variable name",
                    node=node,
                )
                return
            if not symbol.is_mutable:
                self.context.diagnostics.add(
                    "Cannot assign to "
                    f"'{node.lhs.name}': declared as constant",
                    node=node,
                )
            validate_assignment(
                self.context.diagnostics,
                target_name=node.lhs.name,
                target_type=symbol.type_,
                value_type=rhs_type,
                node=node,
            )
            self._set_assignment(node, symbol)
            self._set_symbol(node.lhs, symbol)
            self._set_type(node, symbol.type_)
            self._set_operator(
                node,
                normalize_operator(
                    node.op_code,
                    result_type=symbol.type_,
                    lhs_type=symbol.type_,
                    rhs_type=rhs_type,
                    flags=flags,
                ),
            )
            return

        if flags.fma and flags.fma_rhs is None:
            self.context.diagnostics.add(
                "FMA requires a third operand (fma_rhs)",
                node=node,
            )
        if flags.fma and flags.fma_rhs is not None:
            self.visit(flags.fma_rhs)

        if node.op_code in {"+", "-", "*", "/", "%"} and not (
            (is_numeric_type(lhs_type) and is_numeric_type(rhs_type))
            or (
                node.op_code == "+"
                and is_string_type(lhs_type)
                and is_string_type(rhs_type)
            )
        ):
            if node.op_code not in {"|", "&", "^"}:
                self.context.diagnostics.add(
                    f"Invalid operator '{node.op_code}' for operand types",
                    node=node,
                )

        result_type = binary_result_type(node.op_code, lhs_type, rhs_type)
        self._set_type(node, result_type)
        self._set_operator(
            node,
            normalize_operator(
                node.op_code,
                result_type=result_type,
                lhs_type=lhs_type,
                rhs_type=rhs_type,
                flags=flags,
            ),
        )

    @dispatch
    def visit(self, node: astx.FunctionCall) -> None:
        """
        title: Visit FunctionCall nodes.
        summary: Visit FunctionCall nodes.
        parameters:
          node:
            type: astx.FunctionCall
        """
        arg_types: list[astx.DataType | None] = []
        for arg in node.args:
            self.visit(arg)
            arg_types.append(self._expr_type(arg))
        binding = self._resolve_visible_name(node.fn)
        if binding is None:
            self.context.diagnostics.add(
                "Unknown function referenced",
                node=node,
            )
            return
        if binding.kind != "function" or binding.function is None:
            self.context.diagnostics.add(
                f"Name '{node.fn}' does not resolve to a function",
                node=node,
            )
            return
        function = binding.function
        self._set_function(node, function)
        validate_call(
            self.context.diagnostics,
            function=function,
            arg_types=arg_types,
            node=node,
        )

    @dispatch
    def visit(self, node: astx.FunctionReturn) -> None:
        """
        title: Visit FunctionReturn nodes.
        summary: Visit FunctionReturn nodes.
        parameters:
          node:
            type: astx.FunctionReturn
        """
        if self.context.current_function is None:
            self.context.diagnostics.add(
                "Return statement outside function.",
                node=node,
            )
            return
        if node.value is not None:
            self.visit(node.value)
        return_type = self.context.current_function.return_type
        value_type = self._expr_type(node.value)
        if not is_assignable(return_type, value_type):
            self.context.diagnostics.add(
                "Return type mismatch.",
                node=node,
            )
        self._set_type(node, return_type)

    @dispatch
    def visit(self, node: astx.IfStmt) -> None:
        """
        title: Visit IfStmt nodes.
        summary: Visit IfStmt nodes.
        parameters:
          node:
            type: astx.IfStmt
        """
        self.visit(node.condition)
        self.visit(node.then)
        if node.else_ is not None:
            self.visit(node.else_)
        self._set_type(node, None)

    @dispatch
    def visit(self, node: astx.WhileStmt) -> None:
        """
        title: Visit WhileStmt nodes.
        summary: Visit WhileStmt nodes.
        parameters:
          node:
            type: astx.WhileStmt
        """
        self.visit(node.condition)
        with self.context.in_loop():
            self.visit(node.body)
        self._set_type(node, None)

    @dispatch
    def visit(self, node: astx.ForCountLoopStmt) -> None:
        """
        title: Visit ForCountLoopStmt nodes.
        summary: Visit ForCountLoopStmt nodes.
        parameters:
          node:
            type: astx.ForCountLoopStmt
        """
        with self.context.scope("for-count"):
            if node.initializer.value is not None:
                self.visit(node.initializer.value)
            symbol = self._declare_symbol(
                node.initializer.name,
                node.initializer.type_,
                is_mutable=(
                    node.initializer.mutability != astx.MutabilityKind.constant
                ),
                declaration=node.initializer,
            )
            self._set_symbol(node.initializer, symbol)
            self.visit(node.condition)
            self.visit(node.update)
            with self.context.in_loop():
                self.visit(node.body)
        self._set_type(node, None)

    @dispatch
    def visit(self, node: astx.ForRangeLoopStmt) -> None:
        """
        title: Visit ForRangeLoopStmt nodes.
        summary: Visit ForRangeLoopStmt nodes.
        parameters:
          node:
            type: astx.ForRangeLoopStmt
        """
        with self.context.scope("for-range"):
            self.visit(node.start)
            self.visit(node.end)
            if not isinstance(node.step, astx.LiteralNone):
                self.visit(node.step)
            symbol = self._declare_symbol(
                node.variable.name,
                node.variable.type_,
                is_mutable=(
                    node.variable.mutability != astx.MutabilityKind.constant
                ),
                declaration=node.variable,
            )
            self._set_symbol(node.variable, symbol)
            with self.context.in_loop():
                self.visit(node.body)
        self._set_type(node, None)

    @dispatch
    def visit(self, node: astx.BreakStmt) -> None:
        """
        title: Visit BreakStmt nodes.
        summary: Visit BreakStmt nodes.
        parameters:
          node:
            type: astx.BreakStmt
        """
        if self.context.loop_depth <= 0:
            self.context.diagnostics.add(
                "Break statement outside loop.",
                node=node,
            )
        self._set_type(node, None)

    @dispatch
    def visit(self, node: astx.ContinueStmt) -> None:
        """
        title: Visit ContinueStmt nodes.
        summary: Visit ContinueStmt nodes.
        parameters:
          node:
            type: astx.ContinueStmt
        """
        if self.context.loop_depth <= 0:
            self.context.diagnostics.add(
                "Continue statement outside loop.",
                node=node,
            )
        self._set_type(node, None)

    @dispatch
    def visit(self, node: astx.Cast) -> None:
        """
        title: Visit Cast nodes.
        summary: Visit Cast nodes.
        parameters:
          node:
            type: astx.Cast
        """
        self.visit(node.value)
        source_type = self._expr_type(node.value)
        target_type = cast(astx.DataType | None, node.target_type)
        validate_cast(
            self.context.diagnostics,
            source_type=source_type,
            target_type=target_type,
            node=node,
        )
        self._set_type(node, target_type)

    @dispatch
    def visit(self, node: astx.PrintExpr) -> None:
        """
        title: Visit PrintExpr nodes.
        summary: Visit PrintExpr nodes.
        parameters:
          node:
            type: astx.PrintExpr
        """
        self.visit(node.message)
        message_type = self._expr_type(node.message)
        if not (
            is_string_type(message_type)
            or is_integer_type(message_type)
            or is_float_type(message_type)
            or is_boolean_type(message_type)
        ):
            self.context.diagnostics.add(
                f"Unsupported message type in PrintExpr: {message_type}",
                node=node,
            )
        self._set_type(node, astx.Int32())

    @dispatch
    def visit(self, node: astx.ArrowInt32ArrayLength) -> None:
        """
        title: Visit ArrowInt32ArrayLength nodes.
        summary: Visit ArrowInt32ArrayLength nodes.
        parameters:
          node:
            type: astx.ArrowInt32ArrayLength
        """
        for item in node.values:
            self.visit(item)
            if not is_integer_type(self._expr_type(item)):
                self.context.diagnostics.add(
                    "Arrow helper supports only integer expressions",
                    node=item,
                )
        self._set_type(node, astx.Int32())

    @dispatch
    def visit(self, node: astx.StructDefStmt) -> None:
        """
        title: Visit StructDefStmt nodes.
        summary: Visit StructDefStmt nodes.
        parameters:
          node:
            type: astx.StructDefStmt
        """
        self._set_struct(node, self._register_struct(node))
        seen: set[str] = set()
        for attr in node.attributes:
            if attr.name in seen:
                self.context.diagnostics.add(
                    f"Struct field '{attr.name}' already defined.",
                    node=attr,
                )
            seen.add(attr.name)
        self._set_type(node, None)

    @dispatch
    def visit(self, node: astx.AliasExpr) -> None:
        """
        title: Visit AliasExpr nodes.
        summary: Visit AliasExpr nodes.
        parameters:
          node:
            type: astx.AliasExpr
        """
        self._set_type(node, None)

    @dispatch
    def visit(self, node: astx.ImportStmt) -> None:
        """
        title: Visit ImportStmt nodes.
        summary: Visit ImportStmt nodes.
        parameters:
          node:
            type: astx.ImportStmt
        """
        self._set_type(node, None)
        if not self._imports_supported_here(node):
            return

        resolved_imports: list[ResolvedImportBinding] = []
        for alias in node.names:
            resolved = self.session.resolve_import_specifier(
                self._current_module_key(),
                node,
                alias.name,
            )
            if resolved is None:
                continue
            semantic_module = SemanticModule(
                module_key=resolved.key,
                display_name=resolved.display_name,
            )
            binding = self._module_binding(semantic_module)
            local_name = alias.asname or alias.name
            self._bind_visible_name(local_name, binding, node=alias)
            resolved_binding = ResolvedImportBinding(
                local_name=local_name,
                requested_name=alias.name,
                source_module_key=resolved.key,
                binding=binding,
            )
            resolved_imports.append(resolved_binding)
            self._set_module(alias, semantic_module)
            self._set_imports(alias, (resolved_binding,))
        self._set_imports(node, tuple(resolved_imports))

    @dispatch
    def visit(self, node: astx.ImportFromStmt) -> None:
        """
        title: Visit ImportFromStmt nodes.
        summary: Visit ImportFromStmt nodes.
        parameters:
          node:
            type: astx.ImportFromStmt
        """
        self._set_type(node, None)
        if not self._imports_supported_here(node):
            return

        requested_specifier = f"{'.' * node.level}{node.module or ''}"
        resolved_module = self.session.resolve_import_specifier(
            self._current_module_key(),
            node,
            requested_specifier,
        )
        if resolved_module is None:
            return

        target_module = SemanticModule(
            module_key=resolved_module.key,
            display_name=resolved_module.display_name,
        )
        self._set_module(node, target_module)
        target_bindings = self._visible_bindings.get(resolved_module.key, {})
        resolved_imports: list[ResolvedImportBinding] = []

        for alias in node.names:
            if alias.name == "*":
                self.context.diagnostics.add(
                    "Wildcard imports are not supported in this MVP.",
                    node=alias,
                )
                continue
            target_binding = target_bindings.get(alias.name)
            if target_binding is None or target_binding.kind not in {
                "function",
                "struct",
            }:
                self.context.diagnostics.add(
                    f"Imported symbol '{alias.name}' was not found in "
                    f"module '{requested_specifier}'",
                    node=alias,
                )
                continue
            local_name = alias.asname or alias.name
            self._bind_visible_name(local_name, target_binding, node=alias)
            resolved_binding = ResolvedImportBinding(
                local_name=local_name,
                requested_name=alias.name,
                source_module_key=resolved_module.key,
                binding=target_binding,
            )
            resolved_imports.append(resolved_binding)
            self._set_module(alias, target_module)
            self._set_imports(alias, (resolved_binding,))
            if target_binding.function is not None:
                self._set_function(alias, target_binding.function)
            if target_binding.struct is not None:
                self._set_struct(alias, target_binding.struct)

        self._set_imports(node, tuple(resolved_imports))

    @dispatch
    def visit(self, node: astx.ImportExpr) -> None:
        """
        title: Visit ImportExpr nodes.
        summary: Visit ImportExpr nodes.
        parameters:
          node:
            type: astx.ImportExpr
        """
        self.context.diagnostics.add(
            "Import expressions are not supported in this MVP.",
            node=node,
        )
        self._set_type(node, None)

    @dispatch
    def visit(self, node: astx.ImportFromExpr) -> None:
        """
        title: Visit ImportFromExpr nodes.
        summary: Visit ImportFromExpr nodes.
        parameters:
          node:
            type: astx.ImportFromExpr
        """
        self.context.diagnostics.add(
            "Import expressions are not supported in this MVP.",
            node=node,
        )
        self._set_type(node, None)

    def _visit_temporal_literal(self, node: astx.AST) -> None:
        """
        title: Visit temporal literal.
        summary: Visit temporal literal.
        parameters:
          node:
            type: astx.AST
        """
        try:
            literal_value = cast(str, getattr(node, "value"))
            parsed_value: object
            if isinstance(node, astx.LiteralTime):
                parsed_value = validate_literal_time(literal_value)
            elif isinstance(node, astx.LiteralTimestamp):
                parsed_value = validate_literal_timestamp(literal_value)
            else:
                parsed_value = validate_literal_datetime(literal_value)
            self._semantic(node).extras["parsed_value"] = parsed_value
        except ValueError as exc:
            self.context.diagnostics.add(str(exc), node=node)
        self._set_type(node, getattr(node, "type_", None))

    @dispatch
    def visit(self, node: astx.LiteralTime) -> None:
        """
        title: Visit LiteralTime nodes.
        summary: Visit LiteralTime nodes.
        parameters:
          node:
            type: astx.LiteralTime
        """
        self._visit_temporal_literal(node)

    @dispatch
    def visit(self, node: astx.LiteralTimestamp) -> None:
        """
        title: Visit LiteralTimestamp nodes.
        summary: Visit LiteralTimestamp nodes.
        parameters:
          node:
            type: astx.LiteralTimestamp
        """
        self._visit_temporal_literal(node)

    @dispatch
    def visit(self, node: astx.LiteralDateTime) -> None:
        """
        title: Visit LiteralDateTime nodes.
        summary: Visit LiteralDateTime nodes.
        parameters:
          node:
            type: astx.LiteralDateTime
        """
        self._visit_temporal_literal(node)

    def _visit_element_sequence_literal(self, node: astx.AST) -> None:
        """
        title: Visit element sequence literal.
        summary: Visit element sequence literal.
        parameters:
          node:
            type: astx.AST
        """
        for element in cast(list[astx.AST], getattr(node, "elements")):
            self.visit(element)
        self._set_type(node, getattr(node, "type_", None))

    @dispatch
    def visit(self, node: astx.LiteralList) -> None:
        """
        title: Visit LiteralList nodes.
        summary: Visit LiteralList nodes.
        parameters:
          node:
            type: astx.LiteralList
        """
        self._visit_element_sequence_literal(node)

    @dispatch
    def visit(self, node: astx.LiteralTuple) -> None:
        """
        title: Visit LiteralTuple nodes.
        summary: Visit LiteralTuple nodes.
        parameters:
          node:
            type: astx.LiteralTuple
        """
        self._visit_element_sequence_literal(node)

    @dispatch
    def visit(self, node: astx.LiteralSet) -> None:
        """
        title: Visit LiteralSet nodes.
        summary: Visit LiteralSet nodes.
        parameters:
          node:
            type: astx.LiteralSet
        """
        for element in node.elements:
            self.visit(element)
        if node.elements and not all(
            isinstance(element, astx.Literal) for element in node.elements
        ):
            self.context.diagnostics.add(
                "LiteralSet: only integer constants are "
                "currently supported for lowering",
                node=node,
            )
        self._set_type(
            node, cast(astx.DataType | None, getattr(node, "type_", None))
        )

    @dispatch
    def visit(self, node: astx.LiteralDict) -> None:
        """
        title: Visit LiteralDict nodes.
        summary: Visit LiteralDict nodes.
        parameters:
          node:
            type: astx.LiteralDict
        """
        for key, value in node.elements.items():
            self.visit(key)
            self.visit(value)
        self._set_type(
            node, cast(astx.DataType | None, getattr(node, "type_", None))
        )

    @dispatch
    def visit(self, node: astx.SubscriptExpr) -> None:
        """
        title: Visit SubscriptExpr nodes.
        summary: Visit SubscriptExpr nodes.
        parameters:
          node:
            type: astx.SubscriptExpr
        """
        self.visit(node.value)
        if not isinstance(node.index, astx.LiteralNone):
            self.visit(node.index)
        value_type = self._expr_type(node.value)
        if isinstance(node.value, astx.LiteralDict):
            if not node.value.elements:
                self.context.diagnostics.add(
                    "SubscriptExpr: key lookup on empty dict",
                    node=node,
                )
            elif not isinstance(
                node.index,
                (
                    astx.LiteralInt8,
                    astx.LiteralInt16,
                    astx.LiteralInt32,
                    astx.LiteralInt64,
                    astx.LiteralUInt8,
                    astx.LiteralUInt16,
                    astx.LiteralUInt32,
                    astx.LiteralUInt64,
                    astx.LiteralFloat32,
                    astx.LiteralFloat64,
                    astx.Identifier,
                ),
            ):
                self.context.diagnostics.add(
                    "SubscriptExpr: only integer and floating-point "
                    "dict keys are supported",
                    node=node,
                )
        self._set_type(
            node,
            cast(
                astx.DataType | None,
                getattr(value_type, "value_type", None),
            ),
        )

    def _guarantees_return(self, node: astx.AST) -> bool:
        """
        title: Guarantees return.
        summary: >-
          Return whether a statement subtree guarantees that control flow exits
          through a return on every path.
        parameters:
          node:
            type: astx.AST
        returns:
          type: bool
        """
        if isinstance(node, astx.FunctionReturn):
            return True
        if isinstance(node, astx.Block):
            for child in node.nodes:
                if self._guarantees_return(child):
                    return True
            return False
        if isinstance(node, astx.IfStmt):
            if node.else_ is None:
                return False
            return self._guarantees_return(
                node.then
            ) and self._guarantees_return(node.else_)
        return False


@public
def analyze(node: astx.AST) -> astx.AST:
    """
    title: Analyze one AST root and attach node.semantic sidecars.
    summary: >-
      Run the single-root semantic-analysis path and return the same AST with
      semantic sidecars attached.
    parameters:
      node:
        type: astx.AST
    returns:
      type: astx.AST
    """
    return SemanticAnalyzer().analyze(node)


@public
def analyze_module(module: astx.Module) -> astx.Module:
    """
    title: Analyze an AST module.
    summary: >-
      Convenience wrapper for analyzing one module through the standard single-
      module entry point.
    parameters:
      module:
        type: astx.Module
    returns:
      type: astx.Module
    """
    return cast(astx.Module, analyze(module))


@public
def analyze_modules(
    root: ParsedModule,
    resolver: ImportResolver,
) -> CompilationSession:
    """
    title: Analyze a reachable graph of host-provided parsed modules.
    summary: >-
      Build a compilation session, expand the reachable import graph, and run
      cross-module semantic analysis over all reachable modules.
    parameters:
      root:
        type: ParsedModule
      resolver:
        type: ImportResolver
    returns:
      type: CompilationSession
    """
    session = CompilationSession(root=root, resolver=resolver)
    session.expand_graph()

    analyzer = SemanticAnalyzer(session=session)

    for parsed_module in session.ordered_modules():
        with analyzer.context.in_module(parsed_module.key):
            analyzer._predeclare_module_members(parsed_module.ast)

    for parsed_module in session.ordered_modules():
        with analyzer.context.in_module(parsed_module.key):
            with analyzer.context.scope("module"):
                for node in parsed_module.ast.nodes:
                    if isinstance(
                        node,
                        (astx.ImportStmt, astx.ImportFromStmt),
                    ):
                        analyzer.visit(node)

    for parsed_module in session.ordered_modules():
        analyzer.analyze_parsed_module(parsed_module, predeclared=True)

    session.diagnostics.raise_if_errors()
    return session

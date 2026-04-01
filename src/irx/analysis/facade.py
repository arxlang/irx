"""
title: Public semantic-analysis entry points.
"""

from __future__ import annotations

from typing import cast

import astx

from public import public

from irx import arrow as irx_arrow
from irx import system
from irx.analysis.context import SemanticContext
from irx.analysis.normalization import normalize_flags, normalize_operator
from irx.analysis.resolved_nodes import (
    ResolvedAssignment,
    ResolvedOperator,
    SemanticFlags,
    SemanticFunction,
    SemanticInfo,
    SemanticSymbol,
)
from irx.analysis.symbols import (
    function_symbol,
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


class SemanticAnalyzer:
    """
    title: Walk the AST and attach node.semantic information.
    attributes:
      context:
        type: SemanticContext
    """

    context: SemanticContext

    def __init__(self) -> None:
        self.context = SemanticContext()

    def analyze(self, node: astx.AST) -> astx.AST:
        """
        title: Analyze one AST root.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.AST
        """
        if isinstance(node, astx.Module):
            self._visit_module(node)
        else:
            with self.context.scope("module"):
                self._visit(node)
        self.context.diagnostics.raise_if_errors()
        return node

    def _semantic(self, node: astx.AST) -> SemanticInfo:
        info = cast(SemanticInfo | None, getattr(node, "semantic", None))
        if info is None or not isinstance(info, SemanticInfo):
            info = SemanticInfo()
            setattr(node, "semantic", info)
        return info

    def _set_type(
        self, node: astx.AST, type_: astx.DataType | None
    ) -> astx.DataType | None:
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
        info = self._semantic(node)
        info.resolved_symbol = symbol
        if symbol is not None:
            self._set_type(node, symbol.type_)
        return symbol

    def _set_function(
        self, node: astx.AST, function: SemanticFunction | None
    ) -> SemanticFunction | None:
        info = self._semantic(node)
        info.resolved_function = function
        if function is not None:
            self._set_type(node, function.return_type)
        return function

    def _set_flags(self, node: astx.AST, flags: SemanticFlags) -> None:
        info = self._semantic(node)
        info.semantic_flags = flags

    def _set_operator(
        self,
        node: astx.AST,
        operator: ResolvedOperator | None,
    ) -> None:
        info = self._semantic(node)
        info.resolved_operator = operator

    def _set_assignment(
        self, node: astx.AST, symbol: SemanticSymbol | None
    ) -> None:
        info = self._semantic(node)
        if symbol is None:
            info.resolved_assignment = None
            return
        info.resolved_assignment = ResolvedAssignment(symbol)

    def _expr_type(self, node: astx.AST | None) -> astx.DataType | None:
        if node is None:
            return None
        info = cast(SemanticInfo | None, getattr(node, "semantic", None))
        if info is not None and info.resolved_type is not None:
            return info.resolved_type
        return cast(astx.DataType | None, getattr(node, "type_", None))

    def _declare_symbol(
        self,
        name: str,
        type_: astx.DataType,
        *,
        is_mutable: bool,
        declaration: astx.AST,
        kind: str = "variable",
    ) -> SemanticSymbol:
        symbol = variable_symbol(
            self.context.next_symbol_id(kind),
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
        existing = self.context.functions.get(prototype.name)
        if existing is not None:
            if definition is not None and existing.definition is not None:
                self.context.diagnostics.add(
                    f"Function '{prototype.name}' already defined",
                    node=definition,
                )
            if definition is not None:
                updated = with_definition(existing, definition)
                self.context.functions[prototype.name] = updated
                return updated
            return existing

        args = tuple(
            variable_symbol(
                self.context.next_symbol_id("arg"),
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
            prototype,
            args,
            definition=definition,
        )
        self.context.functions[prototype.name] = function
        return function

    def _predeclare_module_members(self, module: astx.Module) -> None:
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
                if node.name in self.context.structs:
                    self.context.diagnostics.add(
                        f"Struct '{node.name}' already defined.",
                        node=node,
                    )
                else:
                    self.context.structs[node.name] = node

    def _visit(self, node: astx.AST) -> None:
        if isinstance(node, astx.Module):
            self._visit_module(node)
        elif isinstance(node, astx.Block):
            self._visit_block(node)
        elif isinstance(node, astx.FunctionDef):
            self._visit_function_def(node)
        elif isinstance(node, astx.FunctionPrototype):
            self._visit_function_prototype(node)
        elif isinstance(node, astx.FunctionCall):
            self._visit_function_call(node)
        elif isinstance(node, astx.FunctionReturn):
            self._visit_function_return(node)
        elif isinstance(node, astx.VariableDeclaration):
            self._visit_variable_declaration(node)
        elif isinstance(node, astx.InlineVariableDeclaration):
            self._visit_inline_variable_declaration(node)
        elif isinstance(node, astx.VariableAssignment):
            self._visit_variable_assignment(node)
        elif isinstance(node, astx.Identifier):
            self._visit_identifier(node)
        elif isinstance(node, astx.UnaryOp):
            self._visit_unary_op(node)
        elif isinstance(node, astx.BinaryOp):
            self._visit_binary_op(node)
        elif isinstance(node, astx.IfStmt):
            self._visit_if_stmt(node)
        elif isinstance(node, astx.WhileStmt):
            self._visit_while_stmt(node)
        elif isinstance(node, astx.ForCountLoopStmt):
            self._visit_for_count_loop(node)
        elif isinstance(node, astx.ForRangeLoopStmt):
            self._visit_for_range_loop(node)
        elif isinstance(node, astx.BreakStmt):
            self._visit_break(node)
        elif isinstance(node, astx.ContinueStmt):
            self._visit_continue(node)
        elif isinstance(node, system.Cast):
            self._visit_cast(node)
        elif isinstance(node, system.PrintExpr):
            self._visit_print(node)
        elif isinstance(node, irx_arrow.ArrowInt32ArrayLength):
            self._visit_arrow_length(node)
        elif isinstance(node, astx.StructDefStmt):
            self._visit_struct_def(node)
        elif isinstance(
            node,
            (
                astx.LiteralTime,
                astx.LiteralTimestamp,
                astx.LiteralDateTime,
                astx.LiteralList,
                astx.LiteralTuple,
                astx.LiteralSet,
                astx.LiteralDict,
                astx.SubscriptExpr,
            ),
        ):
            self._visit_special_literal(node)
        else:
            self._visit_plain_typed_node(node)

    def _visit_module(self, module: astx.Module) -> None:
        self._set_type(module, None)
        with self.context.scope("module"):
            self._predeclare_module_members(module)
            for node in module.nodes:
                self._visit(node)

    def _visit_block(self, block: astx.Block) -> None:
        self._set_type(block, None)
        with self.context.scope("block"):
            for node in block.nodes:
                self._visit(node)

    def _visit_plain_typed_node(self, node: astx.AST) -> None:
        self._set_type(
            node, cast(astx.DataType | None, getattr(node, "type_", None))
        )

    def _visit_function_prototype(self, node: astx.FunctionPrototype) -> None:
        function = self.context.functions.get(node.name)
        if function is None:
            function = self._register_function(node)
        self._set_function(node, function)

    def _visit_function_def(self, node: astx.FunctionDef) -> None:
        function = self.context.functions.get(node.name)
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
                self._visit_block(node.body)
        if not isinstance(
            function.return_type, astx.NoneType
        ) and not self._guarantees_return(node.body):
            self.context.diagnostics.add(
                f"Function '{node.name}' with return type "
                f"'{function.return_type}' is missing a return statement",
                node=node,
            )

    def _visit_variable_declaration(
        self, node: astx.VariableDeclaration
    ) -> None:
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self._visit(node.value)
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

    def _visit_inline_variable_declaration(
        self, node: astx.InlineVariableDeclaration
    ) -> None:
        if node.value is not None and not isinstance(
            node.value, astx.Undefined
        ):
            self._visit(node.value)
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

    def _visit_identifier(self, node: astx.Identifier) -> None:
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

    def _visit_variable_assignment(
        self, node: astx.VariableAssignment
    ) -> None:
        self._visit(node.value)
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

    def _visit_unary_op(self, node: astx.UnaryOp) -> None:
        self._visit(node.operand)
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

    def _visit_binary_op(self, node: astx.BinaryOp) -> None:
        self._visit(node.lhs)
        self._visit(node.rhs)
        lhs_type = self._expr_type(node.lhs)
        rhs_type = self._expr_type(node.rhs)
        flags = normalize_flags(node, lhs_type=lhs_type, rhs_type=rhs_type)
        self._set_flags(node, flags)

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
            self._visit(flags.fma_rhs)

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

    def _visit_function_call(self, node: astx.FunctionCall) -> None:
        function = self.context.functions.get(node.fn)
        arg_types: list[astx.DataType | None] = []
        for arg in node.args:
            self._visit(arg)
            arg_types.append(self._expr_type(arg))
        if function is None:
            self.context.diagnostics.add(
                "Unknown function referenced",
                node=node,
            )
            return
        self._set_function(node, function)
        validate_call(
            self.context.diagnostics,
            function=function,
            arg_types=arg_types,
            node=node,
        )

    def _visit_function_return(self, node: astx.FunctionReturn) -> None:
        if self.context.current_function is None:
            self.context.diagnostics.add(
                "Return statement outside function.",
                node=node,
            )
            return
        if node.value is not None:
            self._visit(node.value)
        return_type = self.context.current_function.return_type
        value_type = self._expr_type(node.value)
        if not is_assignable(return_type, value_type):
            self.context.diagnostics.add(
                "Return type mismatch.",
                node=node,
            )
        self._set_type(node, return_type)

    def _visit_if_stmt(self, node: astx.IfStmt) -> None:
        self._visit(node.condition)
        self._visit_block(node.then)
        if node.else_ is not None:
            self._visit_block(node.else_)
        self._set_type(node, None)

    def _visit_while_stmt(self, node: astx.WhileStmt) -> None:
        self._visit(node.condition)
        with self.context.in_loop():
            self._visit_block(node.body)
        self._set_type(node, None)

    def _visit_for_count_loop(self, node: astx.ForCountLoopStmt) -> None:
        with self.context.scope("for-count"):
            if node.initializer.value is not None:
                self._visit(node.initializer.value)
            symbol = self._declare_symbol(
                node.initializer.name,
                node.initializer.type_,
                is_mutable=(
                    node.initializer.mutability != astx.MutabilityKind.constant
                ),
                declaration=node.initializer,
            )
            self._set_symbol(node.initializer, symbol)
            self._visit(node.condition)
            self._visit(node.update)
            with self.context.in_loop():
                self._visit_block(node.body)
        self._set_type(node, None)

    def _visit_for_range_loop(self, node: astx.ForRangeLoopStmt) -> None:
        with self.context.scope("for-range"):
            self._visit(node.start)
            self._visit(node.end)
            if not isinstance(node.step, astx.LiteralNone):
                self._visit(node.step)
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
                self._visit_block(node.body)
        self._set_type(node, None)

    def _visit_break(self, node: astx.BreakStmt) -> None:
        if self.context.loop_depth <= 0:
            self.context.diagnostics.add(
                "Break statement outside loop.",
                node=node,
            )
        self._set_type(node, None)

    def _visit_continue(self, node: astx.ContinueStmt) -> None:
        if self.context.loop_depth <= 0:
            self.context.diagnostics.add(
                "Continue statement outside loop.",
                node=node,
            )
        self._set_type(node, None)

    def _visit_cast(self, node: system.Cast) -> None:
        self._visit(node.value)
        source_type = self._expr_type(node.value)
        target_type = cast(astx.DataType | None, node.target_type)
        validate_cast(
            self.context.diagnostics,
            source_type=source_type,
            target_type=target_type,
            node=node,
        )
        self._set_type(node, target_type)

    def _visit_print(self, node: system.PrintExpr) -> None:
        self._visit(node.message)
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

    def _visit_arrow_length(
        self, node: irx_arrow.ArrowInt32ArrayLength
    ) -> None:
        for item in node.values:
            self._visit(item)
            if not is_integer_type(self._expr_type(item)):
                self.context.diagnostics.add(
                    "Arrow helper supports only integer expressions",
                    node=item,
                )
        self._set_type(node, astx.Int32())

    def _visit_struct_def(self, node: astx.StructDefStmt) -> None:
        existing = self.context.structs.get(node.name)
        if existing not in {None, node}:
            self.context.diagnostics.add(
                f"Struct '{node.name}' already defined.",
                node=node,
            )
        else:
            self.context.structs[node.name] = node
        seen: set[str] = set()
        for attr in node.attributes:
            if attr.name in seen:
                self.context.diagnostics.add(
                    f"Struct field '{attr.name}' already defined.",
                    node=attr,
                )
            seen.add(attr.name)
        self._set_type(node, None)

    def _visit_special_literal(self, node: astx.AST) -> None:
        if isinstance(node, astx.LiteralTime):
            try:
                parsed_time = validate_literal_time(node.value)
                self._semantic(node).extras["parsed_value"] = parsed_time
            except ValueError as exc:
                self.context.diagnostics.add(str(exc), node=node)
        elif isinstance(node, astx.LiteralTimestamp):
            try:
                parsed_timestamp = validate_literal_timestamp(node.value)
                self._semantic(node).extras["parsed_value"] = parsed_timestamp
            except ValueError as exc:
                self.context.diagnostics.add(str(exc), node=node)
        elif isinstance(node, astx.LiteralDateTime):
            try:
                parsed_datetime = validate_literal_datetime(node.value)
                self._semantic(node).extras["parsed_value"] = parsed_datetime
            except ValueError as exc:
                self.context.diagnostics.add(str(exc), node=node)
        elif isinstance(node, (astx.LiteralList, astx.LiteralTuple)):
            for element in node.elements:
                self._visit(element)
        elif isinstance(node, astx.LiteralSet):
            for element in node.elements:
                self._visit(element)
            if node.elements:
                if not all(
                    isinstance(element, astx.Literal)
                    for element in node.elements
                ):
                    self.context.diagnostics.add(
                        "LiteralSet: only integer constants are "
                        "currently supported for lowering",
                        node=node,
                    )
        elif isinstance(node, astx.LiteralDict):
            for key, value in node.elements.items():
                self._visit(key)
                self._visit(value)
        elif isinstance(node, astx.SubscriptExpr):
            self._visit(node.value)
            if not isinstance(node.index, astx.LiteralNone):
                self._visit(node.index)
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
            return
        self._set_type(
            node, cast(astx.DataType | None, getattr(node, "type_", None))
        )

    def _guarantees_return(self, node: astx.AST) -> bool:
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
    parameters:
      module:
        type: astx.Module
    returns:
      type: astx.Module
    """
    return cast(astx.Module, analyze(module))

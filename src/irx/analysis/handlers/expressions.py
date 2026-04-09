# mypy: disable-error-code=no-redef

"""
title: Expression-oriented semantic visitors.
summary: >-
  Resolve lexical identifiers, visible function names, and expression typing
  rules while delegating reusable registration and binding logic elsewhere.
"""

from __future__ import annotations

from typing import cast

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.normalization import normalize_flags, normalize_operator
from irx.analysis.resolved_nodes import ResolvedFieldAccess, SemanticInfo
from irx.analysis.types import (
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
from irx.typecheck import typechecked


@typechecked
class ExpressionVisitorMixin(SemanticVisitorMixinBase):
    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.Identifier) -> None:
        """
        title: Visit Identifier nodes.
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
        self._set_type(node, symbol.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.VariableAssignment) -> None:
        """
        title: Visit VariableAssignment nodes.
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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.UnaryOp) -> None:
        """
        title: Visit UnaryOp nodes.
        parameters:
          node:
            type: astx.UnaryOp
        """
        self.visit(node.operand)
        operand_type = self._expr_type(node.operand)
        if (
            node.op_code == "!"
            and operand_type is not None
            and not is_boolean_type(operand_type)
        ):
            self.context.diagnostics.add(
                "unary operator '!' requires Boolean operand",
                node=node,
            )
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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BinaryOp) -> None:
        """
        title: Visit BinaryOp nodes.
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
            if not isinstance(node.lhs, (astx.Identifier, astx.FieldAccess)):
                self.context.diagnostics.add(
                    "destination of '=' must be a variable or field",
                    node=node,
                )
                return
            symbol = self._root_assignment_symbol(node.lhs)
            if symbol is None:
                self.context.diagnostics.add(
                    "destination of '=' must be a variable or field",
                    node=node,
                )
                return
            if not symbol.is_mutable:
                self.context.diagnostics.add(
                    f"Cannot assign to '{symbol.name}': declared as constant",
                    node=node,
                )
            target_name = (
                node.lhs.name
                if isinstance(node.lhs, astx.Identifier)
                else node.lhs.field_name
            )
            target_type = self._expr_type(node.lhs)
            validate_assignment(
                self.context.diagnostics,
                target_name=target_name,
                target_type=target_type,
                value_type=rhs_type,
                node=node,
            )
            self._set_assignment(node, symbol)
            if isinstance(node.lhs, astx.Identifier):
                self._set_symbol(node.lhs, symbol)
            self._set_type(node, target_type)
            self._set_operator(
                node,
                normalize_operator(
                    node.op_code,
                    result_type=target_type,
                    lhs_type=target_type,
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

        if (
            node.op_code in {"&&", "and", "||", "or"}
            and lhs_type is not None
            and rhs_type is not None
            and not (is_boolean_type(lhs_type) and is_boolean_type(rhs_type))
        ):
            self.context.diagnostics.add(
                f"logical operator '{node.op_code}' requires Boolean operands",
                node=node,
            )

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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FunctionCall) -> None:
        """
        title: Visit FunctionCall nodes.
        parameters:
          node:
            type: astx.FunctionCall
        """
        arg_types: list[astx.DataType | None] = []
        for arg in node.args:
            self.visit(arg)
            arg_types.append(self._expr_type(arg))
        binding = self.bindings.resolve(node.fn)
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
        self._set_type(node, function.return_type)
        validate_call(
            self.context.diagnostics,
            function=function,
            arg_types=arg_types,
            node=node,
        )

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FieldAccess) -> None:
        """
        title: Visit FieldAccess nodes.
        parameters:
          node:
            type: astx.FieldAccess
        """
        self.visit(node.value)
        base_type = self._expr_type(node.value)
        struct = self._resolve_struct_from_type(
            base_type,
            node=node,
            unknown_message="field access requires a struct value",
        )
        if struct is None:
            if not isinstance(base_type, astx.StructType):
                self.context.diagnostics.add(
                    "field access requires a struct value",
                    node=node,
                )
            self._set_type(node, None)
            return

        field_index = struct.field_indices.get(node.field_name)
        if field_index is None or field_index >= len(struct.fields):
            self.context.diagnostics.add(
                f"struct '{struct.name}' has no field '{node.field_name}'",
                node=node,
            )
            self._set_type(node, None)
            return

        field = struct.fields[field_index]
        self._set_struct(node, struct)
        self._set_field_access(node, ResolvedFieldAccess(struct, field))
        self._set_type(node, field.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.Cast) -> None:
        """
        title: Visit Cast nodes.
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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.PrintExpr) -> None:
        """
        title: Visit PrintExpr nodes.
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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ArrowInt32ArrayLength) -> None:
        """
        title: Visit ArrowInt32ArrayLength nodes.
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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.AliasExpr) -> None:
        """
        title: Visit AliasExpr nodes.
        parameters:
          node:
            type: astx.AliasExpr
        """
        self._set_type(node, None)

    def _visit_temporal_literal(self, node: astx.AST) -> None:
        """
        title: Visit one temporal literal.
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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralTime) -> None:
        """
        title: Visit LiteralTime nodes.
        parameters:
          node:
            type: astx.LiteralTime
        """
        self._visit_temporal_literal(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralTimestamp) -> None:
        """
        title: Visit LiteralTimestamp nodes.
        parameters:
          node:
            type: astx.LiteralTimestamp
        """
        self._visit_temporal_literal(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralDateTime) -> None:
        """
        title: Visit LiteralDateTime nodes.
        parameters:
          node:
            type: astx.LiteralDateTime
        """
        self._visit_temporal_literal(node)

    def _visit_element_sequence_literal(self, node: astx.AST) -> None:
        """
        title: Visit one element-sequence literal.
        parameters:
          node:
            type: astx.AST
        """
        for element in cast(list[astx.AST], getattr(node, "elements")):
            self.visit(element)
        self._set_type(node, getattr(node, "type_", None))

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralList) -> None:
        """
        title: Visit LiteralList nodes.
        parameters:
          node:
            type: astx.LiteralList
        """
        self._visit_element_sequence_literal(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralTuple) -> None:
        """
        title: Visit LiteralTuple nodes.
        parameters:
          node:
            type: astx.LiteralTuple
        """
        self._visit_element_sequence_literal(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralSet) -> None:
        """
        title: Visit LiteralSet nodes.
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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.LiteralDict) -> None:
        """
        title: Visit LiteralDict nodes.
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

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.SubscriptExpr) -> None:
        """
        title: Visit SubscriptExpr nodes.
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

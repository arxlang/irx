# mypy: disable-error-code=no-redef
# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=untyped-decorator

"""
title: Expression operator visitors.
summary: >-
  Analyze unary, binary, cast, and print expressions plus their assignment-like
  operator behavior.
"""

from __future__ import annotations

from typing import cast

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.normalization import normalize_flags, normalize_operator
from irx.analysis.types import (
    display_type_name,
    is_boolean_type,
    is_float_type,
    is_integer_type,
    is_numeric_type,
    is_string_type,
)
from irx.analysis.typing import binary_result_type, unary_result_type
from irx.analysis.validation import validate_assignment, validate_cast
from irx.astx.binary_op import (
    SPECIALIZED_BINARY_OP_EXTRA,
    specialize_binary_op,
)
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ExpressionOperatorVisitorMixin(SemanticVisitorMixinBase):
    """
    title: Expression visitors for unary, binary, cast, and print operations
    """

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.UnaryOp) -> None:
        """
        title: Visit UnaryOp nodes.
        parameters:
          node:
            type: astx.UnaryOp
        """
        self.visit(node.operand)
        if not self._require_value_expression(
            node.operand,
            context=f"Unary operator '{node.op_code}'",
        ):
            self._set_type(node, None)
            return
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
        if node.op_code in {"++", "--"}:
            resolved_target = self._resolve_mutation_target(
                node.operand,
                node=node,
                action="mutate",
                invalid_message=(
                    "mutation target must be a variable or field"
                ),
            )
            if resolved_target is not None:
                target_symbol, _target_name, _target_type = resolved_target
                self._set_assignment(node, target_symbol)
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
            resolved_target = self._resolve_mutation_target(
                node.lhs,
                node=node,
                action="assign to",
                invalid_message=(
                    "assignment target must be a variable or field"
                ),
            )
            if resolved_target is None:
                return
            assignment_symbol, target_name, target_type = resolved_target
            if self._require_value_expression(
                node.rhs,
                context=f"Assignment to '{target_name}'",
            ):
                validate_assignment(
                    self.context.diagnostics,
                    target_name=target_name,
                    target_type=target_type,
                    value_type=rhs_type,
                    node=node,
                )
            self._set_assignment(node, assignment_symbol)
            if isinstance(node.lhs, astx.Identifier):
                self._set_symbol(node.lhs, assignment_symbol)
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

        lhs_has_value = self._require_value_expression(
            node.lhs,
            context=f"Operator '{node.op_code}'",
        )
        rhs_has_value = self._require_value_expression(
            node.rhs,
            context=f"Operator '{node.op_code}'",
        )
        if not (lhs_has_value and rhs_has_value):
            self._set_type(node, None)
            self._set_operator(
                node,
                normalize_operator(
                    node.op_code,
                    result_type=None,
                    lhs_type=lhs_type,
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
    def visit(self, node: astx.Cast) -> None:
        """
        title: Visit Cast nodes.
        parameters:
          node:
            type: astx.Cast
        """
        self.visit(node.value)
        if not self._require_value_expression(
            node.value,
            context="Cast",
        ):
            self._set_type(node, cast(astx.DataType | None, node.target_type))
            return
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
        if not self._require_value_expression(
            node.message,
            context="PrintExpr",
        ):
            self._set_type(node, astx.Int32())
            return
        message_type = self._expr_type(node.message)
        if not (
            is_string_type(message_type)
            or is_integer_type(message_type)
            or is_float_type(message_type)
            or is_boolean_type(message_type)
        ):
            self.context.diagnostics.add(
                "unsupported PrintExpr message type "
                f"{display_type_name(message_type)}",
                node=node,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        self._set_type(node, astx.Int32())

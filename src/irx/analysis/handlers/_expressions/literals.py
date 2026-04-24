# mypy: disable-error-code=no-redef
# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=untyped-decorator

"""
Expression visitors for literals, list operations, and subscripts.
"""

from __future__ import annotations

from typing import cast

from irx import astx
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.types import is_integer_type
from irx.analysis.validation import (
    validate_assignment,
    validate_literal_datetime,
    validate_literal_time,
    validate_literal_timestamp,
)
from irx.builtins.collections.list import (
    list_element_type,
    list_has_concrete_element_type,
)
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ExpressionLiteralVisitorMixin(SemanticVisitorMixinBase):
    """
    title: Expression visitors for literals, list operations, and subscripts
    """

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
    def visit(self, node: astx.ListCreate) -> None:
        """
        title: Visit ListCreate nodes.
        parameters:
          node:
            type: astx.ListCreate
        """
        self._set_type(node, node.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ListIndex) -> None:
        """
        title: Visit ListIndex nodes.
        parameters:
          node:
            type: astx.ListIndex
        """
        self.visit(node.base)
        self.visit(node.index)

        value_type = self._expr_type(node.base)
        if not isinstance(value_type, astx.ListType):
            self.context.diagnostics.add(
                "list indexing requires a list value",
                node=node.base,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
            self._set_type(node, None)
            return

        index_type = self._expr_type(node.index)
        if not is_integer_type(index_type):
            self.context.diagnostics.add(
                "list indexing requires an integer index",
                node=node.index,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        if not list_has_concrete_element_type(value_type):
            self.context.diagnostics.add(
                "list indexing requires a single concrete list element type",
                node=node.base,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        self._set_type(node, list_element_type(value_type))

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ListLength) -> None:
        """
        title: Visit ListLength nodes.
        parameters:
          node:
            type: astx.ListLength
        """
        self.visit(node.base)
        if not self._require_value_expression(
            node.base,
            context="list length",
        ):
            self._set_type(node, astx.Int32())
            return

        if not isinstance(self._expr_type(node.base), astx.ListType):
            self.context.diagnostics.add(
                "list length requires a list value",
                node=node.base,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ListAppend) -> None:
        """
        title: Visit ListAppend nodes.
        parameters:
          node:
            type: astx.ListAppend
        """
        self.visit(node.base)
        self.visit(node.value)

        resolved_target = self._resolve_mutation_target(
            node.base,
            node=node,
            action="append to",
            invalid_message="list append target must be a variable or field",
        )
        if resolved_target is None:
            self._set_type(node, astx.Int32())
            return

        assignment_symbol, _target_name, target_type = resolved_target
        if not isinstance(target_type, astx.ListType):
            self.context.diagnostics.add(
                "list append requires a list target",
                node=node.base,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
            self._set_assignment(node, assignment_symbol)
            self._set_type(node, astx.Int32())
            return

        element_type = list_element_type(target_type)
        if element_type is None:
            self.context.diagnostics.add(
                "list append requires a single concrete list element type",
                node=node.base,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
            self._set_assignment(node, assignment_symbol)
            self._set_type(node, astx.Int32())
            return

        validate_assignment(
            self.context.diagnostics,
            target_name="list element",
            target_type=element_type,
            value_type=self._expr_type(node.value),
            node=node,
        )
        self._set_assignment(node, assignment_symbol)
        self._set_type(node, astx.Int32())

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
        if isinstance(value_type, astx.ListType):
            if isinstance(node.index, astx.LiteralNone):
                self.context.diagnostics.add(
                    "list slicing is not supported",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
                self._set_type(node, None)
                return
            index_type = self._expr_type(node.index)
            if not is_integer_type(index_type):
                self.context.diagnostics.add(
                    "list indexing requires an integer index",
                    node=node.index,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            if not list_has_concrete_element_type(value_type):
                self.context.diagnostics.add(
                    "list indexing requires a single concrete list element "
                    "type",
                    node=node.value,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            self._set_type(node, list_element_type(value_type))
            return
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

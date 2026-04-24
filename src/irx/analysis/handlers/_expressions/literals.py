# mypy: disable-error-code=no-redef
# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=untyped-decorator

"""
title: Expression literal visitors.
summary: >-
  Handle literal expressions, list operations, and generic subscript
  expressions during semantic analysis.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from irx import astx
from irx.analysis.collections import (
    collection_contains_types,
    collection_supports_sequence_search,
    is_collection_type,
)
from irx.analysis.handlers.base import (
    SemanticAnalyzerCore,
    SemanticVisitorMixinBase,
)
from irx.analysis.iterables import resolve_iteration_capability
from irx.analysis.resolved_nodes import (
    CollectionMethodKind,
    ResolvedCollectionMethod,
)
from irx.analysis.types import (
    display_type_name,
    is_assignable,
    is_boolean_type,
    is_integer_type,
)
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

    def _validate_comprehension_condition(
        self,
        condition: astx.AST,
    ) -> None:
        """
        title: Validate one comprehension filter expression.
        parameters:
          condition:
            type: astx.AST
        """
        if not self._require_value_expression(
            condition,
            context="comprehension filter",
        ):
            return
        condition_type = self._expr_type(condition)
        if condition_type is None or is_boolean_type(condition_type):
            return
        self.context.diagnostics.add(
            "comprehension filter must be Boolean, got "
            f"{display_type_name(condition_type)}",
            node=condition,
            code=DiagnosticCodes.SEMANTIC_INVALID_CONDITION,
        )

    def _visit_comprehension_clause(
        self,
        clause: astx.ComprehensionClause,
    ) -> None:
        """
        title: Visit one comprehension clause in the active scope.
        parameters:
          clause:
            type: astx.ComprehensionClause
        """
        if clause.is_async:
            self.context.diagnostics.add(
                "async comprehensions are not supported",
                node=clause,
                code=DiagnosticCodes.SEMANTIC_INVALID_CONTROL_FLOW,
            )

        self.visit(clause.iterable)
        iterable_type = self._expr_type(clause.iterable)
        iteration = resolve_iteration_capability(
            clause.iterable,
            iterable_type,
        )
        if iteration is None:
            self.context.diagnostics.add(
                "comprehension requires an iterable value, got "
                f"{display_type_name(iterable_type)}",
                node=clause.iterable,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
            self._set_iteration(clause, None)
            self._set_type(clause, None)
            return

        symbol = self._declare_iteration_target(
            clause.target,
            iteration.element_type,
            kind="comprehension",
        )
        resolved_iteration = replace(iteration, target_symbol=symbol)
        self._set_iteration(clause.iterable, resolved_iteration)
        self._set_iteration(clause, resolved_iteration)
        self._set_type(clause, iteration.element_type)

        for condition in clause.conditions.nodes:
            self.visit(condition)
            self._validate_comprehension_condition(condition)

    def _visit_comprehension_clauses(
        self,
        clauses: list[astx.ComprehensionClause],
    ) -> None:
        """
        title: Visit all comprehension clauses in left-to-right order.
        parameters:
          clauses:
            type: list[astx.ComprehensionClause]
        """
        for clause in clauses:
            self._visit_comprehension_clause(clause)

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

    def _has_compatible_collection_probe(
        self,
        expected_types: tuple[astx.DataType, ...],
        actual_type: astx.DataType | None,
    ) -> bool:
        """
        title: Return whether one collection probe type is compatible.
        parameters:
          expected_types:
            type: tuple[astx.DataType, Ellipsis]
          actual_type:
            type: astx.DataType | None
        returns:
          type: bool
        """
        if actual_type is None or not expected_types:
            return True
        return any(
            is_assignable(expected_type, actual_type)
            for expected_type in expected_types
        )

    def _set_resolved_collection_method(
        self,
        node: astx.AST,
        *,
        base: astx.AST,
        method: CollectionMethodKind,
        return_type: astx.DataType,
        argument_types: tuple[astx.DataType, ...] = (),
    ) -> None:
        """
        title: Attach resolved collection method metadata.
        parameters:
          node:
            type: astx.AST
          base:
            type: astx.AST
          method:
            type: CollectionMethodKind
          return_type:
            type: astx.DataType
          argument_types:
            type: tuple[astx.DataType, Ellipsis]
        """
        receiver_type = self._expr_type(base)
        if receiver_type is None:
            self._set_collection_method(node, None)
            return
        self._set_collection_method(
            node,
            ResolvedCollectionMethod(
                receiver_node=base,
                receiver_type=receiver_type,
                method=method,
                return_type=return_type,
                argument_types=argument_types,
            ),
        )

    def _validate_collection_query_base(
        self,
        node: astx.AST,
        base: astx.AST,
        *,
        context: str,
    ) -> astx.DataType | None:
        """
        title: Validate one collection query receiver.
        parameters:
          node:
            type: astx.AST
          base:
            type: astx.AST
          context:
            type: str
        returns:
          type: astx.DataType | None
        """
        self.visit(base)
        if not self._require_value_expression(base, context=context):
            return None
        base_type = self._expr_type(base)
        if is_collection_type(base_type):
            return base_type
        self.context.diagnostics.add(
            f"{context} requires a collection value, got "
            f"{display_type_name(base_type)}",
            node=base,
            code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
        )
        return None

    def _collection_method_supported_by_lowering(
        self,
        *,
        method: CollectionMethodKind,
        base: astx.AST,
        base_type: astx.DataType | None,
    ) -> bool:
        """
        title: Return whether the backend can lower a collection method.
        parameters:
          method:
            type: CollectionMethodKind
          base:
            type: astx.AST
          base_type:
            type: astx.DataType | None
        returns:
          type: bool
        """
        if base_type is None:
            return True
        if isinstance(base_type, astx.ListType):
            return True
        if isinstance(base_type, astx.TupleType):
            if method in (
                CollectionMethodKind.LENGTH,
                CollectionMethodKind.IS_EMPTY,
            ):
                return True
            return isinstance(base, astx.LiteralTuple)
        if isinstance(base_type, astx.SetType):
            return isinstance(base, astx.LiteralSet) and method in (
                CollectionMethodKind.LENGTH,
                CollectionMethodKind.IS_EMPTY,
                CollectionMethodKind.CONTAINS,
            )
        if isinstance(base_type, astx.DictType):
            return isinstance(base, astx.LiteralDict) and method in (
                CollectionMethodKind.LENGTH,
                CollectionMethodKind.IS_EMPTY,
                CollectionMethodKind.CONTAINS,
            )
        return False

    def _validate_collection_lowering_support(
        self,
        *,
        node: astx.AST,
        base: astx.AST,
        base_type: astx.DataType | None,
        method: CollectionMethodKind,
        context: str,
    ) -> bool:
        """
        title: Validate that one collection method is currently lowerable.
        parameters:
          node:
            type: astx.AST
          base:
            type: astx.AST
          base_type:
            type: astx.DataType | None
          method:
            type: CollectionMethodKind
          context:
            type: str
        returns:
          type: bool
        """
        if self._collection_method_supported_by_lowering(
            method=method,
            base=base,
            base_type=base_type,
        ):
            return True

        self.context.diagnostics.add(
            f"{context} currently supports literal collections, "
            "dynamic lists, and tuple length/emptiness only",
            node=base,
            code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
        )
        return False

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.CollectionLength) -> None:
        """
        title: Visit CollectionLength nodes.
        parameters:
          node:
            type: astx.CollectionLength
        """
        return_type = astx.Int32()
        base_type = self._validate_collection_query_base(
            node,
            node.base,
            context="collection length",
        )
        supported = self._validate_collection_lowering_support(
            node=node,
            base=node.base,
            base_type=base_type,
            method=CollectionMethodKind.LENGTH,
            context="collection length",
        )
        if base_type is not None and supported:
            self._set_resolved_collection_method(
                node,
                base=node.base,
                method=CollectionMethodKind.LENGTH,
                return_type=return_type,
            )
        self._set_type(node, return_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.CollectionIsEmpty) -> None:
        """
        title: Visit CollectionIsEmpty nodes.
        parameters:
          node:
            type: astx.CollectionIsEmpty
        """
        return_type = astx.Boolean()
        base_type = self._validate_collection_query_base(
            node,
            node.base,
            context="collection emptiness",
        )
        supported = self._validate_collection_lowering_support(
            node=node,
            base=node.base,
            base_type=base_type,
            method=CollectionMethodKind.IS_EMPTY,
            context="collection emptiness",
        )
        if base_type is not None and supported:
            self._set_resolved_collection_method(
                node,
                base=node.base,
                method=CollectionMethodKind.IS_EMPTY,
                return_type=return_type,
            )
        self._set_type(node, return_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.CollectionContains) -> None:
        """
        title: Visit CollectionContains nodes.
        parameters:
          node:
            type: astx.CollectionContains
        """
        return_type = astx.Boolean()
        base_type = self._validate_collection_query_base(
            node,
            node.base,
            context="collection containment",
        )
        self.visit(node.value)
        expected_types = collection_contains_types(base_type)
        actual_type = self._expr_type(node.value)
        if not self._has_compatible_collection_probe(
            expected_types,
            actual_type,
        ):
            self.context.diagnostics.add(
                "collection containment probe expects "
                + " or ".join(
                    display_type_name(expected_type)
                    for expected_type in expected_types
                )
                + f", got {display_type_name(actual_type)}",
                node=node.value,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        supported = self._validate_collection_lowering_support(
            node=node,
            base=node.base,
            base_type=base_type,
            method=CollectionMethodKind.CONTAINS,
            context="collection containment",
        )
        if base_type is not None and supported:
            self._set_resolved_collection_method(
                node,
                base=node.base,
                method=CollectionMethodKind.CONTAINS,
                return_type=return_type,
                argument_types=expected_types,
            )
        self._set_type(node, return_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.CollectionIndex) -> None:
        """
        title: Visit CollectionIndex nodes.
        parameters:
          node:
            type: astx.CollectionIndex
        """
        return_type = astx.Int32()
        base_type = self._validate_collection_query_base(
            node,
            node.base,
            context="collection index",
        )
        self.visit(node.value)
        expected_types = collection_contains_types(base_type)
        if base_type is not None and not collection_supports_sequence_search(
            base_type
        ):
            self.context.diagnostics.add(
                "collection index requires a list or tuple value",
                node=node.base,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        elif not self._has_compatible_collection_probe(
            expected_types,
            self._expr_type(node.value),
        ):
            self.context.diagnostics.add(
                "collection index probe expects "
                + " or ".join(
                    display_type_name(expected_type)
                    for expected_type in expected_types
                )
                + f", got {display_type_name(self._expr_type(node.value))}",
                node=node.value,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        supported = self._validate_collection_lowering_support(
            node=node,
            base=node.base,
            base_type=base_type,
            method=CollectionMethodKind.INDEX,
            context="collection index",
        )
        if (
            base_type is not None
            and collection_supports_sequence_search(base_type)
            and supported
        ):
            self._set_resolved_collection_method(
                node,
                base=node.base,
                method=CollectionMethodKind.INDEX,
                return_type=return_type,
                argument_types=expected_types,
            )
        self._set_type(node, return_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.CollectionCount) -> None:
        """
        title: Visit CollectionCount nodes.
        parameters:
          node:
            type: astx.CollectionCount
        """
        return_type = astx.Int32()
        base_type = self._validate_collection_query_base(
            node,
            node.base,
            context="collection count",
        )
        self.visit(node.value)
        expected_types = collection_contains_types(base_type)
        if base_type is not None and not collection_supports_sequence_search(
            base_type
        ):
            self.context.diagnostics.add(
                "collection count requires a list or tuple value",
                node=node.base,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        elif not self._has_compatible_collection_probe(
            expected_types,
            self._expr_type(node.value),
        ):
            self.context.diagnostics.add(
                "collection count probe expects "
                + " or ".join(
                    display_type_name(expected_type)
                    for expected_type in expected_types
                )
                + f", got {display_type_name(self._expr_type(node.value))}",
                node=node.value,
                code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
            )
        supported = self._validate_collection_lowering_support(
            node=node,
            base=node.base,
            base_type=base_type,
            method=CollectionMethodKind.COUNT,
            context="collection count",
        )
        if (
            base_type is not None
            and collection_supports_sequence_search(base_type)
            and supported
        ):
            self._set_resolved_collection_method(
                node,
                base=node.base,
                method=CollectionMethodKind.COUNT,
                return_type=return_type,
                argument_types=expected_types,
            )
        self._set_type(node, return_type)

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
    def visit(self, node: astx.ComprehensionClause) -> None:
        """
        title: Visit ComprehensionClause nodes.
        parameters:
          node:
            type: astx.ComprehensionClause
        """
        with self.context.scope("comprehension-clause"):
            self._visit_comprehension_clause(node)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ListComprehension) -> None:
        """
        title: Visit ListComprehension nodes.
        parameters:
          node:
            type: astx.ListComprehension
        """
        with self.context.scope("list-comprehension"):
            self._visit_comprehension_clauses(list(node.generators.nodes))
            self.visit(node.element)
            element_type = self._expr_type(node.element)
        if element_type is None:
            self._set_type(node, None)
            return
        self._set_type(node, astx.ListType([element_type]))

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.GeneratorExpr) -> None:
        """
        title: Visit GeneratorExpr nodes.
        parameters:
          node:
            type: astx.GeneratorExpr
        """
        with self.context.scope("generator-expression"):
            self._visit_comprehension_clauses(list(node.generators.nodes))
            self.visit(node.element)
            element_type = self._expr_type(node.element)
        if element_type is None:
            self._set_type(node, None)
            return
        self._set_type(node, astx.GeneratorType(element_type))

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.SetComprehension) -> None:
        """
        title: Visit SetComprehension nodes.
        parameters:
          node:
            type: astx.SetComprehension
        """
        with self.context.scope("set-comprehension"):
            self._visit_comprehension_clauses(list(node.generators.nodes))
            self.visit(node.element)
            element_type = self._expr_type(node.element)
        if element_type is None:
            self._set_type(node, None)
            return
        self._set_type(node, astx.SetType(element_type))

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.DictComprehension) -> None:
        """
        title: Visit DictComprehension nodes.
        parameters:
          node:
            type: astx.DictComprehension
        """
        with self.context.scope("dict-comprehension"):
            self._visit_comprehension_clauses(list(node.generators.nodes))
            self.visit(node.key)
            self.visit(node.value)
            key_type = self._expr_type(node.key)
            value_type = self._expr_type(node.value)
        if key_type is None or value_type is None:
            self._set_type(node, None)
            return
        self._set_type(node, astx.DictType(key_type, value_type))

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

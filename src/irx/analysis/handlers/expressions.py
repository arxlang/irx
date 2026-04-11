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
    bit_width,
    display_type_name,
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
from irx.buffer import (
    BUFFER_VIEW_ELEMENT_TYPE_EXTRA,
    BUFFER_VIEW_METADATA_EXTRA,
    BufferOwnership,
    BufferViewMetadata,
    buffer_view_is_readonly,
    buffer_view_ownership,
    validate_buffer_view_metadata,
)
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked

RAW_BUFFER_BYTE_BITS = 8


@typechecked
class ExpressionVisitorMixin(SemanticVisitorMixinBase):
    def _static_buffer_view_metadata(
        self,
        node: astx.AST,
    ) -> BufferViewMetadata | None:
        """
        title: Return static buffer metadata when analysis can prove it.
        parameters:
          node:
            type: astx.AST
        returns:
          type: BufferViewMetadata | None
        """
        semantic = self._semantic(node)
        metadata = semantic.extras.get(BUFFER_VIEW_METADATA_EXTRA)
        if isinstance(metadata, BufferViewMetadata):
            return metadata

        symbol = semantic.resolved_symbol
        declaration = symbol.declaration if symbol is not None else None
        initializer = getattr(declaration, "value", None)
        if not isinstance(initializer, astx.AST):
            return None

        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        metadata = initializer_extras.get(BUFFER_VIEW_METADATA_EXTRA)
        if isinstance(metadata, BufferViewMetadata):
            return metadata
        return None

    def _static_buffer_view_element_type(
        self,
        node: astx.AST,
    ) -> astx.DataType | None:
        """
        title: Return the scalar element type when analysis can prove it.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.DataType | None
        """
        semantic = self._semantic(node)
        element_type = semantic.extras.get(BUFFER_VIEW_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type

        view_type = self._expr_type(node)
        if (
            isinstance(view_type, astx.BufferViewType)
            and view_type.element_type is not None
        ):
            return view_type.element_type

        symbol = semantic.resolved_symbol
        declaration = symbol.declaration if symbol is not None else None
        initializer = getattr(declaration, "value", None)
        if not isinstance(initializer, astx.AST):
            return None

        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        element_type = initializer_extras.get(BUFFER_VIEW_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type

        initializer_type = getattr(
            initializer_semantic,
            "resolved_type",
            getattr(initializer, "type_", None),
        )
        if (
            isinstance(initializer_type, astx.BufferViewType)
            and initializer_type.element_type is not None
        ):
            return initializer_type.element_type
        return None

    def _static_integer_literal_value(self, node: astx.AST) -> int | None:
        """
        title: Return a static integer literal value when present.
        parameters:
          node:
            type: astx.AST
        returns:
          type: int | None
        """
        if isinstance(
            node,
            (
                astx.LiteralInt8,
                astx.LiteralInt16,
                astx.LiteralInt32,
                astx.LiteralInt64,
                astx.LiteralUInt8,
                astx.LiteralUInt16,
                astx.LiteralUInt32,
                astx.LiteralUInt64,
                astx.LiteralUInt128,
            ),
        ):
            return int(node.value)
        return None

    def _validate_buffer_view_index_operation(
        self,
        *,
        node: astx.AST,
        base: astx.AST,
        indices: list[astx.AST],
        is_store: bool,
    ) -> astx.DataType | None:
        """
        title: Validate one low-level buffer view indexed access.
        parameters:
          node:
            type: astx.AST
          base:
            type: astx.AST
          indices:
            type: list[astx.AST]
          is_store:
            type: bool
        returns:
          type: astx.DataType | None
        """
        base_type = self._expr_type(base)
        if not isinstance(base_type, astx.BufferViewType):
            self.context.diagnostics.add(
                "buffer view indexing requires a BufferViewType base",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        metadata = self._static_buffer_view_metadata(base)
        if metadata is None:
            self.context.diagnostics.add(
                "buffer view indexing requires static descriptor metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif len(indices) != metadata.ndim:
            self.context.diagnostics.add(
                "buffer view indexing index count must match descriptor ndim",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif len(metadata.shape) == metadata.ndim:
            for axis, index in enumerate(indices):
                extent = metadata.shape[axis]
                static_index = self._static_integer_literal_value(index)
                if static_index is None:
                    continue
                if static_index < 0 or static_index >= extent:
                    self.context.diagnostics.add(
                        "buffer view index "
                        f"{axis} statically out of bounds for extent {extent}",
                        node=index,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )

        if (
            is_store
            and metadata is not None
            and buffer_view_is_readonly(metadata.flags)
        ):
            self.context.diagnostics.add(
                "cannot write through a readonly buffer view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        for index in indices:
            index_type = self._expr_type(index)
            if not is_integer_type(index_type):
                self.context.diagnostics.add(
                    "buffer view indices must be integer typed",
                    node=index,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )
                continue
            if bit_width(index_type) > bit_width(astx.Int64()):
                self.context.diagnostics.add(
                    "buffer view indices must fit 64-bit "
                    "descriptor stride arithmetic",
                    node=index,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )

        element_type = self._static_buffer_view_element_type(base)
        if element_type is None:
            self.context.diagnostics.add(
                "buffer view indexing requires a known element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
            return None
        if not (
            is_integer_type(element_type)
            or is_float_type(element_type)
            or is_boolean_type(element_type)
        ):
            self.context.diagnostics.add(
                "buffer view indexing requires a scalar element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
            return None
        self._semantic(node).extras[BUFFER_VIEW_ELEMENT_TYPE_EXTRA] = (
            element_type
        )
        return element_type

    def _validate_buffer_lifetime_operation(
        self,
        *,
        node: astx.AST,
        view: astx.AST,
        operation: str,
    ) -> None:
        """
        title: Validate one explicit buffer lifetime helper operation.
        parameters:
          node:
            type: astx.AST
          view:
            type: astx.AST
          operation:
            type: str
        """
        metadata = self._static_buffer_view_metadata(view)
        if metadata is None:
            return
        ownership = buffer_view_ownership(metadata.flags)
        if ownership is BufferOwnership.BORROWED or metadata.owner.is_null:
            self.context.diagnostics.add(
                f"buffer {operation} requires an owned or external-owner view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

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
                f"cannot resolve name '{node.name}'",
                node=node,
                code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
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
                f"cannot assign to unresolved name '{node.name}'",
                node=node,
                code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
            )
            return
        if not symbol.is_mutable:
            self.context.diagnostics.add(
                f"Cannot assign to '{node.name}': declared as constant",
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
            )
        if self._require_value_expression(
            node.value,
            context=f"Assignment to '{node.name}'",
        ):
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
                    "assignment target must be a variable or field",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
                )
                return
            symbol = self._root_assignment_symbol(node.lhs)
            if symbol is None:
                self.context.diagnostics.add(
                    "assignment target must be a variable or field",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
                )
                return
            if not symbol.is_mutable:
                self.context.diagnostics.add(
                    f"Cannot assign to '{symbol.name}': declared as constant",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
                )
            target_name = (
                node.lhs.name
                if isinstance(node.lhs, astx.Identifier)
                else node.lhs.field_name
            )
            target_type = self._expr_type(node.lhs)
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
                f"cannot resolve function '{node.fn}'",
                node=node,
                code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
            )
            return
        if binding.kind != "function" or binding.function is None:
            self.context.diagnostics.add(
                f"name '{node.fn}' does not resolve to a function",
                node=node,
                code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
            )
            return
        function = binding.function
        self._set_function(node, function)
        call_resolution = validate_call(
            self.context.diagnostics,
            function=function,
            arg_types=arg_types,
            node=node,
        )
        self._set_call(node, call_resolution)
        self._set_type(node, call_resolution.result_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.FieldAccess) -> None:
        """
        title: Visit FieldAccess nodes.
        parameters:
          node:
            type: astx.FieldAccess
        """
        self.visit(node.value)
        if not self._require_value_expression(
            node.value,
            context="Field access",
        ):
            self._set_type(node, None)
            return
        base_type = self._expr_type(node.value)
        struct = self._resolve_struct_from_type(
            base_type,
            node=node,
            unknown_message="field access requires a struct value",
        )
        if struct is None:
            if not isinstance(base_type, astx.StructType):
                self.context.diagnostics.add(
                    "field access requires a struct value, got "
                    f"{display_type_name(base_type)}",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
                )
            self._set_type(node, None)
            return

        field_index = struct.field_indices.get(node.field_name)
        if field_index is None or field_index >= len(struct.fields):
            self.context.diagnostics.add(
                f"struct '{struct.name}' has no field '{node.field_name}'",
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_FIELD_ACCESS,
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
    def visit(self, node: astx.BufferViewDescriptor) -> None:
        """
        title: Visit BufferViewDescriptor nodes.
        parameters:
          node:
            type: astx.BufferViewDescriptor
        """
        for error in validate_buffer_view_metadata(node.metadata):
            self.context.diagnostics.add(error, node=node)
        self._semantic(node).extras[BUFFER_VIEW_METADATA_EXTRA] = node.metadata
        if node.type_.element_type is not None:
            self._semantic(node).extras[BUFFER_VIEW_ELEMENT_TYPE_EXTRA] = (
                node.type_.element_type
            )
        self._set_type(node, node.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewIndex) -> None:
        """
        title: Visit BufferViewIndex nodes.
        parameters:
          node:
            type: astx.BufferViewIndex
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        element_type = self._validate_buffer_view_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=False,
        )
        if element_type is not None:
            node.type_ = element_type
        self._set_type(node, element_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewStore) -> None:
        """
        title: Visit BufferViewStore nodes.
        parameters:
          node:
            type: astx.BufferViewStore
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        self.visit(node.value)
        element_type = self._validate_buffer_view_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=True,
        )
        if element_type is not None:
            validate_assignment(
                self.context.diagnostics,
                target_name="buffer view element",
                target_type=element_type,
                value_type=self._expr_type(node.value),
                node=node,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewWrite) -> None:
        """
        title: Visit BufferViewWrite nodes.
        parameters:
          node:
            type: astx.BufferViewWrite
        """
        if node.byte_offset < 0:
            self.context.diagnostics.add(
                "buffer view write byte_offset must be non-negative",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        self.visit(node.view)
        view_type = self._expr_type(node.view)
        if not isinstance(view_type, astx.BufferViewType):
            self.context.diagnostics.add(
                "buffer view write requires a BufferViewType view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        view_metadata = self._static_buffer_view_metadata(node.view)
        if view_metadata is not None and buffer_view_is_readonly(
            view_metadata.flags
        ):
            self.context.diagnostics.add(
                "cannot write through a readonly buffer view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        self.visit(node.value)
        value_type = self._expr_type(node.value)
        if (
            not is_integer_type(value_type)
            or bit_width(value_type) != RAW_BUFFER_BYTE_BITS
        ):
            self.context.diagnostics.add(
                "buffer view raw writes require an 8-bit integer value",
                node=node.value,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewRetain) -> None:
        """
        title: Visit BufferViewRetain nodes.
        parameters:
          node:
            type: astx.BufferViewRetain
        """
        self.visit(node.view)
        if not isinstance(self._expr_type(node.view), astx.BufferViewType):
            self.context.diagnostics.add(
                "buffer retain requires a BufferViewType view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._validate_buffer_lifetime_operation(
            node=node,
            view=node.view,
            operation="retain",
        )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.BufferViewRelease) -> None:
        """
        title: Visit BufferViewRelease nodes.
        parameters:
          node:
            type: astx.BufferViewRelease
        """
        self.visit(node.view)
        if not isinstance(self._expr_type(node.view), astx.BufferViewType):
            self.context.diagnostics.add(
                "buffer release requires a BufferViewType view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._validate_buffer_lifetime_operation(
            node=node,
            view=node.view,
            operation="release",
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

# mypy: disable-error-code=no-redef
# mypy: disable-error-code=untyped-decorator

"""
title: Expression buffer-view visitors.
summary: >-
  Handle buffer-view descriptors, indexing, writes, and lifetime helper
  expressions using the shared array-and-buffer support mixin.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.handlers._expressions.array_buffer_support import (
    ExpressionArrayBufferSupportVisitorMixin,
)
from irx.analysis.handlers.base import SemanticAnalyzerCore
from irx.analysis.types import bit_width, is_integer_type
from irx.analysis.validation import validate_assignment
from irx.buffer import (
    BUFFER_VIEW_ELEMENT_TYPE_EXTRA,
    BUFFER_VIEW_METADATA_EXTRA,
    buffer_view_is_readonly,
    validate_buffer_view_metadata,
)
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked

RAW_BUFFER_BYTE_BITS = 8


@typechecked
class ExpressionBufferViewVisitorMixin(
    ExpressionArrayBufferSupportVisitorMixin,
):
    """
    title: Expression buffer-view visitors.
    """

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

# mypy: disable-error-code=no-redef
# mypy: disable-error-code=untyped-decorator

"""
title: Expression Tensor visitors.
summary: >-
  Handle tensor literals, views, indexing, and lifetime helper expressions
  using the shared tensor-and-buffer support mixin.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.handlers._expressions.tensor_buffer_support import (
    ExpressionTensorBufferSupportVisitorMixin,
)
from irx.analysis.handlers.base import SemanticAnalyzerCore
from irx.analysis.validation import validate_assignment
from irx.buffer import (
    BUFFER_FLAG_VALIDITY_BITMAP,
    BufferMutability,
    BufferOwnership,
    buffer_view_flags,
    buffer_view_has_validity_bitmap,
    buffer_view_is_readonly,
    buffer_view_ownership,
)
from irx.builtins.collections.tensor import (
    TENSOR_ELEMENT_TYPE_EXTRA,
    TENSOR_FLAGS_EXTRA,
    TENSOR_LAYOUT_EXTRA,
    TensorLayout,
    tensor_byte_bounds,
    tensor_default_strides,
    tensor_element_count,
    tensor_element_size_bytes,
    tensor_is_c_contiguous,
    tensor_is_f_contiguous,
    validate_tensor_layout,
)
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ExpressionTensorVisitorMixin(ExpressionTensorBufferSupportVisitorMixin):
    """
    title: Expression Tensor visitors.
    """

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorLiteral) -> None:
        """
        title: Visit TensorLiteral nodes.
        parameters:
          node:
            type: astx.TensorLiteral
        """
        for item in node.values:
            self.visit(item)
            validate_assignment(
                self.context.diagnostics,
                target_name="tensor element",
                target_type=node.element_type,
                value_type=self._expr_type(item),
                node=item,
            )

        shape = tuple(node.shape)
        element_size_bytes = tensor_element_size_bytes(node.element_type)
        if element_size_bytes is None:
            if isinstance(node.element_type, astx.Boolean):
                self.context.diagnostics.add(
                    "bool tensors are not supported because bit-packed Arrow "
                    "values are not buffer-view compatible",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )
            else:
                self.context.diagnostics.add(
                    "tensor literals require a fixed-width numeric element "
                    "type",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )

        if node.strides is None:
            if element_size_bytes is None or any(dim < 0 for dim in shape):
                strides = tuple(0 for _ in shape)
            else:
                strides = tensor_default_strides(shape, element_size_bytes)
        else:
            strides = tuple(node.strides)

        layout = TensorLayout(
            shape=shape,
            strides=strides,
            offset_bytes=node.offset_bytes,
        )
        for error in validate_tensor_layout(layout):
            self.context.diagnostics.add(
                error,
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        expected_value_count = tensor_element_count(layout)
        if len(node.values) != expected_value_count:
            self.context.diagnostics.add(
                "tensor literal value count must match the shape extent",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        if element_size_bytes is not None:
            bounds = tensor_byte_bounds(layout)
            if bounds is not None:
                minimum, maximum = bounds
                storage_bytes = len(node.values) * element_size_bytes
                if minimum < 0 or maximum + element_size_bytes > storage_bytes:
                    self.context.diagnostics.add(
                        "tensor literal layout exceeds compact backing "
                        "storage",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )

            flags = buffer_view_flags(
                BufferOwnership.EXTERNAL_OWNER,
                BufferMutability.READONLY,
                c_contiguous=tensor_is_c_contiguous(
                    layout,
                    element_size_bytes,
                ),
                f_contiguous=tensor_is_f_contiguous(
                    layout,
                    element_size_bytes,
                ),
            )
            self._semantic(node).extras[TENSOR_FLAGS_EXTRA] = flags

        self._semantic(node).extras[TENSOR_LAYOUT_EXTRA] = layout
        self._semantic(node).extras[TENSOR_ELEMENT_TYPE_EXTRA] = (
            node.element_type
        )
        node.type_ = astx.TensorType(node.element_type)
        self._set_type(node, node.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorView) -> None:
        """
        title: Visit TensorView nodes.
        parameters:
          node:
            type: astx.TensorView
        """
        self.visit(node.base)
        base_type = self._expr_type(node.base)
        if not isinstance(base_type, astx.TensorType):
            self.context.diagnostics.add(
                "tensor views require a TensorType base",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        base_layout = self._static_tensor_layout(node.base)
        if base_layout is None:
            self.context.diagnostics.add(
                "tensor views require static base layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        element_type = self._static_tensor_element_type(node.base)
        if element_type is None:
            self.context.diagnostics.add(
                "tensor views require a known element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        element_size_bytes = tensor_element_size_bytes(element_type)
        if element_type is not None and element_size_bytes is None:
            self.context.diagnostics.add(
                "tensor views require a fixed-width numeric element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        shape = tuple(node.shape)
        if node.strides is None:
            if (
                base_layout is None
                or element_size_bytes is None
                or any(dim < 0 for dim in shape)
            ):
                strides = tuple(0 for _ in shape)
            else:
                if not tensor_is_c_contiguous(
                    base_layout,
                    element_size_bytes,
                ):
                    self.context.diagnostics.add(
                        "tensor views without explicit strides require a "
                        "C-contiguous base",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )
                expected_count = 1
                for dim in shape:
                    expected_count *= dim
                if expected_count != tensor_element_count(base_layout):
                    self.context.diagnostics.add(
                        "tensor reshape views require the same element count "
                        "as the base",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )
                strides = tensor_default_strides(shape, element_size_bytes)
        else:
            strides = tuple(node.strides)

        base_offset_bytes = (
            base_layout.offset_bytes if base_layout is not None else 0
        )
        layout = TensorLayout(
            shape=shape,
            strides=strides,
            offset_bytes=base_offset_bytes + node.offset_bytes,
        )
        for error in validate_tensor_layout(layout):
            self.context.diagnostics.add(
                error,
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        if base_layout is not None and element_size_bytes is not None:
            base_bounds = tensor_byte_bounds(base_layout)
            view_bounds = tensor_byte_bounds(layout)
            if (
                base_bounds is not None
                and view_bounds is not None
                and (
                    view_bounds[0] < base_bounds[0]
                    or view_bounds[1] > base_bounds[1]
                )
            ):
                self.context.diagnostics.add(
                    "tensor view exceeds base storage bounds",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )

        base_flags = self._static_tensor_flags(node.base)
        if base_flags is None:
            flags = buffer_view_flags(
                BufferOwnership.EXTERNAL_OWNER,
                BufferMutability.READONLY,
                c_contiguous=(
                    False
                    if element_size_bytes is None
                    else tensor_is_c_contiguous(layout, element_size_bytes)
                ),
                f_contiguous=(
                    False
                    if element_size_bytes is None
                    else tensor_is_f_contiguous(layout, element_size_bytes)
                ),
            )
        else:
            ownership = (
                buffer_view_ownership(base_flags)
                or BufferOwnership.EXTERNAL_OWNER
            )
            mutability = (
                BufferMutability.READONLY
                if buffer_view_is_readonly(base_flags)
                else BufferMutability.WRITABLE
            )
            flags = buffer_view_flags(
                ownership,
                mutability,
                c_contiguous=(
                    False
                    if element_size_bytes is None
                    else tensor_is_c_contiguous(layout, element_size_bytes)
                ),
                f_contiguous=(
                    False
                    if element_size_bytes is None
                    else tensor_is_f_contiguous(layout, element_size_bytes)
                ),
            )
            if buffer_view_has_validity_bitmap(base_flags):
                flags |= BUFFER_FLAG_VALIDITY_BITMAP

        if element_type is not None:
            self._semantic(node).extras[TENSOR_ELEMENT_TYPE_EXTRA] = (
                element_type
            )
            node.type_ = astx.TensorType(element_type)
        self._semantic(node).extras[TENSOR_LAYOUT_EXTRA] = layout
        self._semantic(node).extras[TENSOR_FLAGS_EXTRA] = flags
        self._set_type(node, node.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorIndex) -> None:
        """
        title: Visit TensorIndex nodes.
        parameters:
          node:
            type: astx.TensorIndex
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        element_type = self._validate_tensor_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=False,
        )
        if element_type is not None:
            node.type_ = element_type
        self._set_type(node, element_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorStore) -> None:
        """
        title: Visit TensorStore nodes.
        parameters:
          node:
            type: astx.TensorStore
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        self.visit(node.value)
        element_type = self._validate_tensor_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=True,
        )
        if element_type is not None:
            validate_assignment(
                self.context.diagnostics,
                target_name="tensor element",
                target_type=element_type,
                value_type=self._expr_type(node.value),
                node=node,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorNDim) -> None:
        """
        title: Visit TensorNDim nodes.
        parameters:
          node:
            type: astx.TensorNDim
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.TensorType):
            self.context.diagnostics.add(
                "tensor ndim requires a TensorType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorShape) -> None:
        """
        title: Visit TensorShape nodes.
        parameters:
          node:
            type: astx.TensorShape
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.TensorType):
            self.context.diagnostics.add(
                "tensor shape queries require a TensorType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        layout = self._static_tensor_layout(node.base)
        if layout is None:
            self.context.diagnostics.add(
                "tensor shape queries require static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif node.axis < 0 or node.axis >= layout.ndim:
            self.context.diagnostics.add(
                "tensor shape axis is out of bounds",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorStride) -> None:
        """
        title: Visit TensorStride nodes.
        parameters:
          node:
            type: astx.TensorStride
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.TensorType):
            self.context.diagnostics.add(
                "tensor stride queries require a TensorType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        layout = self._static_tensor_layout(node.base)
        if layout is None:
            self.context.diagnostics.add(
                "tensor stride queries require static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif node.axis < 0 or node.axis >= layout.ndim:
            self.context.diagnostics.add(
                "tensor stride axis is out of bounds",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorElementCount) -> None:
        """
        title: Visit TensorElementCount nodes.
        parameters:
          node:
            type: astx.TensorElementCount
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.TensorType):
            self.context.diagnostics.add(
                "tensor element_count requires a TensorType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        if self._static_tensor_layout(node.base) is None:
            self.context.diagnostics.add(
                "tensor element_count requires static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorByteOffset) -> None:
        """
        title: Visit TensorByteOffset nodes.
        parameters:
          node:
            type: astx.TensorByteOffset
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        self._validate_tensor_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=False,
        )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorRetain) -> None:
        """
        title: Visit TensorRetain nodes.
        parameters:
          node:
            type: astx.TensorRetain
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.TensorType):
            self.context.diagnostics.add(
                "tensor retain requires a TensorType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._validate_tensor_lifetime_operation(
            node=node,
            view=node.base,
            operation="retain",
        )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.TensorRelease) -> None:
        """
        title: Visit TensorRelease nodes.
        parameters:
          node:
            type: astx.TensorRelease
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.TensorType):
            self.context.diagnostics.add(
                "tensor release requires a TensorType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._validate_tensor_lifetime_operation(
            node=node,
            view=node.base,
            operation="release",
        )
        self._set_type(node, astx.Int32())

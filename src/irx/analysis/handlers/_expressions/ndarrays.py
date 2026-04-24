# mypy: disable-error-code=no-redef
# mypy: disable-error-code=untyped-decorator

"""
title: Expression NDArray visitors.
summary: >-
  Handle ndarray literals, views, indexing, and lifetime helper expressions
  using the shared array-and-buffer support mixin.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.handlers._expressions.array_buffer_support import (
    ExpressionArrayBufferSupportVisitorMixin,
)
from irx.analysis.handlers.base import SemanticAnalyzerCore
from irx.analysis.types import is_integer_type
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
from irx.builtins.collections.array import (
    NDARRAY_ELEMENT_TYPE_EXTRA,
    NDARRAY_FLAGS_EXTRA,
    NDARRAY_LAYOUT_EXTRA,
    NDArrayLayout,
    ndarray_byte_bounds,
    ndarray_default_strides,
    ndarray_element_count,
    ndarray_element_size_bytes,
    ndarray_is_c_contiguous,
    ndarray_is_f_contiguous,
    validate_ndarray_layout,
)
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ExpressionNDArrayVisitorMixin(ExpressionArrayBufferSupportVisitorMixin):
    """
    title: Expression NDArray visitors.
    """

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.ArrayInt32ArrayLength) -> None:
        """
        title: Visit ArrayInt32ArrayLength nodes.
        parameters:
          node:
            type: astx.ArrayInt32ArrayLength
        """
        for item in node.values:
            self.visit(item)
            if not is_integer_type(self._expr_type(item)):
                self.context.diagnostics.add(
                    "Array helper supports only integer expressions",
                    node=item,
                )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayLiteral) -> None:
        """
        title: Visit NDArrayLiteral nodes.
        parameters:
          node:
            type: astx.NDArrayLiteral
        """
        for item in node.values:
            self.visit(item)
            validate_assignment(
                self.context.diagnostics,
                target_name="ndarray element",
                target_type=node.element_type,
                value_type=self._expr_type(item),
                node=item,
            )

        shape = tuple(node.shape)
        element_size_bytes = ndarray_element_size_bytes(node.element_type)
        if element_size_bytes is None:
            if isinstance(node.element_type, astx.Boolean):
                self.context.diagnostics.add(
                    "bool ndarrays are not supported because bit-packed Arrow "
                    "values are not buffer-view compatible",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )
            else:
                self.context.diagnostics.add(
                    "ndarray literals require a fixed-width numeric element "
                    "type",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )

        if node.strides is None:
            if element_size_bytes is None or any(dim < 0 for dim in shape):
                strides = tuple(0 for _ in shape)
            else:
                strides = ndarray_default_strides(shape, element_size_bytes)
        else:
            strides = tuple(node.strides)

        layout = NDArrayLayout(
            shape=shape,
            strides=strides,
            offset_bytes=node.offset_bytes,
        )
        for error in validate_ndarray_layout(layout):
            self.context.diagnostics.add(
                error,
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        expected_value_count = ndarray_element_count(layout)
        if len(node.values) != expected_value_count:
            self.context.diagnostics.add(
                "ndarray literal value count must match the shape extent",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        if element_size_bytes is not None:
            bounds = ndarray_byte_bounds(layout)
            if bounds is not None:
                minimum, maximum = bounds
                storage_bytes = len(node.values) * element_size_bytes
                if minimum < 0 or maximum + element_size_bytes > storage_bytes:
                    self.context.diagnostics.add(
                        "ndarray literal layout exceeds compact backing "
                        "storage",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )

            flags = buffer_view_flags(
                BufferOwnership.EXTERNAL_OWNER,
                BufferMutability.READONLY,
                c_contiguous=ndarray_is_c_contiguous(
                    layout,
                    element_size_bytes,
                ),
                f_contiguous=ndarray_is_f_contiguous(
                    layout,
                    element_size_bytes,
                ),
            )
            self._semantic(node).extras[NDARRAY_FLAGS_EXTRA] = flags

        self._semantic(node).extras[NDARRAY_LAYOUT_EXTRA] = layout
        self._semantic(node).extras[NDARRAY_ELEMENT_TYPE_EXTRA] = (
            node.element_type
        )
        node.type_ = astx.NDArrayType(node.element_type)
        self._set_type(node, node.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayView) -> None:
        """
        title: Visit NDArrayView nodes.
        parameters:
          node:
            type: astx.NDArrayView
        """
        self.visit(node.base)
        base_type = self._expr_type(node.base)
        if not isinstance(base_type, astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray views require a NDArrayType base",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        base_layout = self._static_ndarray_layout(node.base)
        if base_layout is None:
            self.context.diagnostics.add(
                "ndarray views require static base layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        element_type = self._static_ndarray_element_type(node.base)
        if element_type is None:
            self.context.diagnostics.add(
                "ndarray views require a known element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        element_size_bytes = ndarray_element_size_bytes(element_type)
        if element_type is not None and element_size_bytes is None:
            self.context.diagnostics.add(
                "ndarray views require a fixed-width numeric element type",
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
                if not ndarray_is_c_contiguous(
                    base_layout,
                    element_size_bytes,
                ):
                    self.context.diagnostics.add(
                        "ndarray views without explicit strides require a "
                        "C-contiguous base",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )
                expected_count = 1
                for dim in shape:
                    expected_count *= dim
                if expected_count != ndarray_element_count(base_layout):
                    self.context.diagnostics.add(
                        "ndarray reshape views require the same element count "
                        "as the base",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )
                strides = ndarray_default_strides(shape, element_size_bytes)
        else:
            strides = tuple(node.strides)

        base_offset_bytes = (
            base_layout.offset_bytes if base_layout is not None else 0
        )
        layout = NDArrayLayout(
            shape=shape,
            strides=strides,
            offset_bytes=base_offset_bytes + node.offset_bytes,
        )
        for error in validate_ndarray_layout(layout):
            self.context.diagnostics.add(
                error,
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        if base_layout is not None and element_size_bytes is not None:
            base_bounds = ndarray_byte_bounds(base_layout)
            view_bounds = ndarray_byte_bounds(layout)
            if (
                base_bounds is not None
                and view_bounds is not None
                and (
                    view_bounds[0] < base_bounds[0]
                    or view_bounds[1] > base_bounds[1]
                )
            ):
                self.context.diagnostics.add(
                    "ndarray view exceeds base storage bounds",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )

        base_flags = self._static_ndarray_flags(node.base)
        if base_flags is None:
            flags = buffer_view_flags(
                BufferOwnership.EXTERNAL_OWNER,
                BufferMutability.READONLY,
                c_contiguous=(
                    False
                    if element_size_bytes is None
                    else ndarray_is_c_contiguous(layout, element_size_bytes)
                ),
                f_contiguous=(
                    False
                    if element_size_bytes is None
                    else ndarray_is_f_contiguous(layout, element_size_bytes)
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
                    else ndarray_is_c_contiguous(layout, element_size_bytes)
                ),
                f_contiguous=(
                    False
                    if element_size_bytes is None
                    else ndarray_is_f_contiguous(layout, element_size_bytes)
                ),
            )
            if buffer_view_has_validity_bitmap(base_flags):
                flags |= BUFFER_FLAG_VALIDITY_BITMAP

        if element_type is not None:
            self._semantic(node).extras[NDARRAY_ELEMENT_TYPE_EXTRA] = (
                element_type
            )
            node.type_ = astx.NDArrayType(element_type)
        self._semantic(node).extras[NDARRAY_LAYOUT_EXTRA] = layout
        self._semantic(node).extras[NDARRAY_FLAGS_EXTRA] = flags
        self._set_type(node, node.type_)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayIndex) -> None:
        """
        title: Visit NDArrayIndex nodes.
        parameters:
          node:
            type: astx.NDArrayIndex
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        element_type = self._validate_ndarray_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=False,
        )
        if element_type is not None:
            node.type_ = element_type
        self._set_type(node, element_type)

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayStore) -> None:
        """
        title: Visit NDArrayStore nodes.
        parameters:
          node:
            type: astx.NDArrayStore
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        self.visit(node.value)
        element_type = self._validate_ndarray_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=True,
        )
        if element_type is not None:
            validate_assignment(
                self.context.diagnostics,
                target_name="ndarray element",
                target_type=element_type,
                value_type=self._expr_type(node.value),
                node=node,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayNDim) -> None:
        """
        title: Visit NDArrayNDim nodes.
        parameters:
          node:
            type: astx.NDArrayNDim
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray ndim requires a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayShape) -> None:
        """
        title: Visit NDArrayShape nodes.
        parameters:
          node:
            type: astx.NDArrayShape
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray shape queries require a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        layout = self._static_ndarray_layout(node.base)
        if layout is None:
            self.context.diagnostics.add(
                "ndarray shape queries require static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif node.axis < 0 or node.axis >= layout.ndim:
            self.context.diagnostics.add(
                "ndarray shape axis is out of bounds",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayStride) -> None:
        """
        title: Visit NDArrayStride nodes.
        parameters:
          node:
            type: astx.NDArrayStride
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray stride queries require a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        layout = self._static_ndarray_layout(node.base)
        if layout is None:
            self.context.diagnostics.add(
                "ndarray stride queries require static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif node.axis < 0 or node.axis >= layout.ndim:
            self.context.diagnostics.add(
                "ndarray stride axis is out of bounds",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayElementCount) -> None:
        """
        title: Visit NDArrayElementCount nodes.
        parameters:
          node:
            type: astx.NDArrayElementCount
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray element_count requires a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        if self._static_ndarray_layout(node.base) is None:
            self.context.diagnostics.add(
                "ndarray element_count requires static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayByteOffset) -> None:
        """
        title: Visit NDArrayByteOffset nodes.
        parameters:
          node:
            type: astx.NDArrayByteOffset
        """
        self.visit(node.base)
        for index in node.indices:
            self.visit(index)
        self._validate_ndarray_index_operation(
            node=node,
            base=node.base,
            indices=node.indices,
            is_store=False,
        )
        self._set_type(node, astx.Int64())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayRetain) -> None:
        """
        title: Visit NDArrayRetain nodes.
        parameters:
          node:
            type: astx.NDArrayRetain
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray retain requires a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._validate_ndarray_lifetime_operation(
            node=node,
            view=node.base,
            operation="retain",
        )
        self._set_type(node, astx.Int32())

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.NDArrayRelease) -> None:
        """
        title: Visit NDArrayRelease nodes.
        parameters:
          node:
            type: astx.NDArrayRelease
        """
        self.visit(node.base)
        if not isinstance(self._expr_type(node.base), astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray release requires a NDArrayType value",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        self._validate_ndarray_lifetime_operation(
            node=node,
            view=node.base,
            operation="release",
        )
        self._set_type(node, astx.Int32())

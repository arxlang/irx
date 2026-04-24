"""
title: Array and buffer metadata helpers.
summary: >-
  Provide static ndarray and buffer-view metadata lookup plus shared validation
  used by the array-and-buffer expression visitors.
"""

from __future__ import annotations

from irx import astx
from irx.analysis.handlers.base import SemanticVisitorMixinBase
from irx.analysis.types import (
    bit_width,
    is_boolean_type,
    is_float_type,
    is_integer_type,
)
from irx.buffer import (
    BUFFER_VIEW_ELEMENT_TYPE_EXTRA,
    BUFFER_VIEW_METADATA_EXTRA,
    BufferOwnership,
    BufferViewMetadata,
    buffer_view_is_readonly,
    buffer_view_ownership,
)
from irx.builtins.collections.array import (
    NDARRAY_ELEMENT_TYPE_EXTRA,
    NDARRAY_FLAGS_EXTRA,
    NDARRAY_LAYOUT_EXTRA,
    NDArrayLayout,
    ndarray_element_size_bytes,
)
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ExpressionArrayBufferSupportVisitorMixin(SemanticVisitorMixinBase):
    """
    title: Array and buffer metadata helpers.
    """

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

    def _static_ndarray_layout(
        self,
        node: astx.AST,
    ) -> NDArrayLayout | None:
        """
        title: >-
          Return static ndarray layout metadata when analysis can prove it.
        parameters:
          node:
            type: astx.AST
        returns:
          type: NDArrayLayout | None
        """
        semantic = self._semantic(node)
        layout = semantic.extras.get(NDARRAY_LAYOUT_EXTRA)
        if isinstance(layout, NDArrayLayout):
            return layout

        symbol = semantic.resolved_symbol
        declaration = symbol.declaration if symbol is not None else None
        initializer = getattr(declaration, "value", None)
        if not isinstance(initializer, astx.AST):
            return None

        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        layout = initializer_extras.get(NDARRAY_LAYOUT_EXTRA)
        if isinstance(layout, NDArrayLayout):
            return layout
        return None

    def _static_ndarray_element_type(
        self,
        node: astx.AST,
    ) -> astx.DataType | None:
        """
        title: >-
          Return the scalar ndarray element type when analysis can prove it.
        parameters:
          node:
            type: astx.AST
        returns:
          type: astx.DataType | None
        """
        semantic = self._semantic(node)
        element_type = semantic.extras.get(NDARRAY_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type

        ndarray_type = self._expr_type(node)
        if (
            isinstance(ndarray_type, astx.NDArrayType)
            and ndarray_type.element_type is not None
        ):
            return ndarray_type.element_type

        symbol = semantic.resolved_symbol
        declaration = symbol.declaration if symbol is not None else None
        initializer = getattr(declaration, "value", None)
        if not isinstance(initializer, astx.AST):
            return None

        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        element_type = initializer_extras.get(NDARRAY_ELEMENT_TYPE_EXTRA)
        if isinstance(element_type, astx.DataType):
            return element_type

        initializer_type = getattr(
            initializer_semantic,
            "resolved_type",
            getattr(initializer, "type_", None),
        )
        if (
            isinstance(initializer_type, astx.NDArrayType)
            and initializer_type.element_type is not None
        ):
            return initializer_type.element_type
        return None

    def _static_ndarray_flags(
        self,
        node: astx.AST,
    ) -> int | None:
        """
        title: Return static NDArray flags when analysis can prove them.
        parameters:
          node:
            type: astx.AST
        returns:
          type: int | None
        """
        semantic = self._semantic(node)
        flags = semantic.extras.get(NDARRAY_FLAGS_EXTRA)
        if isinstance(flags, int):
            return flags

        symbol = semantic.resolved_symbol
        declaration = symbol.declaration if symbol is not None else None
        initializer = getattr(declaration, "value", None)
        if not isinstance(initializer, astx.AST):
            return None

        initializer_semantic = getattr(initializer, "semantic", None)
        initializer_extras = getattr(initializer_semantic, "extras", {})
        flags = initializer_extras.get(NDARRAY_FLAGS_EXTRA)
        if isinstance(flags, int):
            return flags
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

    def _validate_ndarray_index_operation(
        self,
        *,
        node: astx.AST,
        base: astx.AST,
        indices: list[astx.AST],
        is_store: bool,
    ) -> astx.DataType | None:
        """
        title: Validate one NDArray indexed access.
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
        if not isinstance(base_type, astx.NDArrayType):
            self.context.diagnostics.add(
                "ndarray indexing requires a NDArrayType base",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        layout = self._static_ndarray_layout(base)
        if layout is None:
            self.context.diagnostics.add(
                "ndarray indexing requires static layout metadata",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        elif len(indices) != layout.ndim:
            self.context.diagnostics.add(
                "ndarray indexing index count must match ndarray ndim",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
        else:
            for axis, index in enumerate(indices):
                extent = layout.shape[axis]
                static_index = self._static_integer_literal_value(index)
                if static_index is None:
                    continue
                if static_index < 0 or static_index >= extent:
                    self.context.diagnostics.add(
                        "ndarray index "
                        f"{axis} statically out of bounds for extent {extent}",
                        node=index,
                        code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                    )

        flags = self._static_ndarray_flags(base)
        if is_store and flags is not None and buffer_view_is_readonly(flags):
            self.context.diagnostics.add(
                "cannot write through a readonly ndarray view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

        for index in indices:
            index_type = self._expr_type(index)
            if not is_integer_type(index_type):
                self.context.diagnostics.add(
                    "ndarray indices must be integer typed",
                    node=index,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )
                continue
            if bit_width(index_type) > bit_width(astx.Int64()):
                self.context.diagnostics.add(
                    "ndarray indices must fit 64-bit stride arithmetic",
                    node=index,
                    code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
                )

        element_type = self._static_ndarray_element_type(base)
        if element_type is None:
            self.context.diagnostics.add(
                "ndarray indexing requires a known element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
            return None
        if ndarray_element_size_bytes(element_type) is None:
            self.context.diagnostics.add(
                "ndarray indexing requires a fixed-width numeric element type",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )
            return None

        self._semantic(node).extras[NDARRAY_ELEMENT_TYPE_EXTRA] = element_type
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

    def _validate_ndarray_lifetime_operation(
        self,
        *,
        node: astx.AST,
        view: astx.AST,
        operation: str,
    ) -> None:
        """
        title: Validate one explicit NDArray lifetime helper operation.
        parameters:
          node:
            type: astx.AST
          view:
            type: astx.AST
          operation:
            type: str
        """
        flags = self._static_ndarray_flags(view)
        if flags is None:
            return
        ownership = buffer_view_ownership(flags)
        if ownership is BufferOwnership.BORROWED:
            self.context.diagnostics.add(
                "ndarray "
                f"{operation} requires an owned or external-owner view",
                node=node,
                code=DiagnosticCodes.SEMANTIC_BUFFER_MISUSE,
            )

# mypy: disable-error-code=no-redef
# mypy: disable-error-code=attr-defined

"""
title: Generator lowering visitors for llvmliteir.
summary: >-
  Lower the initial IRx generator MVP as factory functions plus internal resume
  state machines consumed through the iterable lowering path.
"""

from __future__ import annotations

from typing import Any, cast

from llvmlite import ir

from irx import astx
from irx.analysis.resolved_nodes import (
    ResolvedGeneratorFunction,
    ResolvedIteration,
    ResolvedYield,
    SemanticFunction,
    SemanticSymbol,
)
from irx.builder.core import VisitorCore, semantic_symbol_key
from irx.builder.diagnostics import (
    raise_lowering_error,
    raise_lowering_internal_error,
    require_lowered_value,
    require_semantic_metadata,
)
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked

GENERATOR_FRAME_ALLOCATION_BYTES = 4096
GENERATOR_STATE_FIELD_INDEX = 0
GENERATOR_EXHAUSTED_FIELD_INDEX = 1


@typechecked
def _is_yield_node(node: astx.AST) -> bool:
    """
    title: Return whether one node is a generator suspension site.
    parameters:
      node:
        type: astx.AST
    returns:
      type: bool
    """
    return isinstance(node, (astx.YieldExpr, astx.YieldStmt))


@typechecked
class GeneratorVisitorMixin(VisitorMixinBase):
    def _semantic_generator_function(
        self,
        node: astx.FunctionDef,
    ) -> ResolvedGeneratorFunction | None:
        """
        title: Return generator metadata attached to one function definition.
        parameters:
          node:
            type: astx.FunctionDef
        returns:
          type: ResolvedGeneratorFunction | None
        """
        semantic = getattr(node, "semantic", None)
        generator = getattr(semantic, "resolved_generator_function", None)
        if isinstance(generator, ResolvedGeneratorFunction):
            return generator
        return None

    def _semantic_yield_resolution(self, node: astx.AST) -> ResolvedYield:
        """
        title: Return the semantic yield metadata for one yield node.
        parameters:
          node:
            type: astx.AST
        returns:
          type: ResolvedYield
        """
        semantic = getattr(node, "semantic", None)
        resolution = getattr(semantic, "resolved_yield", None)
        return require_semantic_metadata(
            cast(ResolvedYield | None, resolution),
            node=node,
            metadata="resolved_yield",
            context="yield lowering",
        )

    def _generator_value_type(self) -> ir.LiteralStructType:
        """
        title: Return the low-level generator object type.
        returns:
          type: ir.LiteralStructType
        """
        return ir.LiteralStructType(
            [
                self._llvm.OPAQUE_POINTER_TYPE,
                self._llvm.OPAQUE_POINTER_TYPE,
            ]
        )

    def _generator_yield_type(
        self,
        generator: ResolvedGeneratorFunction,
    ) -> ir.Type:
        """
        title: Return the LLVM element type yielded by one generator.
        parameters:
          generator:
            type: ResolvedGeneratorFunction
        returns:
          type: ir.Type
        """
        llvm_type = self._llvm_type_for_ast_type(generator.yield_type)
        if llvm_type is None or isinstance(llvm_type, ir.VoidType):
            raise_lowering_error(
                "generator yield type is not lowerable",
                node=generator.function.prototype,
                code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
            )
        return llvm_type

    def _malloc_function(self) -> ir.Function:
        """
        title: Return or declare the C malloc function.
        returns:
          type: ir.Function
        """
        existing = self._llvm.module.globals.get("malloc")
        if existing is not None:
            if not isinstance(existing, ir.Function):
                raise_lowering_internal_error(
                    "global 'malloc' exists but is not a function",
                    node=None,
                )
            return existing
        size_type = self._llvm.SIZE_T_TYPE or self._llvm.INT64_TYPE
        malloc_type = ir.FunctionType(
            self._llvm.OPAQUE_POINTER_TYPE,
            [size_type],
        )
        return ir.Function(self._llvm.module, malloc_type, "malloc")

    def _generator_function_name(self, function: SemanticFunction) -> str:
        """
        title: Return the lowered factory symbol name for a generator function.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: str
        """
        return self.llvm_function_name_for_node(
            function.prototype,
            function.name,
        )

    def _generator_resume_name(self, function: SemanticFunction) -> str:
        """
        title: Return the internal resume function name.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: str
        """
        return f"{self._generator_function_name(function)}.__resume"

    def _collect_generator_local_symbols(
        self,
        node: astx.AST,
    ) -> tuple[SemanticSymbol, ...]:
        """
        title: Collect local variable symbols stored in a generator frame.
        parameters:
          node:
            type: astx.AST
        returns:
          type: tuple[SemanticSymbol, Ellipsis]
        """
        symbols: list[SemanticSymbol] = []
        semantic = getattr(node, "semantic", None)
        symbol = getattr(semantic, "resolved_symbol", None)
        if isinstance(
            node,
            (astx.InlineVariableDeclaration, astx.VariableDeclaration),
        ) and isinstance(symbol, SemanticSymbol):
            symbols.append(symbol)

        if isinstance(node, astx.FunctionDef):
            return ()
        if isinstance(node, astx.Block):
            for child in node.nodes:
                symbols.extend(self._collect_generator_local_symbols(child))
            return tuple(symbols)

        for attribute_name in ("body", "then", "else_"):
            child = getattr(node, attribute_name, None)
            if isinstance(child, astx.Block):
                symbols.extend(self._collect_generator_local_symbols(child))
        return tuple(symbols)

    def _generator_frame_layout(
        self,
        generator: ResolvedGeneratorFunction,
    ) -> tuple[ir.IdentifiedStructType, dict[str, int]]:
        """
        title: Return the frame type and symbol-slot map for one generator.
        parameters:
          generator:
            type: ResolvedGeneratorFunction
        returns:
          type: tuple[ir.IdentifiedStructType, dict[str, int]]
        """
        function = generator.function
        existing = self._generator_frame_types.get(function.symbol_id)
        existing_slots = self._generator_frame_slots_by_symbol_id.get(
            function.symbol_id
        )
        if existing is not None and existing_slots is not None:
            return existing, existing_slots

        frame_name = f"{self._generator_function_name(function)}.__frame"
        frame_type = self._llvm.module.context.get_identified_type(frame_name)
        field_types: list[ir.Type] = [
            self._llvm.INT32_TYPE,
            self._llvm.BOOLEAN_TYPE,
        ]
        slots: dict[str, int] = {}
        local_symbols = (
            self._collect_generator_local_symbols(function.definition.body)
            if function.definition is not None
            else ()
        )
        for symbol in (*function.args, *local_symbols):
            if symbol.symbol_id in slots:
                continue
            llvm_type = self._llvm_type_for_ast_type(symbol.type_)
            if llvm_type is None:
                raise_lowering_error(
                    f"cannot lower generator frame field '{symbol.name}'",
                    node=symbol.declaration,
                    code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
                )
            slots[symbol.symbol_id] = len(field_types)
            field_types.append(llvm_type)

        if frame_type.is_opaque:
            frame_type.set_body(*field_types)
        self._generator_frame_types[function.symbol_id] = frame_type
        self._generator_frame_slots_by_symbol_id[function.symbol_id] = slots
        return frame_type, slots

    def _generator_field_address(
        self,
        frame_ptr: ir.Value,
        field_index: int,
        *,
        name: str,
    ) -> ir.Value:
        """
        title: Return the address of one generator frame field.
        parameters:
          frame_ptr:
            type: ir.Value
          field_index:
            type: int
          name:
            type: str
        returns:
          type: ir.Value
        """
        return self._llvm.ir_builder.gep(
            frame_ptr,
            [
                ir.Constant(self._llvm.INT32_TYPE, 0),
                ir.Constant(self._llvm.INT32_TYPE, field_index),
            ],
            inbounds=True,
            name=name,
        )

    def _declare_generator_resume(
        self,
        generator: ResolvedGeneratorFunction,
    ) -> ir.Function:
        """
        title: Declare or return the internal resume function.
        parameters:
          generator:
            type: ResolvedGeneratorFunction
        returns:
          type: ir.Function
        """
        function = generator.function
        existing = self._generator_resume_functions.get(function.symbol_id)
        if existing is not None:
            return existing
        yield_type = self._generator_yield_type(generator)
        resume_type = ir.FunctionType(
            self._llvm.BOOLEAN_TYPE,
            [self._llvm.OPAQUE_POINTER_TYPE, yield_type.as_pointer()],
        )
        resume = ir.Function(
            self._llvm.module,
            resume_type,
            self._generator_resume_name(function),
        )
        resume.linkage = "internal"
        resume.args[0].name = "frame"
        resume.args[1].name = "out"
        self._generator_resume_functions[function.symbol_id] = resume
        return resume

    def _bind_generator_frame_symbols(
        self,
        frame_ptr: ir.Value,
        slots: dict[str, int],
    ) -> None:
        """
        title: Bind generator frame fields as variable storage addresses.
        parameters:
          frame_ptr:
            type: ir.Value
          slots:
            type: dict[str, int]
        """
        for symbol_id, field_index in slots.items():
            self.named_values[symbol_id] = self._generator_field_address(
                frame_ptr,
                field_index,
                name=f"{symbol_id.replace(':', '_')}_addr",
            )

    def _emit_generator_stop(self, node: astx.AST | None = None) -> None:
        """
        title: Emit generator exhaustion and return false from resume.
        parameters:
          node:
            type: astx.AST | None
        """
        frame_ptr = self._current_generator_frame_ptr
        if frame_ptr is None:
            raise_lowering_internal_error(
                "generator stop emitted outside resume function",
                node=node,
            )
        exhausted_addr = self._generator_field_address(
            frame_ptr,
            GENERATOR_EXHAUSTED_FIELD_INDEX,
            name="generator_exhausted_addr",
        )
        self._llvm.ir_builder.store(
            ir.Constant(self._llvm.BOOLEAN_TYPE, 1),
            exhausted_addr,
        )
        self._llvm.ir_builder.ret(ir.Constant(self._llvm.BOOLEAN_TYPE, 0))

    def _lower_generator_suspend(
        self,
        node: astx.AST,
        value: astx.AST | None,
    ) -> None:
        """
        title: Lower one yield site as a resume suspension.
        parameters:
          node:
            type: astx.AST
          value:
            type: astx.AST | None
        """
        resolution = self._semantic_yield_resolution(node)
        frame_ptr = self._current_generator_frame_ptr
        out_ptr = self._current_generator_out_ptr
        next_state = self._current_generator_next_state
        if frame_ptr is None or out_ptr is None or next_state is None:
            raise_lowering_internal_error(
                "yield lowered outside a generator resume function",
                node=node,
            )

        if value is not None:
            self.visit_child(value)
            yielded = require_lowered_value(
                safe_pop(self.result_stack),
                node=value,
                context="yield expression",
            )
            yielded = self._cast_ast_value(
                yielded,
                source_type=self._resolved_ast_type(value),
                target_type=resolution.expected_type,
            )
            self._llvm.ir_builder.store(yielded, out_ptr)

        state_addr = self._generator_field_address(
            frame_ptr,
            GENERATOR_STATE_FIELD_INDEX,
            name="generator_state_addr",
        )
        self._llvm.ir_builder.store(
            ir.Constant(self._llvm.INT32_TYPE, next_state),
            state_addr,
        )
        self._llvm.ir_builder.ret(ir.Constant(self._llvm.BOOLEAN_TYPE, 1))

    def _lower_generator_body_from_state(
        self,
        body: astx.Block,
        *,
        state_index: int,
    ) -> None:
        """
        title: Lower a generator body slice for one resume state.
        parameters:
          body:
            type: astx.Block
          state_index:
            type: int
        """
        yielded_seen = 0
        for node in body.nodes:
            if _is_yield_node(node):
                if yielded_seen < state_index:
                    yielded_seen += 1
                    continue
                self.visit_child(node)
                return
            if yielded_seen < state_index:
                continue
            stack_size_before = len(self.result_stack)
            self.visit_child(node)
            del self.result_stack[stack_size_before:]
            if self._llvm.ir_builder.block.is_terminated:
                return
        if not self._llvm.ir_builder.block.is_terminated:
            self._emit_generator_stop(body)

    def _emit_generator_resume_body(
        self,
        generator: ResolvedGeneratorFunction,
    ) -> None:
        """
        title: Emit the internal resume state machine for one generator.
        parameters:
          generator:
            type: ResolvedGeneratorFunction
        """
        function = generator.function
        if function.definition is None:
            raise_lowering_internal_error(
                "generator resume requires a function definition",
                node=function.prototype,
            )
        resume = self._declare_generator_resume(generator)
        if len(resume.blocks) > 0:
            return

        frame_type, slots = self._generator_frame_layout(generator)
        entry = resume.append_basic_block("entry")
        self._llvm.ir_builder = ir.IRBuilder(entry)
        raw_frame = resume.args[0]
        frame_ptr = self._llvm.ir_builder.bitcast(
            raw_frame,
            frame_type.as_pointer(),
            name="generator_frame",
        )
        state_addr = self._generator_field_address(
            frame_ptr,
            GENERATOR_STATE_FIELD_INDEX,
            name="generator_state_addr",
        )
        state_value = self._llvm.ir_builder.load(
            state_addr,
            name="generator_state",
        )
        done_bb = resume.append_basic_block("done")
        switch = self._llvm.ir_builder.switch(state_value, done_bb)
        state_blocks = [
            resume.append_basic_block(f"state.{index}")
            for index in range(len(generator.yield_nodes) + 1)
        ]
        for index, block in enumerate(state_blocks):
            switch.add_case(ir.Constant(self._llvm.INT32_TYPE, index), block)

        saved_named_values = self.named_values.copy()
        saved_const_vars = set(self.const_vars)
        saved_frame_ptr = self._current_generator_frame_ptr
        saved_slots = self._current_generator_frame_slots
        saved_out_ptr = self._current_generator_out_ptr
        saved_next_state = self._current_generator_next_state
        self._current_generator_frame_ptr = frame_ptr
        self._current_generator_frame_slots = slots
        self._current_generator_out_ptr = resume.args[1]
        try:
            for index, block in enumerate(state_blocks):
                self._llvm.ir_builder.position_at_start(block)
                self._bind_generator_frame_symbols(frame_ptr, slots)
                self._current_generator_next_state = index + 1
                self._lower_generator_body_from_state(
                    function.definition.body,
                    state_index=index,
                )
            self._llvm.ir_builder.position_at_start(done_bb)
            self._emit_generator_stop(function.definition)
        finally:
            self.named_values = saved_named_values
            self.const_vars = saved_const_vars
            self._current_generator_frame_ptr = saved_frame_ptr
            self._current_generator_frame_slots = saved_slots
            self._current_generator_out_ptr = saved_out_ptr
            self._current_generator_next_state = saved_next_state

    def _allocate_generator_frame(
        self,
        frame_type: ir.IdentifiedStructType,
    ) -> ir.Value:
        """
        title: Allocate one generator frame on the heap.
        parameters:
          frame_type:
            type: ir.IdentifiedStructType
        returns:
          type: ir.Value
        """
        malloc = self._malloc_function()
        size_type = self._llvm.SIZE_T_TYPE or self._llvm.INT64_TYPE
        raw = self._llvm.ir_builder.call(
            malloc,
            [
                ir.Constant(
                    size_type,
                    GENERATOR_FRAME_ALLOCATION_BYTES,
                )
            ],
            name="generator_alloc",
        )
        return self._llvm.ir_builder.bitcast(
            raw,
            frame_type.as_pointer(),
            name="generator_frame",
        )

    def _lower_generator_factory(
        self,
        generator: ResolvedGeneratorFunction,
    ) -> ir.Function:
        """
        title: Emit the public generator factory function.
        parameters:
          generator:
            type: ResolvedGeneratorFunction
        returns:
          type: ir.Function
        """
        function = generator.function
        factory = cast(Any, self)._declare_semantic_function(function)
        if function.symbol_id in self._emitted_function_bodies:
            return factory

        frame_type, slots = self._generator_frame_layout(generator)
        resume = self._declare_generator_resume(generator)
        entry = factory.append_basic_block("entry")
        self._llvm.ir_builder = ir.IRBuilder(entry)
        frame_ptr = self._allocate_generator_frame(frame_type)

        state_addr = self._generator_field_address(
            frame_ptr,
            GENERATOR_STATE_FIELD_INDEX,
            name="generator_state_addr",
        )
        exhausted_addr = self._generator_field_address(
            frame_ptr,
            GENERATOR_EXHAUSTED_FIELD_INDEX,
            name="generator_exhausted_addr",
        )
        self._llvm.ir_builder.store(
            ir.Constant(self._llvm.INT32_TYPE, 0),
            state_addr,
        )
        self._llvm.ir_builder.store(
            ir.Constant(self._llvm.BOOLEAN_TYPE, 0),
            exhausted_addr,
        )

        for llvm_arg, symbol in zip(factory.args, function.args):
            field_index = slots.get(symbol.symbol_id)
            if field_index is None:
                continue
            field_addr = self._generator_field_address(
                frame_ptr,
                field_index,
                name=f"{symbol.name}_capture_addr",
            )
            self._llvm.ir_builder.store(llvm_arg, field_addr)

        raw_frame = self._llvm.ir_builder.bitcast(
            frame_ptr,
            self._llvm.OPAQUE_POINTER_TYPE,
            name="generator_frame_raw",
        )
        raw_resume = self._llvm.ir_builder.bitcast(
            resume,
            self._llvm.OPAQUE_POINTER_TYPE,
            name="generator_resume_raw",
        )
        generator_value = ir.Constant(self._generator_value_type(), None)
        generator_value = self._llvm.ir_builder.insert_value(
            generator_value,
            raw_frame,
            0,
            name="generator_with_frame",
        )
        generator_value = self._llvm.ir_builder.insert_value(
            generator_value,
            raw_resume,
            1,
            name="generator_with_resume",
        )
        self._llvm.ir_builder.ret(generator_value)
        self._emitted_function_bodies.add(function.symbol_id)
        return factory

    def _lower_generator_function(
        self,
        node: astx.FunctionDef,
        generator: ResolvedGeneratorFunction,
    ) -> ir.Function:
        """
        title: Lower one generator function as factory plus resume function.
        parameters:
          node:
            type: astx.FunctionDef
          generator:
            type: ResolvedGeneratorFunction
        returns:
          type: ir.Function
        """
        _ = node
        factory = self._lower_generator_factory(generator)
        self._emit_generator_resume_body(generator)
        self.result_stack.append(factory)
        return factory

    def _lower_generator_for_in_loop(
        self,
        node: astx.ForInLoopStmt,
        iteration: ResolvedIteration,
    ) -> None:
        """
        title: Lower one for-in loop over a generator value.
        parameters:
          node:
            type: astx.ForInLoopStmt
          iteration:
            type: ResolvedIteration
        """
        target_name = getattr(node.target, "name", "item")
        target_type = (
            iteration.target_symbol.type_
            if iteration.target_symbol is not None
            else iteration.element_type
        )
        llvm_target_type = self._llvm_type_for_ast_type(target_type)
        if llvm_target_type is None or isinstance(
            llvm_target_type,
            ir.VoidType,
        ):
            raise_lowering_error(
                "generator loop target type is not lowerable",
                node=node.target,
                code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
            )

        self.visit_child(node.iterable)
        generator_value = require_lowered_value(
            safe_pop(self.result_stack),
            node=node.iterable,
            context="generator iterable",
        )
        frame_value = self._llvm.ir_builder.extract_value(
            generator_value,
            0,
            name="generator_frame",
        )
        resume_raw = self._llvm.ir_builder.extract_value(
            generator_value,
            1,
            name="generator_resume_raw",
        )
        resume_type = ir.FunctionType(
            self._llvm.BOOLEAN_TYPE,
            [self._llvm.OPAQUE_POINTER_TYPE, llvm_target_type.as_pointer()],
        )
        resume = self._llvm.ir_builder.bitcast(
            resume_raw,
            resume_type.as_pointer(),
            name="generator_resume",
        )

        target_addr = self.create_entry_block_alloca(
            target_name,
            llvm_target_type,
        )
        yielded_addr = self.create_entry_block_alloca(
            f"{target_name}_yielded",
            llvm_target_type,
        )

        cond_bb, body_bb, advance_bb, exit_bb = cast(
            Any,
            self,
        )._append_basic_blocks(
            "for.generator",
            "cond",
            "body",
            "advance",
            "exit",
        )
        self._llvm.ir_builder.branch(cond_bb)

        self._llvm.ir_builder.position_at_start(cond_bb)
        has_value = self._llvm.ir_builder.call(
            resume,
            [frame_value, yielded_addr],
            name="generator_has_value",
        )
        self._llvm.ir_builder.cbranch(has_value, body_bb, exit_bb)

        target_key = semantic_symbol_key(node.target, target_name)
        is_constant = not isinstance(
            node.target,
            astx.InlineVariableDeclaration,
        )
        if isinstance(node.target, astx.InlineVariableDeclaration):
            is_constant = (
                node.target.mutability == astx.MutabilityKind.constant
            )

        with cast(Any, self)._loop_scope(
            break_target=exit_bb,
            continue_target=advance_bb,
        ):
            self._llvm.ir_builder.position_at_start(body_bb)
            yielded_value = self._llvm.ir_builder.load(
                yielded_addr,
                name=f"{target_name}_yielded_value",
            )
            yielded_value = self._cast_ast_value(
                yielded_value,
                source_type=iteration.element_type,
                target_type=target_type,
            )
            self._llvm.ir_builder.store(yielded_value, target_addr)
            with cast(Any, self)._temporary_named_value(
                target_key,
                target_addr,
                is_constant=is_constant,
            ):
                cast(Any, self)._discard_child_results(node.body)
            if not self._llvm.ir_builder.block.is_terminated:
                self._llvm.ir_builder.branch(advance_bb)

        self._llvm.ir_builder.position_at_start(advance_bb)
        self._llvm.ir_builder.branch(cond_bb)
        self._llvm.ir_builder.position_at_start(exit_bb)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.YieldStmt) -> None:
        """
        title: Visit YieldStmt nodes.
        parameters:
          node:
            type: astx.YieldStmt
        """
        self._lower_generator_suspend(node, node.value)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.YieldExpr) -> None:
        """
        title: Visit YieldExpr nodes.
        parameters:
          node:
            type: astx.YieldExpr
        """
        self._lower_generator_suspend(node, node.value)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.YieldFromExpr) -> None:
        """
        title: Visit YieldFromExpr nodes.
        parameters:
          node:
            type: astx.YieldFromExpr
        """
        raise_lowering_error(
            "yield from lowering is not supported yet",
            node=node,
            code=DiagnosticCodes.LOWERING_INVALID_CONTROL_FLOW,
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.GeneratorExpr) -> None:
        """
        title: Visit GeneratorExpr nodes.
        parameters:
          node:
            type: astx.GeneratorExpr
        """
        raise_lowering_error(
            "generator expression lowering is not supported yet",
            node=node,
            code=DiagnosticCodes.LOWERING_INVALID_CONTROL_FLOW,
        )


__all__ = ["GeneratorVisitorMixin"]

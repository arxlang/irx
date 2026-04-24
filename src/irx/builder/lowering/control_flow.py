# mypy: disable-error-code=no-redef

"""
title: Control-flow visitor mixins for llvmliteir.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Literal, cast

from llvmlite import ir

from irx import astx
from irx.analysis.resolved_nodes import IterationKind, ResolvedIteration
from irx.analysis.types import (
    is_boolean_type,
    is_float_type,
    is_unsigned_type,
)
from irx.builder.core import (
    VisitorCore,
    semantic_assignment_key,
    semantic_symbol_key,
)
from irx.builder.diagnostics import (
    lowered_value_type_name,
    raise_lowering_error,
    raise_lowering_internal_error,
    require_lowered_value,
    resolved_ast_type_name,
)
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builder.runtime.assertions import (
    ASSERT_FAILURE_SYMBOL_NAME,
    ASSERT_RUNTIME_FEATURE_NAME,
)
from irx.builder.state import LoopTargets
from irx.builder.types import is_fp_type, is_int_type
from irx.builder.vector import emit_add
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ControlFlowVisitorMixin(VisitorMixinBase):
    def _lower_boolean_condition(
        self,
        value: ir.Value | None,
        *,
        node: astx.AST,
        context: str,
    ) -> ir.Value:
        """
        title: Require one lowered control-flow condition to be Boolean.
        parameters:
          value:
            type: ir.Value | None
          node:
            type: astx.AST
          context:
            type: str
        returns:
          type: ir.Value
        """
        if value is None:
            raise_lowering_internal_error(
                f"{context} condition did not lower to a value",
                node=node,
                notes=(f"semantic type: {resolved_ast_type_name(node)}",),
            )
        if not is_int_type(value.type) or value.type.width != 1:
            raise_lowering_internal_error(
                f"{context} condition must lower to LLVM i1, got "
                f"{lowered_value_type_name(value)}",
                node=node,
                code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
                notes=(f"semantic type: {resolved_ast_type_name(node)}",),
            )
        return value

    def _append_basic_blocks(
        self,
        prefix: str,
        *roles: str,
    ) -> tuple[ir.Block, ...]:
        """
        title: Append one canonical sequence of basic blocks.
        parameters:
          prefix:
            type: str
          roles:
            type: str
            variadic: positional
        returns:
          type: tuple[ir.Block, Ellipsis]
        """
        function = self._llvm.ir_builder.function
        return tuple(
            function.append_basic_block(f"{prefix}.{role}") for role in roles
        )

    def _assert_source_name(self, node: astx.AST) -> str:
        """
        title: Return one stable source label for assertion reporting.
        parameters:
          node:
            type: astx.AST
        returns:
          type: str
        """
        current_display_name = cast(
            str | None,
            getattr(self, "_current_module_display_name", None),
        )
        if current_display_name:
            return current_display_name

        current: astx.AST | None = node
        while current is not None:
            if isinstance(current, astx.Module):
                display_names = cast(
                    dict[int, str],
                    getattr(self, "_module_display_names", {}),
                )
                display_name = display_names.get(id(current))
                if display_name:
                    return display_name
                module_name = cast(str, getattr(current, "name", ""))
                return module_name or "<module>"
            parent = getattr(current, "parent", None)
            current = parent if isinstance(parent, astx.AST) else None
        return "<module>"

    def _constant_c_string_pointer(
        self,
        text: str,
        *,
        name_hint: str,
    ) -> ir.Value:
        """
        title: Return one pointer to one internal constant UTF-8 string.
        parameters:
          text:
            type: str
          name_hint:
            type: str
        returns:
          type: ir.Value
        """
        interned_globals = cast(
            dict[tuple[str, str], ir.GlobalVariable],
            getattr(self, "_interned_c_strings", {}),
        )
        cache_key = (name_hint, text)
        global_value = interned_globals.get(cache_key)

        if global_value is None:
            encoded = text.encode("utf8") + b"\0"
            array_type = ir.ArrayType(self._llvm.INT8_TYPE, len(encoded))
            normalized_hint = (
                "".join(
                    character if character.isalnum() else "_"
                    for character in name_hint
                ).strip("_")
                or "cstring"
            )
            counter = cast(
                int,
                getattr(self, "_c_string_global_counter", 0),
            )
            global_name = f"{normalized_hint}_{counter}"
            while global_name in self._llvm.module.globals:
                counter += 1
                global_name = f"{normalized_hint}_{counter}"
            setattr(self, "_c_string_global_counter", counter + 1)

            global_value = ir.GlobalVariable(
                self._llvm.module,
                array_type,
                name=global_name,
            )
            global_value.linkage = "internal"
            global_value.global_constant = True
            global_value.initializer = ir.Constant(
                array_type,
                bytearray(encoded),
            )
            interned_globals[cache_key] = global_value
            setattr(self, "_interned_c_strings", interned_globals)

        return self._llvm.ir_builder.gep(
            global_value,
            [
                ir.Constant(self._llvm.INT32_TYPE, 0),
                ir.Constant(self._llvm.INT32_TYPE, 0),
            ],
            inbounds=True,
            name=f"{name_hint}_ptr",
        )

    def _lower_assert_message_pointer(self, node: astx.AssertStmt) -> ir.Value:
        """
        title: Lower one assertion message to one runtime C string pointer.
        parameters:
          node:
            type: astx.AssertStmt
        returns:
          type: ir.Value
        """
        if node.message is None:
            return self._constant_c_string_pointer(
                "assertion failed",
                name_hint="assert_message",
            )

        self.visit_child(node.message)
        message_value = require_lowered_value(
            safe_pop(self.result_stack),
            node=node.message,
            context="assert message",
        )
        message_source_type = self._resolved_ast_type(node.message)
        message_type = message_value.type

        if (
            isinstance(message_type, ir.PointerType)
            and message_type.pointee == self._llvm.INT8_TYPE
        ):
            return message_value

        if is_int_type(message_type):
            int_arg, int_fmt = self._normalize_int_for_printf(
                message_value,
                unsigned=is_unsigned_type(message_source_type)
                or is_boolean_type(message_source_type),
            )
            int_fmt_gv = self._get_or_create_format_global(int_fmt)
            return self._snprintf_heap(int_fmt_gv, [int_arg])

        if isinstance(
            message_type, (ir.HalfType, ir.FloatType, ir.DoubleType)
        ):
            if isinstance(message_type, (ir.HalfType, ir.FloatType)):
                float_arg = self._llvm.ir_builder.fpext(
                    message_value,
                    self._llvm.DOUBLE_TYPE,
                    "assert_msg_to_double",
                )
            else:
                float_arg = message_value
            float_fmt_gv = self._get_or_create_format_global("%.6f")
            return self._snprintf_heap(float_fmt_gv, [float_arg])

        raise_lowering_error(
            "unsupported AssertStmt message lowering for type "
            f"{lowered_value_type_name(message_value)}",
            code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
            node=node.message,
        )

    def _discard_child_results(self, node: astx.AST) -> None:
        """
        title: Visit one statement child without leaking stack values.
        parameters:
          node:
            type: astx.AST
        """
        stack_size_before = len(self.result_stack)
        self.visit_child(node)
        del self.result_stack[stack_size_before:]

    @contextmanager
    def _loop_scope(
        self,
        *,
        break_target: ir.Block,
        continue_target: ir.Block,
    ) -> Iterator[None]:
        """
        title: Push and pop one active loop target pair.
        parameters:
          break_target:
            type: ir.Block
          continue_target:
            type: ir.Block
        returns:
          type: Iterator[None]
        """
        self.loop_stack.append(
            LoopTargets(
                break_target=break_target,
                continue_target=continue_target,
            )
        )
        try:
            yield
        finally:
            self.loop_stack.pop()

    @contextmanager
    def _temporary_named_value(
        self,
        key: str,
        value: ir.Value,
        *,
        is_constant: bool = False,
    ) -> Iterator[None]:
        """
        title: Bind one temporary symbol slot during lowering.
        parameters:
          key:
            type: str
          value:
            type: ir.Value
          is_constant:
            type: bool
        returns:
          type: Iterator[None]
        """
        had_previous = key in self.named_values
        previous_value = self.named_values.get(key)
        had_const = key in self.const_vars

        self.named_values[key] = value
        if is_constant:
            self.const_vars.add(key)

        try:
            yield
        finally:
            if had_previous:
                self.named_values[key] = previous_value
            else:
                self.named_values.pop(key, None)

            if had_const:
                self.const_vars.add(key)
            else:
                self.const_vars.discard(key)

    def _require_llvm_type(
        self,
        type_: astx.DataType | None,
        *,
        node: astx.AST,
        context: str,
    ) -> ir.Type:
        """
        title: Require one concrete LLVM type for lowering.
        parameters:
          type_:
            type: astx.DataType | None
          node:
            type: astx.AST
          context:
            type: str
        returns:
          type: ir.Type
        """
        llvm_type = self._llvm_type_for_ast_type(type_)
        if llvm_type is None:
            raise_lowering_error(
                f"cannot lower {context} with type "
                f"{resolved_ast_type_name(node)}",
                code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
                node=node,
            )
        return llvm_type

    def _lower_typed_value(
        self,
        node: astx.AST,
        *,
        context: str,
        target_type: astx.DataType | None,
    ) -> ir.Value:
        """
        title: Lower one value and cast it to one target semantic type.
        parameters:
          node:
            type: astx.AST
          context:
            type: str
          target_type:
            type: astx.DataType | None
        returns:
          type: ir.Value
        """
        self.visit_child(node)
        lowered = require_lowered_value(
            safe_pop(self.result_stack),
            node=node,
            context=context,
        )
        return self._cast_ast_value(
            lowered,
            source_type=self._resolved_ast_type(node),
            target_type=target_type,
        )

    def _resolved_iteration(
        self,
        node: astx.AST,
    ) -> ResolvedIteration | None:
        """
        title: Return one node's resolved iteration sidecar.
        parameters:
          node:
            type: astx.AST
        returns:
          type: ResolvedIteration | None
        """
        semantic = getattr(node, "semantic", None)
        iteration = getattr(semantic, "resolved_iteration", None)
        return iteration if isinstance(iteration, ResolvedIteration) else None

    def _lower_list_for_in_loop(
        self,
        node: astx.ForInLoopStmt,
        iteration: ResolvedIteration,
    ) -> None:
        """
        title: Lower one for-in loop over a list iterable.
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
        llvm_target_type = self._require_llvm_type(
            target_type,
            node=node.target,
            context="for-in loop target",
        )
        target_addr = self.create_entry_block_alloca(
            target_name,
            llvm_target_type,
        )
        index_addr = self.create_entry_block_alloca(
            f"{target_name}_iter_index",
            self._llvm.INT64_TYPE,
        )
        self._llvm.ir_builder.store(
            ir.Constant(self._llvm.INT64_TYPE, 0),
            index_addr,
        )

        list_ptr, length = cast(
            Any,
            self,
        )._list_pointer_and_length_for_iteration(node.iterable)

        cond_bb, body_bb, advance_bb, exit_bb = self._append_basic_blocks(
            "for.in",
            "cond",
            "body",
            "advance",
            "exit",
        )
        self._llvm.ir_builder.branch(cond_bb)

        self._llvm.ir_builder.position_at_start(cond_bb)
        current_index = self._llvm.ir_builder.load(
            index_addr,
            name="for_in_index",
        )
        loop_cond = self._llvm.ir_builder.icmp_signed(
            "<",
            current_index,
            length,
            name="for_in_has_item",
        )
        self._llvm.ir_builder.cbranch(loop_cond, body_bb, exit_bb)

        target_key = semantic_symbol_key(node.target, target_name)
        is_constant = False
        if isinstance(node.target, astx.InlineVariableDeclaration):
            is_constant = (
                node.target.mutability == astx.MutabilityKind.constant
            )
        else:
            is_constant = True

        with self._loop_scope(
            break_target=exit_bb,
            continue_target=advance_bb,
        ):
            self._llvm.ir_builder.position_at_start(body_bb)
            item_value = cast(Any, self)._load_list_element_at_index(
                base=node.iterable,
                list_ptr=list_ptr,
                index=current_index,
            )
            item_value = self._cast_ast_value(
                item_value,
                source_type=iteration.element_type,
                target_type=target_type,
            )
            self._llvm.ir_builder.store(item_value, target_addr)
            with self._temporary_named_value(
                target_key,
                target_addr,
                is_constant=is_constant,
            ):
                self._discard_child_results(node.body)
            if not self._llvm.ir_builder.block.is_terminated:
                self._llvm.ir_builder.branch(advance_bb)

        self._llvm.ir_builder.position_at_start(advance_bb)
        next_index = self._llvm.ir_builder.add(
            self._llvm.ir_builder.load(index_addr, name="for_in_step_index"),
            ir.Constant(self._llvm.INT64_TYPE, 1),
            name="for_in_next_index",
        )
        self._llvm.ir_builder.store(next_index, index_addr)
        self._llvm.ir_builder.branch(cond_bb)

        self._llvm.ir_builder.position_at_start(exit_bb)

    def _zero_value(self, llvm_type: ir.Type) -> ir.Constant:
        """
        title: Return the zero constant for one numeric LLVM type.
        parameters:
          llvm_type:
            type: ir.Type
        returns:
          type: ir.Constant
        """
        if is_fp_type(llvm_type):
            return ir.Constant(llvm_type, 0.0)
        return ir.Constant(llvm_type, 0)

    def _constant_numeric_direction(
        self,
        value: ir.Value,
    ) -> Literal["positive", "negative", "zero"] | None:
        """
        title: Return one compile-time sign classification when available.
        parameters:
          value:
            type: ir.Value
        returns:
          type: Literal[positive, negative, zero] | None
        """
        if not isinstance(value, ir.Constant):
            return None
        constant = value.constant
        if not isinstance(constant, (int, float)):
            return None
        if constant > 0:
            return "positive"
        if constant < 0:
            return "negative"
        return "zero"

    def _range_step_flags(
        self,
        step_value: ir.Value,
        *,
        unsigned: bool,
    ) -> tuple[ir.Value, ir.Value]:
        """
        title: Compute one runtime step-direction pair for for-range loops.
        parameters:
          step_value:
            type: ir.Value
          unsigned:
            type: bool
        returns:
          type: tuple[ir.Value, ir.Value]
        """
        zero = self._zero_value(step_value.type)
        if is_fp_type(step_value.type):
            return (
                self._llvm.ir_builder.fcmp_ordered(
                    ">",
                    step_value,
                    zero,
                    "for.range.step.positive",
                ),
                self._llvm.ir_builder.fcmp_ordered(
                    "<",
                    step_value,
                    zero,
                    "for.range.step.negative",
                ),
            )
        if unsigned:
            return (
                self._llvm.ir_builder.icmp_unsigned(
                    "!=",
                    step_value,
                    zero,
                    "for.range.step.nonzero",
                ),
                ir.Constant(self._llvm.BOOLEAN_TYPE, 0),
            )
        return (
            self._llvm.ir_builder.icmp_signed(
                ">",
                step_value,
                zero,
                "for.range.step.positive",
            ),
            self._llvm.ir_builder.icmp_signed(
                "<",
                step_value,
                zero,
                "for.range.step.negative",
            ),
        )

    def _for_count_update_stores_loop_variable(
        self,
        update: astx.AST,
        *,
        loop_symbol_key: str,
    ) -> bool:
        """
        title: >-
          Return whether one for-count update already writes the loop slot.
        parameters:
          update:
            type: astx.AST
          loop_symbol_key:
            type: str
        returns:
          type: bool
        """
        if isinstance(update, astx.UnaryOp) and update.op_code in {"++", "--"}:
            operand = update.operand
            if isinstance(operand, astx.Identifier):
                return (
                    semantic_symbol_key(operand, operand.name)
                    == loop_symbol_key
                )
            return False

        if isinstance(update, astx.VariableAssignment):
            return (
                semantic_assignment_key(update, update.name) == loop_symbol_key
            )

        if (
            isinstance(update, astx.BinaryOp)
            and update.op_code == "="
            and isinstance(update.lhs, astx.Identifier)
        ):
            return (
                semantic_assignment_key(update, update.lhs.name)
                == loop_symbol_key
            )

        return False

    def _for_range_condition(
        self,
        current_value: ir.Value,
        end_value: ir.Value,
        *,
        unsigned: bool,
        step_direction: Literal["positive", "negative", "zero"] | None,
        step_positive: ir.Value | None = None,
        step_negative: ir.Value | None = None,
    ) -> ir.Value:
        """
        title: Lower one canonical for-range loop condition.
        parameters:
          current_value:
            type: ir.Value
          end_value:
            type: ir.Value
          unsigned:
            type: bool
          step_direction:
            type: Literal[positive, negative, zero] | None
          step_positive:
            type: ir.Value | None
          step_negative:
            type: ir.Value | None
        returns:
          type: ir.Value
        """
        builder = self._llvm.ir_builder
        before_end = self._emit_numeric_compare(
            "<",
            current_value,
            end_value,
            unsigned=unsigned,
            name="for.range.before_end",
        )

        if step_direction == "positive":
            return before_end
        if step_direction == "zero":
            return ir.Constant(self._llvm.BOOLEAN_TYPE, 0)

        if unsigned:
            if step_positive is None:
                raise_lowering_internal_error(
                    "for-range loop missing nonzero-step flag",
                )
            return builder.and_(step_positive, before_end, "for.range.cond")

        after_end = self._emit_numeric_compare(
            ">",
            current_value,
            end_value,
            unsigned=unsigned,
            name="for.range.after_end",
        )

        if step_direction == "negative":
            return after_end

        if step_positive is None or step_negative is None:
            raise_lowering_internal_error(
                "for-range loop missing runtime step-direction flags",
            )

        positive_path = builder.and_(
            step_positive,
            before_end,
            "for.range.cond.positive",
        )
        negative_path = builder.and_(
            step_negative,
            after_end,
            "for.range.cond.negative",
        )
        return builder.or_(positive_path, negative_path, "for.range.cond")

    @VisitorCore.visit.dispatch
    def visit(self, block: astx.Block) -> None:
        """
        title: Visit Block nodes.
        parameters:
          block:
            type: astx.Block
        """
        for node in block.nodes:
            if isinstance(node, (astx.StructDefStmt, astx.ClassDefStmt)):
                self.visit_child(node)

        result = None
        for node in block.nodes:
            if self._llvm.ir_builder.block.terminator is not None:
                break

            stack_size_before = len(self.result_stack)
            self.visit_child(node)
            if len(self.result_stack) > stack_size_before:
                result = self.result_stack.pop()

            if self._llvm.ir_builder.block.terminator is not None:
                result = None
                break

        if result is not None:
            self.result_stack.append(result)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.AssertStmt) -> None:
        """
        title: Visit AssertStmt nodes.
        parameters:
          node:
            type: astx.AssertStmt
        """
        self.visit_child(node.condition)
        condition_value = self._lower_boolean_condition(
            safe_pop(self.result_stack),
            node=node.condition,
            context="assert",
        )

        pass_bb, fail_bb = self._append_basic_blocks("assert", "pass", "fail")
        self._llvm.ir_builder.cbranch(condition_value, pass_bb, fail_bb)

        self._llvm.ir_builder.position_at_start(fail_bb)
        source_ptr = self._constant_c_string_pointer(
            self._assert_source_name(node),
            name_hint="assert_source",
        )
        line_value = ir.Constant(self._llvm.INT32_TYPE, node.loc.line)
        col_value = ir.Constant(self._llvm.INT32_TYPE, node.loc.col)
        message_ptr = self._lower_assert_message_pointer(node)
        fail_function = self.require_runtime_symbol(
            ASSERT_RUNTIME_FEATURE_NAME,
            ASSERT_FAILURE_SYMBOL_NAME,
        )
        self._llvm.ir_builder.call(
            fail_function,
            [source_ptr, line_value, col_value, message_ptr],
        )
        self._llvm.ir_builder.unreachable()

        self._llvm.ir_builder.position_at_start(pass_bb)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.IfStmt) -> None:
        """
        title: Visit IfStmt nodes.
        parameters:
          node:
            type: astx.IfStmt
        """
        self.visit_child(node.condition)
        cond_v = self._lower_boolean_condition(
            safe_pop(self.result_stack),
            node=node.condition,
            context="if",
        )

        then_bb = self._llvm.ir_builder.function.append_basic_block(
            "bb_if_then"
        )
        else_bb = self._llvm.ir_builder.function.append_basic_block(
            "bb_if_else"
        )
        merge_bb = self._llvm.ir_builder.function.append_basic_block(
            "bb_if_end"
        )
        self._llvm.ir_builder.cbranch(cond_v, then_bb, else_bb)

        self._llvm.ir_builder.position_at_start(then_bb)
        then_stack_size = len(self.result_stack)
        self.visit_child(node.then)
        then_terminated = self._llvm.ir_builder.block.terminator is not None
        then_v = None
        if len(self.result_stack) > then_stack_size:
            then_v = self.result_stack.pop()
        if not then_terminated:
            self._llvm.ir_builder.branch(merge_bb)
            then_bb = self._llvm.ir_builder.block

        self._llvm.ir_builder.position_at_start(else_bb)
        else_stack_size = len(self.result_stack)
        if node.else_ is not None:
            self.visit_child(node.else_)
        else_terminated = self._llvm.ir_builder.block.terminator is not None
        else_v = None
        if len(self.result_stack) > else_stack_size:
            else_v = self.result_stack.pop()
        if not else_terminated:
            self._llvm.ir_builder.branch(merge_bb)
            else_bb = self._llvm.ir_builder.block

        then_reaches_merge = not then_terminated
        else_reaches_merge = not else_terminated
        if not then_reaches_merge and not else_reaches_merge:
            self._llvm.ir_builder.position_at_start(merge_bb)
            self._llvm.ir_builder.unreachable()
            return

        self._llvm.ir_builder.position_at_start(merge_bb)
        if (
            then_reaches_merge
            and else_reaches_merge
            and then_v is not None
            and else_v is not None
            and then_v.type == else_v.type
        ):
            phi = self._llvm.ir_builder.phi(then_v.type, "iftmp")
            phi.add_incoming(then_v, then_bb)
            phi.add_incoming(else_v, else_bb)
            self.result_stack.append(phi)
            return

        if (
            then_reaches_merge
            and not else_reaches_merge
            and then_v is not None
        ):
            self.result_stack.append(then_v)
            return

        if (
            else_reaches_merge
            and not then_reaches_merge
            and else_v is not None
        ):
            self.result_stack.append(else_v)

    @VisitorCore.visit.dispatch
    def visit(self, expr: astx.WhileStmt) -> None:
        """
        title: Visit WhileStmt nodes.
        parameters:
          expr:
            type: astx.WhileStmt
        """
        cond_bb, body_bb, exit_bb = self._append_basic_blocks(
            "while",
            "cond",
            "body",
            "exit",
        )
        self._llvm.ir_builder.branch(cond_bb)

        self._llvm.ir_builder.position_at_start(cond_bb)
        self.visit_child(expr.condition)
        cond_val = self._lower_boolean_condition(
            safe_pop(self.result_stack),
            node=expr.condition,
            context="while",
        )
        self._llvm.ir_builder.cbranch(cond_val, body_bb, exit_bb)

        with self._loop_scope(
            break_target=exit_bb,
            continue_target=cond_bb,
        ):
            self._llvm.ir_builder.position_at_start(body_bb)
            self._discard_child_results(expr.body)
            if not self._llvm.ir_builder.block.is_terminated:
                self._llvm.ir_builder.branch(cond_bb)

        self._llvm.ir_builder.position_at_start(exit_bb)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.ForCountLoopStmt) -> None:
        """
        title: Visit ForCountLoopStmt nodes.
        parameters:
          node:
            type: astx.ForCountLoopStmt
        """
        initializer_key = semantic_symbol_key(
            node.initializer,
            node.initializer.name,
        )
        had_previous = initializer_key in self.named_values
        previous_value = self.named_values.get(initializer_key)
        had_const = initializer_key in self.const_vars

        initializer_stack_size = len(self.result_stack)
        self.visit_child(node.initializer)
        del self.result_stack[initializer_stack_size:]

        var_addr = self.named_values.get(initializer_key)
        if var_addr is None:
            raise_lowering_internal_error(
                "for-count loop initializer did not create storage",
                node=node.initializer,
            )

        cond_bb, body_bb, update_bb, exit_bb = self._append_basic_blocks(
            "for.count",
            "cond",
            "body",
            "update",
            "exit",
        )
        self._llvm.ir_builder.branch(cond_bb)

        try:
            self._llvm.ir_builder.position_at_start(cond_bb)
            self.visit_child(node.condition)
            cond_val = self._lower_boolean_condition(
                safe_pop(self.result_stack),
                node=node.condition,
                context="for-count loop",
            )
            self._llvm.ir_builder.cbranch(cond_val, body_bb, exit_bb)

            with self._loop_scope(
                break_target=exit_bb,
                continue_target=update_bb,
            ):
                self._llvm.ir_builder.position_at_start(body_bb)
                self._discard_child_results(node.body)
                if not self._llvm.ir_builder.block.is_terminated:
                    self._llvm.ir_builder.branch(update_bb)

            self._llvm.ir_builder.position_at_start(update_bb)
            update_val = self._lower_typed_value(
                node.update,
                context="for-count loop update",
                target_type=node.initializer.type_,
            )
            if not self._for_count_update_stores_loop_variable(
                node.update,
                loop_symbol_key=initializer_key,
            ):
                self._llvm.ir_builder.store(update_val, var_addr)
            self._llvm.ir_builder.branch(cond_bb)

            self._llvm.ir_builder.position_at_start(exit_bb)
        finally:
            if had_previous:
                self.named_values[initializer_key] = previous_value
            else:
                self.named_values.pop(initializer_key, None)

            if had_const:
                self.const_vars.add(initializer_key)
            else:
                self.const_vars.discard(initializer_key)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.ForRangeLoopStmt) -> None:
        """
        title: Visit ForRangeLoopStmt nodes.
        parameters:
          node:
            type: astx.ForRangeLoopStmt
        """
        loop_type = node.variable.type_
        llvm_loop_type = self._require_llvm_type(
            loop_type,
            node=node.variable,
            context="for-range loop variable",
        )
        var_addr = self.create_entry_block_alloca(
            node.variable.name,
            llvm_loop_type,
        )

        start_val = self._lower_typed_value(
            node.start,
            context="for-range start expression",
            target_type=loop_type,
        )
        self._llvm.ir_builder.store(start_val, var_addr)

        end_val = self._lower_typed_value(
            node.end,
            context="for-range end expression",
            target_type=loop_type,
        )

        if not isinstance(node.step, astx.LiteralNone):
            step_val = self._lower_typed_value(
                node.step,
                context="for-range step expression",
                target_type=loop_type,
            )
        else:
            step_val = ir.Constant(
                llvm_loop_type,
                1.0 if is_float_type(loop_type) else 1,
            )

        unsigned_loop = bool(
            loop_type is not None and is_unsigned_type(loop_type)
        )
        step_direction = self._constant_numeric_direction(step_val)
        step_positive = None
        step_negative = None
        if step_direction is None:
            step_positive, step_negative = self._range_step_flags(
                step_val,
                unsigned=unsigned_loop,
            )

        cond_bb, body_bb, step_bb, exit_bb = self._append_basic_blocks(
            "for.range",
            "cond",
            "body",
            "step",
            "exit",
        )
        self._llvm.ir_builder.branch(cond_bb)

        variable_key = semantic_symbol_key(node.variable, node.variable.name)
        self._llvm.ir_builder.position_at_start(cond_bb)
        current_value = self._llvm.ir_builder.load(
            var_addr, node.variable.name
        )
        loop_cond = self._for_range_condition(
            current_value,
            end_val,
            unsigned=unsigned_loop,
            step_direction=step_direction,
            step_positive=step_positive,
            step_negative=step_negative,
        )
        self._llvm.ir_builder.cbranch(loop_cond, body_bb, exit_bb)

        with self._loop_scope(
            break_target=exit_bb,
            continue_target=step_bb,
        ):
            self._llvm.ir_builder.position_at_start(body_bb)
            with self._temporary_named_value(
                variable_key,
                var_addr,
                is_constant=(
                    node.variable.mutability == astx.MutabilityKind.constant
                ),
            ):
                self._discard_child_results(node.body)
            if not self._llvm.ir_builder.block.is_terminated:
                self._llvm.ir_builder.branch(step_bb)

        self._llvm.ir_builder.position_at_start(step_bb)
        current_value = self._llvm.ir_builder.load(
            var_addr, node.variable.name
        )
        next_value = emit_add(
            self._llvm.ir_builder,
            current_value,
            step_val,
            "nextvar",
        )
        self._llvm.ir_builder.store(next_value, var_addr)
        self._llvm.ir_builder.branch(cond_bb)

        self._llvm.ir_builder.position_at_start(exit_bb)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.ForInLoopStmt) -> None:
        """
        title: Visit ForInLoopStmt nodes.
        parameters:
          node:
            type: astx.ForInLoopStmt
        """
        iteration = self._resolved_iteration(node)
        if iteration is None:
            raise_lowering_error(
                "for-in loop is missing resolved iteration metadata",
                node=node,
                code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
            )
        if iteration.kind is IterationKind.LIST:
            self._lower_list_for_in_loop(node, iteration)
            return
        raise_lowering_error(
            f"for-in lowering does not yet support {iteration.kind.value} "
            "iterables",
            node=node.iterable,
            code=DiagnosticCodes.LOWERING_TYPE_MISMATCH,
        )

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.BreakStmt) -> None:
        """
        title: Visit BreakStmt nodes.
        parameters:
          node:
            type: astx.BreakStmt
        """
        if not self.loop_stack:
            raise_lowering_error(
                "break statement used outside an active loop",
                node=node,
                code=DiagnosticCodes.LOWERING_INVALID_CONTROL_FLOW,
                hint=(
                    "use `break` only inside while, for-count, or "
                    "for-range loop bodies"
                ),
            )
        self._llvm.ir_builder.branch(self.loop_stack[-1].break_target)

    @VisitorCore.visit.dispatch
    def visit(self, node: astx.ContinueStmt) -> None:
        """
        title: Visit ContinueStmt nodes.
        parameters:
          node:
            type: astx.ContinueStmt
        """
        if not self.loop_stack:
            raise_lowering_error(
                "continue statement used outside an active loop",
                node=node,
                code=DiagnosticCodes.LOWERING_INVALID_CONTROL_FLOW,
                hint=(
                    "use `continue` only inside while, for-count, or "
                    "for-range loop bodies"
                ),
            )
        self._llvm.ir_builder.branch(self.loop_stack[-1].continue_target)

# mypy: disable-error-code=no-redef

"""
title: Control-flow visitor mixins for llvmliteir.
"""

from llvmlite import ir

from irx import astx
from irx.builders.base import BuilderVisitor
from irx.builders.llvmliteir.core import _semantic_symbol_key
from irx.builders.llvmliteir.protocols import VisitorMixinBase
from irx.builders.llvmliteir.runtime import safe_pop
from irx.builders.llvmliteir.types import is_fp_type
from irx.builders.llvmliteir.vector import emit_add


class ControlFlowVisitorMixin(VisitorMixinBase):
    @BuilderVisitor.visit.dispatch
    def visit(self, block: astx.Block) -> None:
        """
        title: Visit Block nodes.
        parameters:
          block:
            type: astx.Block
        """
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

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.IfStmt) -> None:
        """
        title: Visit IfStmt nodes.
        parameters:
          node:
            type: astx.IfStmt
        """
        self.visit_child(node.condition)
        cond_v = safe_pop(self.result_stack)
        if cond_v is None:
            raise Exception("codegen: Invalid condition expression.")

        if is_fp_type(cond_v.type):
            cmp_instruction = self._llvm.ir_builder.fcmp_ordered
            zero_val = ir.Constant(cond_v.type, 0.0)
        else:
            cmp_instruction = self._llvm.ir_builder.icmp_signed
            zero_val = ir.Constant(cond_v.type, 0)

        cond_v = cmp_instruction("!=", cond_v, zero_val)

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

    @BuilderVisitor.visit.dispatch
    def visit(self, expr: astx.WhileStmt) -> None:
        """
        title: Visit WhileStmt nodes.
        parameters:
          expr:
            type: astx.WhileStmt
        """
        cond_bb = self._llvm.ir_builder.function.append_basic_block(
            "whilecond"
        )
        body_bb = self._llvm.ir_builder.function.append_basic_block(
            "whilebody"
        )
        after_bb = self._llvm.ir_builder.function.append_basic_block(
            "afterwhile"
        )
        self._llvm.ir_builder.branch(cond_bb)

        self._llvm.ir_builder.position_at_end(cond_bb)
        self.visit_child(expr.condition)
        cond_val = safe_pop(self.result_stack)
        if cond_val is None:
            raise Exception("codegen: Invalid condition expression.")

        if is_fp_type(cond_val.type):
            cmp_instruction = self._llvm.ir_builder.fcmp_ordered
            zero_val = ir.Constant(cond_val.type, 0.0)
        else:
            cmp_instruction = self._llvm.ir_builder.icmp_signed
            zero_val = ir.Constant(cond_val.type, 0)

        cond_val = cmp_instruction("!=", cond_val, zero_val, "whilecond")
        self._llvm.ir_builder.cbranch(cond_val, body_bb, after_bb)
        self.loop_stack.append(
            {
                "break_target": after_bb,
                "continue_target": cond_bb,
            }
        )

        self._llvm.ir_builder.position_at_end(body_bb)
        self.visit_child(expr.body)
        safe_pop(self.result_stack)
        if not self._llvm.ir_builder.block.is_terminated:
            self._llvm.ir_builder.branch(cond_bb)

        self.loop_stack.pop()
        self._llvm.ir_builder.position_at_end(after_bb)
        self.result_stack.append(ir.Constant(self._llvm.INT32_TYPE, 0))

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.ForCountLoopStmt) -> None:
        """
        title: Visit ForCountLoopStmt nodes.
        parameters:
          node:
            type: astx.ForCountLoopStmt
        """
        saved_block = self._llvm.ir_builder.block
        var_addr = self.create_entry_block_alloca(
            "for_count_loop", node.initializer.type_.__class__.__name__.lower()
        )
        self._llvm.ir_builder.position_at_end(saved_block)

        self.visit_child(node.initializer)
        initializer_val = safe_pop(self.result_stack)
        if initializer_val is None:
            raise Exception("codegen: Invalid initializer expression.")
        self._llvm.ir_builder.store(initializer_val, var_addr)

        loop_header_bb = self._llvm.ir_builder.function.append_basic_block(
            "loop.header"
        )
        self._llvm.ir_builder.branch(loop_header_bb)
        self._llvm.ir_builder.position_at_start(loop_header_bb)

        initializer_key = _semantic_symbol_key(
            node.initializer, node.initializer.name
        )
        old_val = self.named_values.get(initializer_key)
        self.named_values[initializer_key] = var_addr

        self.visit_child(node.condition)
        cond_val = safe_pop(self.result_stack)

        loop_body_bb = self._llvm.ir_builder.function.append_basic_block(
            "loop.body"
        )
        loop_update_bb = self._llvm.ir_builder.function.append_basic_block(
            "loop.update"
        )
        after_loop_bb = self._llvm.ir_builder.function.append_basic_block(
            "after.loop"
        )
        self._llvm.ir_builder.cbranch(cond_val, loop_body_bb, after_loop_bb)
        self.loop_stack.append(
            {
                "break_target": after_loop_bb,
                "continue_target": loop_update_bb,
            }
        )

        self._llvm.ir_builder.position_at_start(loop_body_bb)
        self.visit_child(node.body)
        safe_pop(self.result_stack)
        if not self._llvm.ir_builder.block.is_terminated:
            self._llvm.ir_builder.branch(loop_update_bb)

        self.loop_stack.pop()
        self._llvm.ir_builder.position_at_start(loop_update_bb)
        self.visit_child(node.update)
        update_val = safe_pop(self.result_stack)
        if update_val is None:
            raise Exception("codegen: Invalid update expression.")
        self._llvm.ir_builder.store(update_val, var_addr)
        self._llvm.ir_builder.branch(loop_header_bb)

        self._llvm.ir_builder.position_at_start(after_loop_bb)
        if old_val:
            self.named_values[initializer_key] = old_val
        else:
            self.named_values.pop(initializer_key, None)

        result = ir.Constant(
            self._llvm.get_data_type(
                node.initializer.type_.__class__.__name__.lower()
            ),
            0,
        )
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.ForRangeLoopStmt) -> None:
        """
        title: Visit ForRangeLoopStmt nodes.
        parameters:
          node:
            type: astx.ForRangeLoopStmt
        """
        saved_block = self._llvm.ir_builder.block
        var_addr = self.create_entry_block_alloca(
            "for_range_loop",
            node.variable.type_.__class__.__name__.lower(),
        )
        self._llvm.ir_builder.position_at_end(saved_block)

        self.visit_child(node.start)
        start_val = safe_pop(self.result_stack)
        if start_val is None:
            raise Exception("codegen: Invalid start argument.")
        self._llvm.ir_builder.store(start_val, var_addr)

        func = self._llvm.ir_builder.function
        header_bb = func.append_basic_block("for.header")
        body_bb = func.append_basic_block("for.body")
        step_bb = func.append_basic_block("for.step")
        after_bb = func.append_basic_block("for.after")
        self._llvm.ir_builder.branch(header_bb)

        self._llvm.ir_builder.position_at_start(header_bb)
        cur_var = self._llvm.ir_builder.load(var_addr, node.variable.name)
        self.visit_child(node.end)
        end_val = safe_pop(self.result_stack)
        if end_val is None:
            raise Exception("codegen: Invalid end argument.")

        if node.step:
            self.visit_child(node.step)
            step_val = safe_pop(self.result_stack)
            if step_val is None:
                raise Exception("codegen: Invalid step argument.")
        else:
            step_val = ir.Constant(
                self._llvm.get_data_type(
                    node.variable.type_.__class__.__name__.lower()
                ),
                1,
            )

        if is_fp_type(cur_var.type):
            cmp_instruction = self._llvm.ir_builder.fcmp_ordered
            cmp_op = (
                "<"
                if isinstance(step_val, ir.Constant) and step_val.constant > 0
                else ">"
            )
        else:
            cmp_instruction = self._llvm.ir_builder.icmp_signed
            cmp_op = (
                "<"
                if isinstance(step_val, ir.Constant) and step_val.constant > 0
                else ">"
            )

        loop_cond = cmp_instruction(cmp_op, cur_var, end_val, "loopcond")
        self._llvm.ir_builder.cbranch(loop_cond, body_bb, after_bb)
        self.loop_stack.append(
            {"break_target": after_bb, "continue_target": step_bb}
        )

        self._llvm.ir_builder.position_at_start(body_bb)
        variable_key = _semantic_symbol_key(node.variable, node.variable.name)
        old_val = self.named_values.get(variable_key)
        self.named_values[variable_key] = var_addr

        self.visit_child(node.body)
        safe_pop(self.result_stack)
        if not self._llvm.ir_builder.block.is_terminated:
            self._llvm.ir_builder.branch(step_bb)

        self.loop_stack.pop()
        self._llvm.ir_builder.position_at_start(step_bb)
        cur_var = self._llvm.ir_builder.load(var_addr, node.variable.name)
        next_var = emit_add(
            self._llvm.ir_builder, cur_var, step_val, "nextvar"
        )
        self._llvm.ir_builder.store(next_var, var_addr)
        self._llvm.ir_builder.branch(header_bb)

        self._llvm.ir_builder.position_at_start(after_bb)
        if old_val:
            self.named_values[variable_key] = old_val
        else:
            self.named_values.pop(variable_key, None)

        result = ir.Constant(
            self._llvm.get_data_type(
                node.variable.type_.__class__.__name__.lower()
            ),
            0,
        )
        self.result_stack.append(result)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.BreakStmt) -> None:
        """
        title: Visit BreakStmt nodes.
        parameters:
          node:
            type: astx.BreakStmt
        """
        if not self.loop_stack:
            raise Exception("codegen: Break statement outside loop.")
        break_target = self.loop_stack[-1]["break_target"]
        self._llvm.ir_builder.branch(break_target)

    @BuilderVisitor.visit.dispatch
    def visit(self, node: astx.ContinueStmt) -> None:
        """
        title: Visit ContinueStmt nodes.
        parameters:
          node:
            type: astx.ContinueStmt
        """
        if not self.loop_stack:
            raise Exception("codegen: Continue statement outside loop.")
        continue_target = self.loop_stack[-1]["continue_target"]
        self._llvm.ir_builder.branch(continue_target)

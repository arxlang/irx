# mypy: disable-error-code=no-redef

"""
title: Unary-operator visitor mixins for llvmliteir.
"""

from llvmlite import ir

from irx import astx
from irx.builder.core import VisitorCore, semantic_symbol_key
from irx.builder.protocols import VisitorMixinBase
from irx.builder.runtime import safe_pop
from irx.builder.types import is_fp_type, is_int_type
from irx.typecheck import typechecked


@typechecked
class UnaryOpVisitorMixin(VisitorMixinBase):
    @VisitorCore.visit.dispatch  # type: ignore[attr-defined,untyped-decorator]
    def visit(self, node: astx.UnaryOp) -> None:
        """
        title: Visit UnaryOp nodes.
        parameters:
          node:
            type: astx.UnaryOp
        """
        if node.op_code == "++":
            self.visit_child(node.operand)
            operand_val = safe_pop(self.result_stack)
            if operand_val is None:
                raise Exception("codegen: Invalid unary operand.")
            operand_key = (
                semantic_symbol_key(node.operand, node.operand.name)
                if isinstance(node.operand, astx.Identifier)
                else ""
            )

            one = ir.Constant(operand_val.type, 1)
            if is_fp_type(operand_val.type):
                result = self._llvm.ir_builder.fadd(operand_val, one, "inctmp")
            else:
                result = self._llvm.ir_builder.add(operand_val, one, "inctmp")

            if isinstance(node.operand, astx.Identifier):
                if operand_key in self.const_vars:
                    raise Exception(
                        f"Cannot mutate '{node.operand.name}':"
                        "declared as constant"
                    )
                var_addr = self.named_values.get(operand_key)
                if var_addr:
                    self._llvm.ir_builder.store(result, var_addr)

            self.result_stack.append(result)
            return

        if node.op_code == "--":
            self.visit_child(node.operand)
            operand_val = safe_pop(self.result_stack)
            if operand_val is None:
                raise Exception("codegen: Invalid unary operand.")
            operand_key = (
                semantic_symbol_key(node.operand, node.operand.name)
                if isinstance(node.operand, astx.Identifier)
                else ""
            )
            one = ir.Constant(operand_val.type, 1)
            if is_fp_type(operand_val.type):
                result = self._llvm.ir_builder.fsub(operand_val, one, "dectmp")
            else:
                result = self._llvm.ir_builder.sub(operand_val, one, "dectmp")

            if isinstance(node.operand, astx.Identifier):
                if operand_key in self.const_vars:
                    raise Exception(
                        f"Cannot mutate '{node.operand.name}':"
                        "declared as constant"
                    )
                var_addr = self.named_values.get(operand_key)
                if var_addr:
                    self._llvm.ir_builder.store(result, var_addr)

            self.result_stack.append(result)
            return

        if node.op_code == "!":
            self.visit_child(node.operand)
            val = safe_pop(self.result_stack)
            if val is None:
                raise Exception("codegen: Invalid unary operand.")
            if not is_int_type(val.type) or val.type.width != 1:
                raise Exception(
                    "codegen: unary operator '!' must lower a Boolean operand."
                )

            result = self._llvm.ir_builder.xor(
                val,
                ir.Constant(self._llvm.BOOLEAN_TYPE, 1),
                "nottmp",
            )

            if isinstance(node.operand, astx.Identifier):
                operand_key = semantic_symbol_key(
                    node.operand, node.operand.name
                )
                if operand_key in self.const_vars:
                    raise Exception(
                        f"Cannot mutate '{node.operand.name}':"
                        "declared as constant"
                    )
                addr = self.named_values.get(operand_key)
                if addr:
                    self._llvm.ir_builder.store(result, addr)

            self.result_stack.append(result)
            return

        raise Exception(f"Unary operator {node.op_code} not implemented yet.")

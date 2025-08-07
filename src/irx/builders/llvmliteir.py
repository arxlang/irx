"""LLVM-IR builder."""

from __future__ import annotations

import os
import tempfile

from typing import Any, Callable, Optional, cast

import astx
import xh

from llvmlite import binding as llvm
from llvmlite import ir
from plum import dispatch
from public import public

from irx import system
from irx.builders.base import Builder, BuilderVisitor
from irx.tools.typing import typechecked


@typechecked
def safe_pop(lst: list[ir.Value | ir.Function]) -> ir.Value | ir.Function:
    """Implement a safe pop operation for lists."""
    try:
        return lst.pop()
    except IndexError:
        return None


@typechecked
class VariablesLLVM:
    """Store all the LLVM variables that is used for the code generation."""

    FLOAT_TYPE: ir.types.Type
    FLOAT16_TYPE: ir.types.Type
    DOUBLE_TYPE: ir.types.Type
    INT8_TYPE: ir.types.Type
    INT64_TYPE: ir.types.Type
    INT16_TYPE: ir.types.Type
    INT32_TYPE: ir.types.Type
    VOID_TYPE: ir.types.Type
    BOOLEAN_TYPE: ir.types.Type

    context: ir.context.Context
    module: ir.module.Module

    ir_builder: ir.builder.IRBuilder

    def get_data_type(self, type_name: str) -> ir.types.Type:
        """
        Get the LLVM data type for the given type name.

        Parameters
        ----------
            type_name (str): The name of the type.

        Returns
        -------
            ir.Type: The LLVM data type.
        """
        if type_name == "float32":
            return self.FLOAT_TYPE
        elif type_name == "float16":
            return self.FLOAT16_TYPE
        elif type_name == "double":
            return self.DOUBLE_TYPE
        elif type_name == "boolean":
            return self.BOOLEAN_TYPE
        elif type_name == "int8":
            return self.INT8_TYPE
        elif type_name == "int16":
            return self.INT16_TYPE
        elif type_name == "int32":
            return self.INT32_TYPE
        elif type_name == "int64":
            return self.INT64_TYPE
        elif type_name == "char":
            return self.INT8_TYPE
        elif type_name == "nonetype":
            return self.VOID_TYPE

        raise Exception(f"[EE]: Type name {type_name} not valid.")


@typechecked
class LLVMLiteIRVisitor(BuilderVisitor):
    """LLVM-IR Translator."""

    # AllocaInst
    named_values: dict[str, Any] = {}

    _llvm: VariablesLLVM

    function_protos: dict[str, astx.FunctionPrototype]
    result_stack: list[ir.Value | ir.Function] = []

    def __init__(self) -> None:
        """Initialize LLVMTranslator object."""
        super().__init__()

        # named_values as instance variable so it isn't shared across instances
        self.named_values: dict[str, Any] = {}
        self.function_protos: dict[str, astx.FunctionPrototype] = {}
        self.result_stack: list[ir.Value | ir.Function] = []

        self.initialize()

        self.target = llvm.Target.from_default_triple()
        self.target_machine = self.target.create_target_machine(
            codemodel="small"
        )

        self._add_builtins()

    def translate(self, node: astx.AST) -> str:
        """Translate an ASTx expression to string."""
        self.visit(node)
        return str(self._llvm.module)

    def initialize(self) -> None:
        """Initialize self."""
        # self._llvm.context = ir.context.Context()
        self._llvm = VariablesLLVM()
        self._llvm.module = ir.module.Module("Arx")

        # initialize the target registry etc.
        llvm.initialize()
        llvm.initialize_all_asmprinters()
        llvm.initialize_all_targets()
        llvm.initialize_native_target()
        llvm.initialize_native_asmparser()
        llvm.initialize_native_asmprinter()

        # Create a new builder for the module.
        self._llvm.ir_builder = ir.IRBuilder()

        # Data Types
        self._llvm.FLOAT_TYPE = ir.FloatType()
        self._llvm.FLOAT16_TYPE = ir.HalfType()
        self._llvm.DOUBLE_TYPE = ir.DoubleType()
        self._llvm.BOOLEAN_TYPE = ir.IntType(1)
        self._llvm.INT8_TYPE = ir.IntType(8)
        self._llvm.INT16_TYPE = ir.IntType(16)
        self._llvm.INT32_TYPE = ir.IntType(32)
        self._llvm.INT64_TYPE = ir.IntType(64)
        self._llvm.VOID_TYPE = ir.VoidType()

    def _add_builtins(self) -> None:
        # The C++ tutorial adds putchard() simply by defining it in the host
        # C++ code, which is then accessible to the JIT. It doesn't work as
        # simply for us; but luckily it's very easy to define new "C level"
        # functions for our JITed code to use - just emit them as LLVM IR.
        # This is what this method does.

        # Add the declaration of putchar
        putchar_ty = ir.FunctionType(
            self._llvm.INT32_TYPE, [self._llvm.INT32_TYPE]
        )
        putchar = ir.Function(self._llvm.module, putchar_ty, "putchar")

        # Add putchard
        putchard_ty = ir.FunctionType(
            self._llvm.INT32_TYPE, [self._llvm.INT32_TYPE]
        )
        putchard = ir.Function(self._llvm.module, putchard_ty, "putchard")

        ir_builder = ir.IRBuilder(putchard.append_basic_block("entry"))

        ival = ir_builder.fptoui(
            putchard.args[0], self._llvm.INT32_TYPE, "intcast"
        )

        ir_builder.call(putchar, [ival])
        ir_builder.ret(ir.Constant(self._llvm.INT32_TYPE, 0))

    def get_function(self, name: str) -> Optional[ir.Function]:
        """
        Put the function defined by the given name to result stack.

        Parameters
        ----------
            name: Function name
        """
        if name in self._llvm.module.globals:
            return self._llvm.module.get_global(name)

        if name in self.function_protos:
            self.visit(self.function_protos[name])
            return cast(ir.Function, self.result_stack.pop())

        return None

    def create_entry_block_alloca(
        self, var_name: str, type_name: str
    ) -> Any:  # llvm.AllocaInst
        """
        Create an alloca instruction in the entry block of the function.

        This is used for mutable variables, etc.

        Parameters
        ----------
        fn: The llvm function
        var_name: The variable name
        type_name: The type name

        Returns
        -------
          An llvm allocation instance.
        """
        self._llvm.ir_builder.position_at_start(
            self._llvm.ir_builder.function.entry_basic_block
        )
        alloca = self._llvm.ir_builder.alloca(
            self._llvm.get_data_type(type_name), None, var_name
        )
        self._llvm.ir_builder.position_at_end(self._llvm.ir_builder.block)
        return alloca

    def promote_operands(
        self, lhs: ir.Value, rhs: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        """
        Promote two LLVM IR numeric operands to a common type.

        Parameters
        ----------
        lhs : ir.Value
            The left-hand operand.
        rhs : ir.Value
            The right-hand operand.

        Returns
        -------
        tuple[ir.Value, ir.Value]
            A tuple containing the promoted operands.
        """
        if lhs.type == rhs.type:
            return lhs, rhs

        # perform sign extension (for integer operands)
        if isinstance(lhs.type, ir.IntType) and isinstance(
            rhs.type, ir.IntType
        ):
            if lhs.type.width < rhs.type.width:
                lhs = self._llvm.ir_builder.sext(lhs, rhs.type, "promote_lhs")
            elif lhs.type.width > rhs.type.width:
                rhs = self._llvm.ir_builder.sext(rhs, lhs.type, "promote_rhs")
            return lhs, rhs

        # ranking dictionary for floating point types
        fp_types_order = {"float": 1, "double": 2}

        lhs_str = str(lhs.type)
        rhs_str = str(rhs.type)

        # perform floating-point extension
        if lhs_str in fp_types_order and rhs_str in fp_types_order:
            if fp_types_order[lhs_str] < fp_types_order[rhs_str]:
                lhs = self._llvm.ir_builder.fpext(lhs, rhs.type, "promote_lhs")
            elif fp_types_order[lhs_str] > fp_types_order[rhs_str]:
                rhs = self._llvm.ir_builder.fpext(rhs, lhs.type, "promote_rhs")
            return lhs, rhs

        return lhs, rhs

    @dispatch.abstract
    def visit(self, node: astx.AST) -> None:
        """Translate an ASTx expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.UnaryOp) -> None:
        """Translate an ASTx UnaryOp expression."""
        if node.op_code == "++":
            self.visit(node.operand)
            operand_val = safe_pop(self.result_stack)

            one = ir.Constant(operand_val.type, 1)

            # Perform the increment operation
            result = self._llvm.ir_builder.add(operand_val, one, "inctmp")

            # If operand is a variable, store the new value back
            if isinstance(node.operand, astx.Variable):
                var_addr = self.named_values.get(node.operand.name)
                if var_addr:
                    self._llvm.ir_builder.store(result, var_addr)

            self.result_stack.append(result)
            return

        elif node.op_code == "--":
            self.visit(node.operand)
            operand_val = safe_pop(self.result_stack)
            one = ir.Constant(operand_val.type, 1)
            result = self._llvm.ir_builder.sub(operand_val, one, "dectmp")

            if isinstance(node.operand, astx.Variable):
                var_addr = self.named_values.get(node.operand.name)
                if var_addr:
                    self._llvm.ir_builder.store(result, var_addr)

            self.result_stack.append(result)
            return

        elif node.op_code == "!":
            self.visit(node.operand)
            val = safe_pop(self.result_stack)
            result = self._llvm.ir_builder.xor(
                val, ir.Constant(val.type, 1), "nottmp"
            )

            if isinstance(node.operand, astx.Variable):
                addr = self.named_values.get(node.operand.name)
                if addr:
                    self._llvm.ir_builder.store(result, addr)

            self.result_stack.append(result)
            return

        raise Exception(f"Unary operator {node.op_code} not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.BinaryOp) -> None:
        """Translate binary operation expression."""
        if node.op_code == "=":
            # Special case '=' because we don't want to emit the lhs as an
            # expression.
            # Assignment requires the lhs to be an identifier.
            # This assumes we're building without RTTI because LLVM builds
            # that way by default.
            # If you build LLVM with RTTI, this can be changed to a
            # dynamic_cast for automatic error checking.
            var_lhs = node.lhs

            if not isinstance(var_lhs, astx.VariableExprAST):
                raise Exception("destination of '=' must be a variable")

            # Codegen the rhs.
            self.visit(node.rhs)
            llvm_rhs = safe_pop(self.result_stack)

            if not llvm_rhs:
                raise Exception("codegen: Invalid rhs expression.")

            llvm_lhs = self.named_values.get(var_lhs.get_name())

            if not llvm_lhs:
                raise Exception("codegen: Invalid lhs variable name")

            self._llvm.ir_builder.store(llvm_rhs, llvm_lhs)
            result = llvm_rhs
            self.result_stack.append(result)
            return

        self.visit(node.lhs)
        llvm_lhs = safe_pop(self.result_stack)

        self.visit(node.rhs)
        llvm_rhs = safe_pop(self.result_stack)

        if not llvm_lhs or not llvm_rhs:
            raise Exception("codegen: Invalid lhs/rhs")

        # automatic type promotion
        llvm_lhs, llvm_rhs = self.promote_operands(llvm_lhs, llvm_rhs)

        if node.op_code == "&&":
            result = self._llvm.ir_builder.and_(llvm_lhs, llvm_rhs, "andtmp")
            self.result_stack.append(result)
            return
        elif node.op_code == "||":
            result = self._llvm.ir_builder.or_(llvm_lhs, llvm_rhs, "ortmp")
            self.result_stack.append(result)
            return

        if node.op_code == "+":
            # note: it should be according the datatype,
            #       e.g. for float it should be fadd
            if self._llvm.FLOAT_TYPE in (llvm_lhs.type, llvm_rhs.type):
                result = self._llvm.ir_builder.fadd(
                    llvm_lhs, llvm_rhs, "addtmp"
                )
            else:
                # there's more conditions to be handled
                result = self._llvm.ir_builder.add(
                    llvm_lhs, llvm_rhs, "addtmp"
                )
            self.result_stack.append(result)
            return
        elif node.op_code == "-":
            # note: it should be according the datatype,
            if self._llvm.FLOAT_TYPE in (llvm_lhs.type, llvm_rhs.type):
                result = self._llvm.ir_builder.fsub(
                    llvm_lhs, llvm_rhs, "subtmp"
                )
            else:
                # note: be careful you should handle this as  INT32
                result = self._llvm.ir_builder.sub(
                    llvm_lhs, llvm_rhs, "subtmp"
                )
            self.result_stack.append(result)
            return
        elif node.op_code == "*":
            # note: it should be according the datatype,
            #       e.g. for float it should be fmul
            if self._llvm.FLOAT_TYPE in (llvm_lhs.type, llvm_rhs.type):
                result = self._llvm.ir_builder.fmul(
                    llvm_lhs, llvm_rhs, "multmp"
                )
            else:
                # note: be careful you should handle this as INT32
                result = self._llvm.ir_builder.mul(
                    llvm_lhs, llvm_rhs, "multmp"
                )
            self.result_stack.append(result)
            return
        elif node.op_code == "<":
            # note: it should be according the datatype,
            #       e.g. for float it should be fcmp
            if self._llvm.FLOAT_TYPE in (llvm_lhs.type, llvm_rhs.type):
                cmp_result = self._llvm.ir_builder.fcmp_ordered(
                    "<", llvm_lhs, llvm_rhs, "lttmp"
                )
                result = self._llvm.ir_builder.zext(
                    cmp_result, self._llvm.FLOAT_TYPE, "booltmp"
                )
            else:
                # handle it depend on datatype
                cmp_result = self._llvm.ir_builder.icmp_signed(
                    "<", llvm_lhs, llvm_rhs, "lttmp"
                )
                result = self._llvm.ir_builder.zext(
                    cmp_result, self._llvm.INT32_TYPE, "booltmp"
                )
            self.result_stack.append(cmp_result)
            return
        elif node.op_code == ">":
            # note: it should be according the datatype,
            #       e.g. for float it should be fcmp
            if self._llvm.FLOAT_TYPE in (llvm_lhs.type, llvm_rhs.type):
                cmp_result = self._llvm.ir_builder.fcmp_ordered(
                    ">", llvm_lhs, llvm_rhs, "gttmp"
                )
                result = self._llvm.ir_builder.uitofp(
                    cmp_result, self._llvm.FLOAT_TYPE, "booltmp"
                )
            else:
                # be careful we havn't  handled all the conditions
                cmp_result = self._llvm.ir_builder.icmp_signed(
                    ">", llvm_lhs, llvm_rhs, "gttmp"
                )
                result = self._llvm.ir_builder.zext(
                    cmp_result, self._llvm.INT32_TYPE, "booltmp"
                )
            self.result_stack.append(cmp_result)
            return
        elif node.op_code == "<=":
            if self._llvm.FLOAT_TYPE in (llvm_lhs.type, llvm_rhs.type):
                cmp_result = self._llvm.ir_builder.fcmp_ordered(
                    "<=", llvm_lhs, llvm_rhs, "letmp"
                )
                result = self._llvm.ir_builder.zext(
                    cmp_result, self._llvm.FLOAT_TYPE, "booltmp"
                )
            else:
                cmp_result = self._llvm.ir_builder.icmp_signed(
                    "<=", llvm_lhs, llvm_rhs, "letmp"
                )
                result = self._llvm.ir_builder.zext(
                    cmp_result, self._llvm.INT32_TYPE, "booltmp"
                )
            self.result_stack.append(cmp_result)
            return
        elif node.op_code == ">=":
            if self._llvm.FLOAT_TYPE in (llvm_lhs.type, llvm_rhs.type):
                cmp_result = self._llvm.ir_builder.fcmp_ordered(
                    ">=", llvm_lhs, llvm_rhs, "getmp"
                )
                result = self._llvm.ir_builder.zext(
                    cmp_result, self._llvm.FLOAT_TYPE, "booltmp"
                )
            else:
                cmp_result = self._llvm.ir_builder.icmp_signed(
                    ">=", llvm_lhs, llvm_rhs, "getmp"
                )
                result = self._llvm.ir_builder.zext(
                    cmp_result, self._llvm.INT32_TYPE, "booltmp"
                )
            self.result_stack.append(cmp_result)
            return
        elif node.op_code == "/":
            # Check the datatype to decide between floating-point and integer
            # division
            if self._llvm.FLOAT_TYPE in (llvm_lhs.type, llvm_rhs.type):
                # Floating-point division
                result = self._llvm.ir_builder.fdiv(
                    llvm_lhs, llvm_rhs, "divtmp"
                )
            else:
                # Assuming the division is signed by default. Use `udiv` for
                # unsigned division.
                result = self._llvm.ir_builder.sdiv(
                    llvm_lhs, llvm_rhs, "divtmp"
                )
            self.result_stack.append(result)
            return

        raise Exception(f"Binary op {node.op_code} not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, block: astx.Block) -> None:
        """Translate ASTx Block to LLVM-IR."""
        result = None
        for node in block.nodes:
            self.visit(node)
            try:
                result = self.result_stack.pop()
            except IndexError:
                # some nodes doesn't add anything in the stack
                pass
        if result is not None:
            self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.IfStmt) -> None:
        """Translate IF statement."""
        self.visit(node.condition)
        cond_v = self.result_stack.pop()
        if not cond_v:
            raise Exception("codegen: Invalid condition expression.")

        if isinstance(cond_v.type, (ir.FloatType, ir.DoubleType)):
            cmp_instruction = self._llvm.ir_builder.fcmp_ordered
            zero_val = ir.Constant(cond_v.type, 0.0)
        else:
            cmp_instruction = self._llvm.ir_builder.icmp_signed
            zero_val = ir.Constant(cond_v.type, 0)

        cond_v = cmp_instruction(
            "!=",
            cond_v,
            zero_val,
        )

        # Create blocks for the then and else cases.
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

        # Emit then value.
        self._llvm.ir_builder.position_at_start(then_bb)
        self.visit(node.then)
        then_v = self.result_stack.pop()
        if not then_v:
            raise Exception("codegen: `Then` expression is invalid.")

        self._llvm.ir_builder.branch(merge_bb)

        # Update reference to final block of 'then'
        then_bb = self._llvm.ir_builder.block

        # Emit else block.
        self._llvm.ir_builder.position_at_start(else_bb)
        else_v = None
        if node.else_ is not None:
            self.visit(node.else_)
            else_v = self.result_stack.pop()
        else:
            else_v = ir.Constant(self._llvm.INT32_TYPE, 0)

        # Update reference to final block of 'else'
        else_bb = self._llvm.ir_builder.block
        self._llvm.ir_builder.branch(merge_bb)

        # Emit merge block and PHI node
        self._llvm.ir_builder.position_at_start(merge_bb)
        phi = self._llvm.ir_builder.phi(self._llvm.INT32_TYPE, "iftmp")
        phi.add_incoming(then_v, then_bb)
        phi.add_incoming(else_v, else_bb)

        self.result_stack.append(phi)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.WhileStmt) -> None:
        """Translate ASTx While Loop to LLVM-IR."""
        # Create blocks for the condition check, the loop body,
        # and the block after the loop.
        cond_bb = self._llvm.ir_builder.function.append_basic_block(
            "whilecond"
        )
        body_bb = self._llvm.ir_builder.function.append_basic_block(
            "whilebody"
        )
        after_bb = self._llvm.ir_builder.function.append_basic_block(
            "afterwhile"
        )

        # Branch to the condition check block.
        self._llvm.ir_builder.branch(cond_bb)

        # Start inserting into the condition check block.
        self._llvm.ir_builder.position_at_start(cond_bb)

        # Emit the condition.
        self.visit(expr.condition)
        cond_val = self.result_stack.pop()
        if not cond_val:
            raise Exception("codegen: Invalid condition expression.")

        # Convert condition to a bool by comparing non-equal to 0.
        if isinstance(cond_val.type, (ir.FloatType, ir.DoubleType)):
            cmp_instruction = self._llvm.ir_builder.fcmp_ordered
            zero_val = ir.Constant(cond_val.type, 0.0)
        else:
            cmp_instruction = self._llvm.ir_builder.icmp_signed
            zero_val = ir.Constant(cond_val.type, 0)

        cond_val = cmp_instruction(
            "!=",
            cond_val,
            zero_val,
            "whilecond",
        )

        # Conditional branch based on the condition.
        self._llvm.ir_builder.cbranch(cond_val, body_bb, after_bb)

        # Start inserting into the loop body block.
        self._llvm.ir_builder.position_at_start(body_bb)

        # Emit the body of the loop.
        self.visit(expr.body)
        body_val = self.result_stack.pop()

        if not body_val:
            return

        # Branch back to the condition check.
        self._llvm.ir_builder.branch(cond_bb)

        # Start inserting into the block after the loop.
        self._llvm.ir_builder.position_at_start(after_bb)

        # While loop always returns 0.
        result = ir.Constant(self._llvm.INT32_TYPE, 0)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.VariableAssignment) -> None:
        """Translate variable assignment expression."""
        # Get the name of the variable to assign to
        var_name = expr.name

        # Codegen the value expression on the right-hand side
        self.visit(expr.value)
        llvm_value = safe_pop(self.result_stack)

        if not llvm_value:
            raise Exception("codegen: Invalid value in VariableAssignment.")

        # Look up the variable in the named values
        llvm_var = self.named_values.get(var_name)

        if not llvm_var:
            raise Exception(
                f"Variable '{var_name}' not found in the named values."
            )

        # Store the value in the variable
        self._llvm.ir_builder.store(llvm_value, llvm_var)

        # Optionally, you can push the result onto the result stack if needed
        self.result_stack.append(llvm_value)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.ForCountLoopStmt) -> None:
        """Translate ASTx For Range Loop to LLVM-IR."""
        saved_block = self._llvm.ir_builder.block
        var_addr = self.create_entry_block_alloca(
            "for_count_loop", node.initializer.type_.__class__.__name__.lower()
        )
        self._llvm.ir_builder.position_at_end(saved_block)

        # Emit the start code first, without 'variable' in scope.
        self.visit(node.initializer)
        initializer_val = self.result_stack.pop()
        if not initializer_val:
            raise Exception("codegen: Invalid start argument.")

        # Store the value into the alloca.
        self._llvm.ir_builder.store(initializer_val, var_addr)

        loop_header_bb = self._llvm.ir_builder.function.append_basic_block(
            "loop.header"
        )
        self._llvm.ir_builder.branch(loop_header_bb)

        # Start insertion in loop header
        self._llvm.ir_builder.position_at_start(loop_header_bb)

        # Save old value if variable shadows an existing one
        old_val = self.named_values.get(node.initializer.name)
        self.named_values[node.initializer.name] = var_addr

        # Emit condition check (e.g., i < 10)
        self.visit(node.condition)
        cond_val = self.result_stack.pop()

        # Create blocks for loop body and after loop
        loop_body_bb = self._llvm.ir_builder.function.append_basic_block(
            "loop.body"
        )
        after_loop_bb = self._llvm.ir_builder.function.append_basic_block(
            "after.loop"
        )

        # Branch based on condition
        self._llvm.ir_builder.cbranch(cond_val, loop_body_bb, after_loop_bb)

        # Emit loop body
        self._llvm.ir_builder.position_at_start(loop_body_bb)
        self.visit(node.body)
        _body_val = self.result_stack.pop()

        # Emit update expression
        self.visit(node.update)
        update_val = self.result_stack.pop()

        # Store updated value
        self._llvm.ir_builder.store(update_val, var_addr)

        # Branch back to loop header
        self._llvm.ir_builder.branch(loop_header_bb)

        # Move to after-loop block
        self._llvm.ir_builder.position_at_start(after_loop_bb)

        # Restore the unshadowed variable.
        if old_val:
            self.named_values[node.initializer.name] = old_val
        else:
            self.named_values.pop(node.initializer.name, None)

        result = ir.Constant(
            self._llvm.get_data_type(
                node.initializer.type_.__class__.__name__.lower()
            ),
            0,
        )
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.ForRangeLoopStmt) -> None:
        """Translate ASTx For Range Loop to LLVM-IR."""
        saved_block = self._llvm.ir_builder.block
        var_addr = self.create_entry_block_alloca(
            "for_count_loop", node.variable.type_.__class__.__name__.lower()
        )
        self._llvm.ir_builder.position_at_end(saved_block)

        # Emit the start code first, without 'variable' in scope.
        self.visit(node.start)
        start_val = self.result_stack.pop()
        if not start_val:
            raise Exception("codegen: Invalid start argument.")
        self._llvm.ir_builder.store(start_val, var_addr)

        # Create and jump to the loop block
        loop_bb = self._llvm.ir_builder.function.append_basic_block("loop")
        self._llvm.ir_builder.branch(loop_bb)
        self._llvm.ir_builder.position_at_start(loop_bb)

        # Store current var in scope
        old_val = self.named_values.get(node.variable.name)
        self.named_values[node.variable.name] = var_addr

        # Emit the body of the loop.
        self.visit(node.body)
        body_val = self.result_stack.pop()

        if not body_val:
            return

        # Emit the step value.
        if node.step:
            self.visit(node.step)
            step_val = self.result_stack.pop()
            if not step_val:
                return
        else:
            step_val = ir.Constant(
                self._llvm.get_data_type(
                    node.variable.type_.__class__.__name__.lower()
                ),
                1,
            )

        # Compute the end condition.
        self.visit(node.end)
        end_cond = self.result_stack.pop()
        if not end_cond:
            return

        # Increment loop variable: i = i + step
        cur_var = self._llvm.ir_builder.load(var_addr, node.variable.name)
        next_var = self._llvm.ir_builder.add(cur_var, step_val, "nextvar")
        self._llvm.ir_builder.store(next_var, var_addr)

        if isinstance(end_cond.type, (ir.FloatType, ir.DoubleType)):
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

        end_cond = cmp_instruction(
            cmp_op,
            cur_var,
            end_cond,
            "loopcond",
        )

        # Create the "after loop" block and insert it.
        after_bb = self._llvm.ir_builder.function.append_basic_block(
            "afterloop"
        )

        # Insert the conditional branch into the end of loop_bb.
        self._llvm.ir_builder.cbranch(end_cond, loop_bb, after_bb)

        # Any new code will be inserted in after_bb.
        self._llvm.ir_builder.position_at_start(after_bb)

        # Restore the unshadowed variable.
        if old_val:
            self.named_values[node.variable.name] = old_val
        else:
            self.named_values.pop(node.variable.name, None)

        # for node always returns 0.0.
        result = ir.Constant(
            self._llvm.get_data_type(
                node.variable.type_.__class__.__name__.lower()
            ),
            0,
        )
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.Module) -> None:
        """Translate ASTx Module to LLVM-IR."""
        for mod_node in node.nodes:
            self.visit(mod_node)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralInt32) -> None:
        """Translate ASTx LiteralInt32 to LLVM-IR."""
        result = ir.Constant(self._llvm.INT32_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralFloat32) -> None:
        """Translate ASTx LiteralFloat32 to LLVM-IR."""
        result = ir.Constant(self._llvm.FLOAT_TYPE, expr.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralFloat16) -> None:
        """Translate ASTx LiteralFloat16 to LLVM-IR."""
        result = ir.Constant(self._llvm.FLOAT16_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralNone) -> None:
        """Translate ASTx LiteralNone to LLVM-IR."""
        self.result_stack.append(None)  # No IR emitted for void

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralBoolean) -> None:
        """Translate ASTx LiteralBoolean to LLVM-IR."""
        result = ir.Constant(self._llvm.BOOLEAN_TYPE, int(node.value))
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralInt64) -> None:
        """Translate ASTx LiteralInt64 to LLVM-IR."""
        result = ir.Constant(self._llvm.INT64_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralInt8) -> None:
        """Translate ASTx LiteralInt8 to LLVM-IR."""
        result = ir.Constant(self._llvm.INT8_TYPE, node.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.FunctionCall) -> None:
        """Translate Function FunctionCall."""
        # callee_f = self.get_function(node.fn)
        fn_name = node.fn

        callee_f = self.get_function(fn_name)
        if not callee_f:
            raise Exception("Unknown function referenced")

        if len(callee_f.args) != len(node.args):
            raise Exception("codegen: Incorrect # arguments passed.")

        llvm_args = []
        for arg in node.args:
            self.visit(arg)
            llvm_arg = self.result_stack.pop()
            if not llvm_arg:
                raise Exception("codegen: Invalid callee argument.")
            llvm_args.append(llvm_arg)

        result = self._llvm.ir_builder.call(callee_f, llvm_args, "calltmp")
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.FunctionDef) -> None:
        """Translate ASTx Function to LLVM-IR."""
        proto = node.prototype
        self.function_protos[proto.name] = proto
        fn = self.get_function(proto.name)

        if not fn:
            raise Exception("Invalid function.")

        # Create a new basic block to start insertion into.
        basic_block = fn.append_basic_block("entry")
        self._llvm.ir_builder = ir.IRBuilder(basic_block)

        for idx, llvm_arg in enumerate(fn.args):
            arg_ast = proto.args.nodes[idx]
            type_str = arg_ast.type_.__class__.__name__.lower()
            arg_type = self._llvm.get_data_type(type_str)

            # Create an alloca for this variable.
            alloca = self._llvm.ir_builder.alloca(arg_type, name=llvm_arg.name)

            # Store the initial value into the alloca.
            self._llvm.ir_builder.store(llvm_arg, alloca)

            # Add arguments to variable symbol table.
            self.named_values[llvm_arg.name] = alloca

        self.visit(node.body)
        self.result_stack.append(fn)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.FunctionPrototype) -> None:
        """Translate ASTx Function Prototype to LLVM-IR."""
        args_type = []
        for arg in node.args.nodes:
            type_str = arg.type_.__class__.__name__.lower()
            args_type.append(self._llvm.get_data_type(type_str))
        # note: it should be dynamic
        return_type = self._llvm.get_data_type(
            node.return_type.__class__.__name__.lower()
        )
        fn_type = ir.FunctionType(return_type, args_type, False)

        fn = ir.Function(self._llvm.module, fn_type, node.name)

        # Set names for all arguments.
        for idx, llvm_arg in enumerate(fn.args):
            llvm_arg.name = node.args.nodes[idx].name

        self.result_stack.append(fn)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.FunctionReturn) -> None:
        """Translate ASTx FunctionReturn to LLVM-IR."""
        if node.value is not None:
            self.visit(node.value)
            try:
                retval = self.result_stack.pop()
            except IndexError:
                retval = None
        else:
            retval = None

        if retval is not None:
            fn_return_type = (
                self._llvm.ir_builder.function.function_type.return_type
            )
            if (
                isinstance(fn_return_type, ir.IntType)
                and fn_return_type.width == 1
            ):
                # Force cast retval to i1 if not already
                if (
                    isinstance(retval.type, ir.IntType)
                    and retval.type.width != 1
                ):
                    retval = self._llvm.ir_builder.trunc(retval, ir.IntType(1))
            self._llvm.ir_builder.ret(retval)
            return
        self._llvm.ir_builder.ret_void()

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.InlineVariableDeclaration) -> None:
        """Translate an ASTx InlineVariableDeclaration expression."""
        if self.named_values.get(node.name):
            raise Exception(f"Variable already declared: {node.name}")

        type_str = node.type_.__class__.__name__.lower()

        # Emit the initializer
        if node.value is not None:
            self.visit(node.value)
            init_val = self.result_stack.pop()
            if init_val is None:
                raise Exception("Initializer code generation failed.")
        # Default zero value based on type
        elif "float" in type_str:
            init_val = ir.Constant(self._llvm.get_data_type(type_str), 0.0)
        else:
            init_val = ir.Constant(self._llvm.get_data_type(type_str), 0)

        alloca = self.create_entry_block_alloca(node.name, type_str)
        self._llvm.ir_builder.store(init_val, alloca)
        self.named_values[node.name] = alloca

        self.result_stack.append(init_val)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: system.Cast) -> None:
        """Translate Cast expression to LLVM-IR."""
        self.visit(node.value)
        value = self.result_stack.pop()

        target_type_str = node.target_type.__class__.__name__.lower()
        target_type = self._llvm.get_data_type(target_type_str)

        if value.type == target_type:
            self.result_stack.append(value)
            return

        result: ir.Value

        if isinstance(value.type, ir.IntType) and isinstance(
            target_type, ir.IntType
        ):
            if value.type.width < target_type.width:
                result = self._llvm.ir_builder.sext(
                    value, target_type, "cast_int_up"
                )
            else:
                result = self._llvm.ir_builder.trunc(
                    value, target_type, "cast_int_down"
                )
        elif isinstance(value.type, ir.IntType) and isinstance(
            target_type, ir.FloatType
        ):
            result = self._llvm.ir_builder.sitofp(
                value, target_type, "cast_int_to_fp"
            )

        elif isinstance(value.type, ir.FloatType) and isinstance(
            target_type, ir.IntType
        ):
            result = self._llvm.ir_builder.fptosi(
                value, target_type, "cast_fp_to_int"
            )

        elif isinstance(value.type, ir.FloatType) and isinstance(
            target_type, ir.HalfType
        ):
            result = self._llvm.ir_builder.fptrunc(
                value, target_type, "cast_fp_to_half"
            )

        elif isinstance(value.type, ir.HalfType) and isinstance(
            target_type, ir.FloatType
        ):
            result = self._llvm.ir_builder.fpext(
                value, target_type, "cast_half_to_fp"
            )

        elif isinstance(value.type, ir.FloatType) and isinstance(
            target_type, ir.FloatType
        ):
            if value.type.width < target_type.width:
                result = self._llvm.ir_builder.fpext(
                    value, target_type, "cast_fp_up"
                )

            else:
                result = self._llvm.ir_builder.fptrunc(
                    value, target_type, "cast_fp_down"
                )

        else:
            raise Exception(
                f"Unsupported cast from {value.type} to {target_type}"
            )

        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: system.PrintExpr) -> None:
        """Generate LLVM IR for a PrintExpr node."""
        message = node.message.value

        msg_length = len(message) + 1

        msg_type = ir.ArrayType(self._llvm.INT8_TYPE, msg_length)

        global_msg = ir.GlobalVariable(
            self._llvm.module, msg_type, name=node._name
        )
        global_msg.linkage = "internal"
        global_msg.global_constant = True
        global_msg.initializer = ir.Constant(
            msg_type, bytearray(message + "\0", "utf8")
        )

        ptr = self._llvm.ir_builder.gep(
            global_msg,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)],
            inbounds=True,
        )

        puts_fn = self._llvm.module.globals.get("puts")
        if puts_fn is None:
            puts_ty = ir.FunctionType(
                self._llvm.INT32_TYPE, [ir.PointerType(self._llvm.INT8_TYPE)]
            )
            puts_fn = ir.Function(self._llvm.module, puts_ty, name="puts")

        self._llvm.ir_builder.call(puts_fn, [ptr])

        self.result_stack.append(ir.Constant(self._llvm.INT32_TYPE, 0))

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.Variable) -> None:
        """Translate ASTx Variable to LLVM-IR."""
        expr_var = self.named_values.get(node.name)

        if not expr_var:
            raise Exception(f"Unknown variable name: {node.name}")

        result = self._llvm.ir_builder.load(expr_var, node.name)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.VariableDeclaration) -> None:
        """Translate ASTx Variable to LLVM-IR."""
        if self.named_values.get(node.name):
            raise Exception(f"Variable already declared: {node.name}")

        type_str = node.type_.__class__.__name__.lower()

        # Emit the initializer
        if node.value is not None:
            self.visit(node.value)
            init_val = self.result_stack.pop()
            if init_val is None:
                raise Exception("Initializer code generation failed.")
        else:
            # If not specified, use 0 as the initializer.
            # note: it should create something according to the defined type
            init_val = ir.Constant(self._llvm.get_data_type(type_str), 0)

        # Create an alloca in the entry block.
        # note: it should create the type according to the defined type
        alloca = self.create_entry_block_alloca(node.name, type_str)

        # Store the initial value.
        self._llvm.ir_builder.store(init_val, alloca)

        # Remember this binding.
        self.named_values[node.name] = alloca

    @dispatch  # type: ignore[no-redef]
    def visit(self, node: astx.LiteralInt16) -> None:
        """Translate ASTx LiteralInt16 to LLVM-IR."""
        result = ir.Constant(self._llvm.INT16_TYPE, node.value)
        self.result_stack.append(result)


@public
class LLVMLiteIR(Builder):
    """LLVM-IR transpiler and compiler."""

    def __init__(self) -> None:
        """Initialize LLVMIR."""
        super().__init__()
        self.translator: LLVMLiteIRVisitor = LLVMLiteIRVisitor()

    def build(self, node: astx.AST, output_file: str) -> None:
        """Transpile the ASTx to LLVM-IR and build it to an executable file."""
        self.translator = LLVMLiteIRVisitor()
        result = self.translator.translate(node)

        result_mod = llvm.parse_assembly(result)
        result_object = self.translator.target_machine.emit_object(result_mod)

        with tempfile.NamedTemporaryFile(suffix="", delete=True) as temp_file:
            self.tmp_path = temp_file.name

        file_path_o = f"{self.tmp_path}.o"

        with open(file_path_o, "wb") as f:
            f.write(result_object)

        self.output_file = output_file

        # fix xh typing
        clang: Callable[..., Any] = xh.clang

        clang(
            file_path_o,
            "-o",
            self.output_file,
        )
        os.chmod(self.output_file, 0o755)

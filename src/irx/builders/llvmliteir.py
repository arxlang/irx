"""LLVM-IR builder."""

from __future__ import annotations

import subprocess
import tempfile

from typing import Any, Optional, cast

import astx
import sh

from llvmlite import binding as llvm
from llvmlite import ir
from plum import dispatch
from public import public

from irx.builders.base import Builder, BuilderVisitor


def run_command(command: list[str]) -> None:
    """Run a command in the operating system."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        # Handle the error as needed


def safe_pop(lst: list[ir.Value | ir.Function]) -> ir.Value | ir.Function:
    """Implement a safe pop operation for lists."""
    try:
        return lst.pop()
    except IndexError:
        return None


class VariablesLLVM:
    """Store all the LLVM variables that is used for the code generation."""

    FLOAT_TYPE: ir.types.Type
    DOUBLE_TYPE: ir.types.Type
    INT8_TYPE: ir.types.Type
    INT16_TYPE: ir.types.Type
    INT32_TYPE: ir.types.Type
    VOID_TYPE: ir.types.Type

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
        if type_name == "float":
            return self.FLOAT_TYPE
        elif type_name == "double":
            return self.DOUBLE_TYPE
        elif type_name == "int8":
            return self.INT8_TYPE
        elif type_name == "int16":
            return self.INT16_TYPE
        elif type_name == "int32":
            return self.INT32_TYPE
        elif type_name == "char":
            return self.INT8_TYPE
        elif type_name == "void":
            return self.VOID_TYPE

        raise Exception("[EE]: type_name not valid.")


class LLVMLiteIRVisitor(BuilderVisitor):
    """LLVM-IR Translator."""

    # AllocaInst

    _llvm: VariablesLLVM

    function_protos: dict[str, astx.FunctionPrototype]
    result_stack: list[ir.Value | ir.Function] = []  # noqa: RUF012

    def __init__(self) -> None:
        """Initialize LLVMTranslator object."""
        super().__init__()
        # named_values as instance variable so it isn't shared across instances
        self.named_values: dict[str, Any] = {}
        self.function_protos = {}
        self.result_stack = []

        self.initialize()

        self.target = llvm.Target.from_default_triple()
        self.target_machine = self.target.create_target_machine(
            codemodel="small"
        )

        self._add_builtins()

    def translate(self, expr: astx.AST) -> str:
        """Translate an ASTx expression to string."""
        self.visit(expr)
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
        self._llvm.DOUBLE_TYPE = ir.DoubleType()
        self._llvm.INT8_TYPE = ir.IntType(8)
        self._llvm.INT16_TYPE = ir.IntType(16)
        self._llvm.INT32_TYPE = ir.IntType(32)
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

    @dispatch.abstract
    def visit(self, expr: astx.AST) -> None:
        """Translate an ASTx expression."""
        raise Exception("Not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.UnaryOp) -> None:
        """Translate an ASTx UnaryOp expression."""
        if expr.op_code == "++":
            self.visit(expr.operand)
            operand_val = safe_pop(self.result_stack)

            one = ir.Constant(operand_val.type, 1)

            # Perform the increment operation
            result = self._llvm.ir_builder.add(operand_val, one, "inctmp")

            # If operand is a variable, store the new value back
            if isinstance(expr.operand, astx.Variable):
                var_addr = self.named_values.get(expr.operand.name)
                if var_addr:
                    self._llvm.ir_builder.store(result, var_addr)

            self.result_stack.append(result)
            return

        elif expr.op_code == "--":
            self.visit(expr.operand)
            operand_val = safe_pop(self.result_stack)
            one = ir.Constant(operand_val.type, 1)
            result = self._llvm.ir_builder.sub(operand_val, one, "dectmp")

            if isinstance(expr.operand, astx.Variable):
                var_addr = self.named_values.get(expr.operand.name)
                if var_addr:
                    self._llvm.ir_builder.store(result, var_addr)

            self.result_stack.append(result)
            return

        raise Exception(f"Unary operator {expr.op_code} not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.BinaryOp) -> None:
        """Translate binary operation expression."""
        if expr.op_code == "=":
            # Special case '=' because we don't want to emit the lhs as an
            # expression.
            # Assignment requires the lhs to be an identifier.
            # This assumes we're building without RTTI because LLVM builds
            # that way by default.
            # If you build LLVM with RTTI, this can be changed to a
            # dynamic_cast for automatic error checking.
            var_lhs = expr.lhs

            if not isinstance(var_lhs, astx.VariableExprAST):
                raise Exception("destination of '=' must be a variable")

            # Codegen the rhs.
            self.visit(expr.rhs)
            llvm_rhs = safe_pop(self.result_stack)

            if not llvm_rhs:
                raise Exception("codegen: Invalid rhs expression.")

            # Look up the name.
            llvm_lhs = self.named_values.get(var_lhs.get_name())

            if not llvm_lhs:
                raise Exception("codegen: Invalid lhs variable name")

            self._llvm.ir_builder.store(llvm_rhs, llvm_lhs)
            result = llvm_rhs
            self.result_stack.append(result)
            return

        self.visit(expr.lhs)
        llvm_lhs = safe_pop(self.result_stack)

        self.visit(expr.rhs)
        llvm_rhs = safe_pop(self.result_stack)

        if not llvm_lhs or not llvm_rhs:
            raise Exception("codegen: Invalid lhs/rhs")

        if expr.op_code == "+":
            # note: it should be according the datatype,
            #       e.g. for float it should be fadd
            result = self._llvm.ir_builder.add(llvm_lhs, llvm_rhs, "addtmp")
            self.result_stack.append(result)
            return
        elif expr.op_code == "-":
            # note: it should be according the datatype,
            #       e.g. for float it should be fsub
            result = self._llvm.ir_builder.sub(llvm_lhs, llvm_rhs, "subtmp")
            self.result_stack.append(result)
            return
        elif expr.op_code == "*":
            # note: it should be according the datatype,
            #       e.g. for float it should be fmul
            result = self._llvm.ir_builder.mul(llvm_lhs, llvm_rhs, "multmp")
            self.result_stack.append(result)
            return
        elif expr.op_code == "<":
            # note: it should be according the datatype,
            #       e.g. for float it should be fcmp
            cmp_result = self._llvm.ir_builder.icmp_signed(
                "<", llvm_lhs, llvm_rhs, "lttmp"
            )
            # result = self._llvm.ir_builder.zext(
            #     cmp_result, self._llvm.INT32_TYPE, "booltmp"
            # )
            self.result_stack.append(cmp_result)
            return
        elif expr.op_code == ">":
            # note: it should be according the datatype,
            #       e.g. for float it should be fcmp
            cmp_result = self._llvm.ir_builder.icmp_signed(
                ">", llvm_lhs, llvm_rhs, "gttmp"
            )
            # result = self._llvm.ir_builder.zext(
            #     cmp_result, self._llvm.INT32_TYPE, "booltmp"
            # )
            self.result_stack.append(cmp_result)
            return
        elif expr.op_code == "/":
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

        raise Exception(f"Binary op {expr.op_code} not implemented yet.")

    @dispatch  # type: ignore[no-redef]
    def visit(self, block: astx.Block) -> None:
        """Translate ASTx Block to LLVM-IR."""
        result = []
        for node in block.nodes:
            self.visit(node)
            try:
                result.append(self.result_stack.pop())
            except IndexError:
                # some nodes doesn't add anything in the stack
                pass
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.IfStmt) -> None:
        """Translate IF statement."""
        self.visit(expr.cond)
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

        # fn = self._llvm.ir_builder.position_at_start().getParent()

        # Create blocks for the then and else cases. Insert the 'then' block
        # at the end of the function.
        # then_bb = ir.Block(self._llvm.ir_builder.function, "then", fn)
        then_bb = self._llvm.ir_builder.function.append_basic_block("then")
        else_bb = ir.Block(self._llvm.ir_builder.function, "else")
        merge_bb = ir.Block(self._llvm.ir_builder.function, "ifcont")

        self._llvm.ir_builder.cbranch(cond_v, then_bb, else_bb)

        # Emit then value.
        self._llvm.ir_builder.position_at_start(then_bb)
        self.visit(expr.then_)
        then_v = self.result_stack.pop()

        if not then_v:
            raise Exception("codegen: `Then` expression is invalid.")

        self._llvm.ir_builder.branch(merge_bb)

        # Codegen of 'then' can change the current block, update then_bb
        # for the PHI.
        then_bb = self._llvm.ir_builder.block

        # Emit else block.
        self._llvm.ir_builder.function.basic_blocks.append(else_bb)
        self._llvm.ir_builder.position_at_start(else_bb)
        self.visit(expr.else_)
        else_v = self.result_stack.pop()
        if not else_v:
            raise Exception("Revisit this!")

        # Emission of else_val could have modified the current basic block.
        else_bb = self._llvm.ir_builder.block
        self._llvm.ir_builder.branch(merge_bb)

        # Emit merge block.
        self._llvm.ir_builder.function.basic_blocks.append(merge_bb)
        self._llvm.ir_builder.position_at_start(merge_bb)
        phi = self._llvm.ir_builder.phi(self._llvm.INT32_TYPE, "iftmp")

        phi.add_incoming(then_v, then_bb)
        phi.add_incoming(else_v, else_bb)

        self.result_stack.append(phi)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.ForCountLoopStmt) -> None:
        """Translate ASTx For Range Loop to LLVM-IR."""
        saved_block = self._llvm.ir_builder.block
        var_addr = self.create_entry_block_alloca("for_count_loop", "int32")
        self._llvm.ir_builder.position_at_end(saved_block)

        # Emit the start code first, without 'variable' in scope.
        self.visit(expr.initializer)
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
        old_val = self.named_values.get(expr.initializer.name)
        self.named_values[expr.initializer.name] = var_addr

        # Emit condition check (e.g., i < 10)
        self.visit(expr.condition)
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
        self.visit(expr.body)
        _body_val = self.result_stack.pop()

        # Emit update expression
        self.visit(expr.update)
        update_val = self.result_stack.pop()

        # Store updated value
        self._llvm.ir_builder.store(update_val, var_addr)

        # Branch back to loop header
        self._llvm.ir_builder.branch(loop_header_bb)

        # Move to after-loop block
        self._llvm.ir_builder.position_at_start(after_loop_bb)

        # Restore the unshadowed variable.
        if old_val:
            self.named_values[expr.initializer.name] = old_val
        else:
            self.named_values.pop(expr.initializer.name, None)

        result = ir.Constant(self._llvm.INT32_TYPE, 0)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.ForRangeLoopStmt) -> None:
        """Translate ASTx For Range Loop to LLVM-IR."""
        saved_block = self._llvm.ir_builder.block
        var_addr = self.create_entry_block_alloca("for_count_loop", "int32")
        self._llvm.ir_builder.position_at_end(saved_block)

        # Emit the start code first, without 'variable' in scope.
        self.visit(expr.start)
        start_val = self.result_stack.pop()
        if not start_val:
            raise Exception("codegen: Invalid start argument.")

        # Store the value into the alloca.
        self._llvm.ir_builder.store(start_val, var_addr)

        # Make the new basic block for the loop header, inserting after
        # current block.
        loop_bb = self._llvm.ir_builder.function.append_basic_block("loop")

        # Insert an explicit fall through from the current block to the
        # loop_bb.
        self._llvm.ir_builder.branch(loop_bb)

        # Start insertion in loop_bb.
        self._llvm.ir_builder.position_at_start(loop_bb)

        # Within the loop, the variable is defined equal to the PHI node.
        # If it shadows an existing variable, we have to restore it, so save
        # it now.
        old_val = self.named_values.get(expr.variable.name)
        self.named_values[expr.variable.name] = var_addr

        # Emit the body of the loop. This, like any other expr, can change
        # the current basic_block. Note that we ignore the value computed by
        # the body, but don't allow an error.
        self.visit(expr.body)
        body_val = self.result_stack.pop()

        if not body_val:
            return

        # Emit the step value.
        if expr.step:
            self.visit(expr.step)
            step_val = self.result_stack.pop()
            if not step_val:
                return
        else:
            # If not specified, use 1.0.
            step_val = ir.Constant(self._llvm.INT32_TYPE, 1)

        # Compute the end condition.
        self.visit(expr.end)
        end_cond = self.result_stack.pop()
        if not end_cond:
            return

        # Reload, increment, and restore the var_addr. This handles the case
        # where the body of the loop mutates the variable.
        cur_var = self._llvm.ir_builder.load(var_addr, expr.variable.name)
        next_var = self._llvm.ir_builder.add(cur_var, step_val, "nextvar")
        self._llvm.ir_builder.store(next_var, var_addr)

        # Convert condition to a bool by comparing non-equal to 0.0.
        if isinstance(end_cond.type, (ir.FloatType, ir.DoubleType)):
            cmp_instruction = self._llvm.ir_builder.fcmp_ordered
            zero_val = ir.Constant(end_cond.type, 0.0)
        else:
            cmp_instruction = self._llvm.ir_builder.icmp_signed
            zero_val = ir.Constant(end_cond.type, 0)

        end_cond = cmp_instruction(
            "!=",
            end_cond,
            zero_val,
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
            self.named_values[expr.variable.name] = old_val
        else:
            self.named_values.pop(expr.variable.name, None)

        # for expr always returns 0.0.
        result = ir.Constant(self._llvm.INT32_TYPE, 0)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.Module) -> None:
        """Translate ASTx Module to LLVM-IR."""
        for node in expr.nodes:
            self.visit(node)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralInt32) -> None:
        """Translate ASTx LiteralInt32 to LLVM-IR."""
        result = ir.Constant(self._llvm.INT32_TYPE, expr.value)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.FunctionCall) -> None:
        """Translate Function FunctionCall."""
        callee_f = self.get_function(expr.callee)

        if not callee_f:
            raise Exception("Unknown function referenced")

        if len(callee_f.args) != len(expr.args):
            raise Exception("codegen: Incorrect # arguments passed.")

        llvm_args = []
        for arg in expr.args:
            self.visit(arg)
            llvm_arg = self.result_stack.pop()
            if not llvm_arg:
                raise Exception("codegen: Invalid callee argument.")
            llvm_args.append(llvm_arg)

        result = self._llvm.ir_builder.call(callee_f, llvm_args, "calltmp")
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.Function) -> None:
        """Translate ASTx Function to LLVM-IR."""
        proto = expr.prototype
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

        self.visit(expr.body)
        self.result_stack.append(fn)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.FunctionPrototype) -> None:
        """Translate ASTx Function Prototype to LLVM-IR."""
        args_type = []
        for arg in expr.args.nodes:
            type_str = arg.type_.__class__.__name__.lower()
            args_type.append(self._llvm.get_data_type(type_str))
        # note: it should be dynamic
        return_type = self._llvm.get_data_type(
            expr.return_type.__class__.__name__.lower()
        )
        fn_type = ir.FunctionType(return_type, args_type, False)

        fn = ir.Function(self._llvm.module, fn_type, expr.name)

        # Set names for all arguments.
        for idx, llvm_arg in enumerate(fn.args):
            llvm_arg.name = expr.args.nodes[idx].name

        self.result_stack.append(fn)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.FunctionReturn) -> None:
        """Translate ASTx FunctionReturn to LLVM-IR."""
        self.visit(expr.value)

        try:
            retval = self.result_stack.pop()
        except IndexError:
            retval = None

        if retval:
            self._llvm.ir_builder.ret(retval)
            return
        self._llvm.ir_builder.ret_void()

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.InlineVariableDeclaration) -> None:
        """Translate an ASTx InlineVariableDeclaration expression."""
        if self.named_values.get(expr.name):
            raise Exception(f"Variable already declared: {expr.name}")

        type_str = expr.type_.__class__.__name__.lower()

        # Emit the initializer
        if expr.value is not None:
            self.visit(expr.value)
            init_val = self.result_stack.pop()
            if init_val is None:
                raise Exception("Initializer code generation failed.")
        else:
            init_val = ir.Constant(self._llvm.get_data_type(type_str), 0)

        alloca = self.create_entry_block_alloca(expr.name, type_str)
        self._llvm.ir_builder.store(init_val, alloca)
        self.named_values[expr.name] = alloca

        self.result_stack.append(init_val)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.Variable) -> None:
        """Translate ASTx Variable to LLVM-IR."""
        expr_var = self.named_values.get(expr.name)

        if not expr_var:
            raise Exception(f"Unknown variable name: {expr.name}")

        result = self._llvm.ir_builder.load(expr_var, expr.name)
        self.result_stack.append(result)

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.VariableDeclaration) -> None:
        """Translate ASTx Variable to LLVM-IR."""
        if self.named_values.get(expr.name):
            raise Exception(f"Variable already declared: {expr.name}")

        type_str = expr.type_.__class__.__name__.lower()

        # Emit the initializer
        if expr.value is not None:
            self.visit(expr.value)
            init_val = self.result_stack.pop()
            if init_val is None:
                raise Exception("Initializer code generation failed.")
        else:
            # If not specified, use 0 as the initializer.
            # note: it should create something according to the defined type
            init_val = ir.Constant(self._llvm.get_data_type(type_str), 0)

        # Create an alloca in the entry block.
        # note: it should create the type according to the defined type
        alloca = self.create_entry_block_alloca(expr.name, type_str)

        # Store the initial value.
        self._llvm.ir_builder.store(init_val, alloca)

        # Remember this binding.
        self.named_values[expr.name] = alloca

    @dispatch  # type: ignore[no-redef]
    def visit(self, expr: astx.LiteralInt16) -> None:
        """Translate ASTx LiteralInt16 to LLVM-IR."""
        result = ir.Constant(self._llvm.INT16_TYPE, expr.value)
        self.result_stack.append(result)


@public
class LLVMLiteIR(Builder):
    """LLVM-IR transpiler and compiler."""

    def __init__(self) -> None:
        """Initialize LLVMIR."""
        super().__init__()
        self.translator: LLVMLiteIRVisitor = LLVMLiteIRVisitor()

    def build(self, expr: astx.AST, output_file: str) -> None:
        """Transpile the ASTx to LLVM-IR and build it to an executable file."""
        result = self.translate(expr)

        result_mod = llvm.parse_assembly(result)
        result_object = self.translator.target_machine.emit_object(result_mod)

        with tempfile.NamedTemporaryFile(suffix="", delete=False) as temp_file:
            self.tmp_path = temp_file.name

        file_path_o = f"{self.tmp_path}.o"

        with open(file_path_o, "wb") as f:
            f.write(result_object)

        self.output_file = output_file

        run_command(
            [
                "clang",
                file_path_o,
                "-o",
                self.output_file,
            ]
        )

    def run(self) -> None:
        """Run the generated executable."""
        sh([self.output_file])

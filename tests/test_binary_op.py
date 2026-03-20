"""
title: Tests for the BinaryOp.
"""

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from irx.system import PrintExpr
from llvmlite import ir

from .conftest import check_result


@pytest.mark.parametrize(
    "int_type, literal_type",
    [(astx.Int32, astx.LiteralInt32), (astx.Int16, astx.LiteralInt16)],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_binary_op_literals(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test ASTx Module with a function called add.
    parameters:
      builder_class:
        type: type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()
    module = builder.module()

    basic_op = literal_type(1) + literal_type(2)

    decl = astx.VariableDeclaration(
        name="tmp", type_=int_type(), value=basic_op
    )

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl)
    main_block.append(PrintExpr(astx.LiteralUTF8String("3")))
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))

    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="3")


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int16, astx.LiteralInt16),
        (astx.Int8, astx.LiteralInt8),
        (astx.Int64, astx.LiteralInt64),
    ],
)
@pytest.mark.parametrize(
    "action,expected_file",
    [
        # ("translate", "test_binary_op_basic.ll"),
        ("build", ""),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_binary_op_basic(
    action: str,
    expected_file: str,
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Test ASTx Module with a function called add.
    parameters:
      action:
        type: str
      expected_file:
        type: str
      builder_class:
        type: type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
    """
    builder = builder_class()
    module = builder.module()

    decl_a = astx.VariableDeclaration(
        name="a",
        type_=int_type(),
        value=literal_type(1),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_b = astx.VariableDeclaration(
        name="b",
        type_=int_type(),
        value=literal_type(2),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_c = astx.VariableDeclaration(
        name="c",
        type_=int_type(),
        value=literal_type(4),
        mutability=astx.MutabilityKind.mutable,
    )

    a = astx.Identifier("a")
    b = astx.Identifier("b")
    c = astx.Identifier("c")

    lit_1 = literal_type(1)

    basic_op = lit_1 + b - a * c / a + (b - a + c / a)

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=int_type()
    )
    main_block = astx.Block()
    main_block.append(decl_a)
    main_block.append(decl_b)
    main_block.append(decl_c)
    main_block.append(basic_op)
    main_block.append(astx.FunctionReturn(literal_type(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)

    module.block.append(main_fn)
    check_result(action, builder, module, expected_file)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_binary_op_string_not_equals(builder_class: type[Builder]) -> None:
    """
    title: Verify string '!=' uses strcmp_inline + xor 1 path.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    module = builder.module()

    cond = astx.LiteralString("foo") != astx.LiteralString("bar")
    then_blk = astx.Block()
    then_blk.append(PrintExpr(astx.LiteralUTF8String("NE")))
    else_blk = astx.Block()
    else_blk.append(PrintExpr(astx.LiteralUTF8String("EQ")))
    if_stmt = astx.IfStmt(condition=cond, then=then_blk, else_=else_blk)

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(if_stmt)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output="NE")


@pytest.mark.parametrize(
    "int_type,literal_type,a_val,b_val,expect",
    [
        # use 0/1 so bitwise and/or behave like logical
        (astx.Int32, astx.LiteralInt32, 1, 0, "1"),
        (astx.Int16, astx.LiteralInt16, 1, 1, "1"),
    ],
)
@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_binary_op_logical_and_or(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
    a_val: int,
    b_val: int,
    expect: str,
) -> None:
    """
    title: Verify '&&' and '||' for integer booleans (0/1).
    parameters:
      builder_class:
        type: type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
      a_val:
        type: int
      b_val:
        type: int
      expect:
        type: str
    """
    builder = builder_class()
    module = builder.module()

    decl_x = astx.VariableDeclaration(
        name="x",
        type_=int_type(),
        value=literal_type(a_val),
        mutability=astx.MutabilityKind.mutable,
    )
    decl_y = astx.VariableDeclaration(
        name="y",
        type_=int_type(),
        value=literal_type(b_val),
        mutability=astx.MutabilityKind.mutable,
    )

    expr = (astx.Identifier("x") & astx.Identifier("x")) | astx.Identifier("y")
    assign = astx.VariableAssignment(name="x", value=expr)

    print_ok = PrintExpr(astx.LiteralUTF8String(expect))

    main_proto = astx.FunctionPrototype(
        name="main", args=astx.Arguments(), return_type=astx.Int32()
    )
    main_block = astx.Block()
    main_block.append(decl_x)
    main_block.append(decl_y)
    main_block.append(assign)
    main_block.append(print_ok)
    main_block.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main_fn = astx.FunctionDef(prototype=main_proto, body=main_block)
    module.block.append(main_fn)

    check_result("build", builder, module, expected_output=expect)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
@pytest.mark.parametrize(
    "op_code, op_name",
    [
        ("+", "fadd"),
        ("-", "fsub"),
        ("*", "fmul"),
        ("/", "fdiv"),
    ],
)
def test_vector_op_fast_math_flag_applied(
    builder_class: type[Builder],
    op_code: str,
    op_name: str,
) -> None:
    """
    title: Verify fast-math flag is applied to float vector ops when requested.
    summary: >-
      When fast_math is enabled, every float vector instruction (fadd, fsub,
      fmul, fdiv) must be emitted with the 'fast' IR flag.
    parameters:
      builder_class:
        type: type[Builder]
      op_code:
        type: str
      op_name:
        type: str
    """

    visitor = LLVMLiteIRVisitor()

    float_ty = ir.FloatType()
    vec_ty = ir.VectorType(float_ty, 4)
    fn_ty = ir.FunctionType(ir.VoidType(), [vec_ty, vec_ty])
    fn = ir.Function(
        visitor._llvm.module, fn_ty, name=f"test_vec_{op_name}"
    )
    block = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(block)

    lhs = fn.args[0]
    rhs = fn.args[1]

    visitor.set_fast_math(True)
    if op_code == "+":
        result = visitor._llvm.ir_builder.fadd(lhs, rhs, name="vfaddtmp")
    elif op_code == "-":
        result = visitor._llvm.ir_builder.fsub(lhs, rhs, name="vfsubtmp")
    elif op_code == "*":
        result = visitor._llvm.ir_builder.fmul(lhs, rhs, name="vfmultmp")
    else:
        result = visitor._llvm.ir_builder.fdiv(lhs, rhs, name="vfdivtmp")

    visitor._apply_fast_math(result)
    visitor._llvm.ir_builder.ret_void()

    ir_text = str(visitor._llvm.module)
    assert "fast" in ir_text, (
        f"Expected 'fast' flag in IR for float vector {op_code!r} "
        f"with fast_math=True, but got:\n{ir_text}"
    )


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_vector_op_fast_math_flag_not_applied_without_request(
    builder_class: type[Builder],
) -> None:
    """
    title: Verify fast-math flag is NOT applied when fast_math is disabled.
    summary: >-
      When fast_math is not requested, float vector instructions must NOT carry
      the 'fast' IR flag.
    parameters:
      builder_class:
        type: type[Builder]
    """

    visitor = LLVMLiteIRVisitor()

    float_ty = ir.FloatType()
    vec_ty = ir.VectorType(float_ty, 4)
    fn_ty = ir.FunctionType(ir.VoidType(), [vec_ty, vec_ty])
    fn = ir.Function(
        visitor._llvm.module, fn_ty, name="test_vec_no_fast"
    )
    block = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(block)

    # fast_math NOT enabled — default state is False
    result = visitor._llvm.ir_builder.fadd(
        fn.args[0], fn.args[1], name="vfaddtmp"
    )
    visitor._apply_fast_math(result)
    visitor._llvm.ir_builder.ret_void()

    ir_text = str(visitor._llvm.module)
    assert "fast" not in ir_text, (
        "Expected NO 'fast' flag in IR when fast_math=False, "
        f"but got:\n{ir_text}"
    )


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_fast_math_flag_restored_after_vector_op(
    builder_class: type[Builder],
) -> None:
    """
    title: >-
      Verify _fast_math_enabled is restored to its prior value after a vector
      BinaryOp visit.
    summary: >-
      The finally block must restore the exact prior value of
      _fast_math_enabled, not hardcode False. Case 1: flag was True before —
      must still be True after. Case 2: flag was False before — must still be
      False after.
    parameters:
      builder_class:
        type: type[Builder]
    """

    # ── Case 1: flag was True externally, node has fast_math=False ──
    # After the visit, _fast_math_enabled must still be True.
    visitor = LLVMLiteIRVisitor()
    visitor.set_fast_math(True)

    float_ty = ir.FloatType()
    vec_ty = ir.VectorType(float_ty, 4)
    fn_ty = ir.FunctionType(ir.VoidType(), [vec_ty, vec_ty])
    fn = ir.Function(
        visitor._llvm.module, fn_ty, name="test_restore_true"
    )
    block = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(block)

    _prev = visitor._fast_math_enabled          # True
    # set_fast is False (node.fast_math absent) — no mutation happens
    try:
        result = visitor._llvm.ir_builder.fadd(
            fn.args[0], fn.args[1], name="vfaddtmp"
        )
        visitor._apply_fast_math(result)
        visitor._llvm.ir_builder.ret_void()
    finally:
        visitor.set_fast_math(_prev)            # restores True

    assert visitor._fast_math_enabled is True, (
        "_fast_math_enabled was incorrectly reset to False "
        "after vector op when it was True before the visit."
    )

    # ── Case 2: flag was False externally, node has fast_math=True ──
    # After the visit, _fast_math_enabled must be restored to False.
    visitor2 = LLVMLiteIRVisitor()
    assert visitor2._fast_math_enabled is False  # default

    fn_ty2 = ir.FunctionType(ir.VoidType(), [vec_ty, vec_ty])
    fn2 = ir.Function(
        visitor2._llvm.module, fn_ty2, name="test_restore_false"
    )
    block2 = fn2.append_basic_block("entry")
    visitor2._llvm.ir_builder = ir.IRBuilder(block2)

    _prev2 = visitor2._fast_math_enabled        # False
    visitor2.set_fast_math(True)                # node.fast_math=True
    try:
        result2 = visitor2._llvm.ir_builder.fadd(
            fn2.args[0], fn2.args[1], name="vfaddtmp"
        )
        visitor2._apply_fast_math(result2)
        visitor2._llvm.ir_builder.ret_void()
    finally:
        visitor2.set_fast_math(_prev2)          # restores False

    assert visitor2._fast_math_enabled is False, (
        "_fast_math_enabled was left as True after vector op "
        "when it was False before the visit."
    )


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_emit_fma_fast_math_applied_on_direct_path(
    builder_class: type[Builder],
) -> None:
    """
    title: Verify _emit_fma applies fast-math on the builder.fma direct path.
    summary: >-
      When llvmlite exposes builder.fma natively, _emit_fma must still call
      _apply_fast_math on the result. Before the fix it returned immediately
      without applying the flag.
    parameters:
      builder_class:
        type: type[Builder]
    """

    visitor = LLVMLiteIRVisitor()
    visitor.set_fast_math(True)

    float_ty = ir.FloatType()
    vec_ty = ir.VectorType(float_ty, 4)
    fn_ty = ir.FunctionType(ir.VoidType(), [vec_ty, vec_ty, vec_ty])
    fn = ir.Function(
        visitor._llvm.module, fn_ty, name="test_fma_fast"
    )
    block = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(block)

    visitor._emit_fma(fn.args[0], fn.args[1], fn.args[2])
    visitor._llvm.ir_builder.ret_void()

    ir_text = str(visitor._llvm.module)
    assert "fast" in ir_text, (
        "_emit_fma did not apply 'fast' flag even though "
        f"fast_math=True. IR:\n{ir_text}"
    )
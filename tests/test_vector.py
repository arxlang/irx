"""
title: Tests for vector operations in the LLVM-IR builder.
"""

from typing import Any

import astx
import pytest

from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

VEC4 = 4
VEC2 = 2


def setup_builder() -> LLVMLiteIRVisitor:
    """
    title: Return a visitor with a live IRBuilder positioned inside main().
    returns:
      type: LLVMLiteIRVisitor
    """
    main_builder = LLVMLiteIR()
    visitor = main_builder.translator
    func_type = ir.FunctionType(visitor._llvm.INT32_TYPE, [])
    fn = ir.Function(visitor._llvm.module, func_type, name="main")
    bb = fn.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(bb)
    return visitor


def _make_binop_visitor(
    lhs_val: ir.Value,
    rhs_val: ir.Value,
    fma_rhs: ir.Value | None = None,
) -> LLVMLiteIRVisitor:
    """
    title: >-
      Return a fresh visitor patched to inject pre-built IR values for the LHS,
      RHS, and FMA_RHS identifiers.
    parameters:
      lhs_val:
        type: ir.Value
      rhs_val:
        type: ir.Value
      fma_rhs:
        type: ir.Value | None
    returns:
      type: LLVMLiteIRVisitor
    """
    builder = setup_builder()
    original_visit = builder.visit

    def mock_visit(node: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(node, astx.Identifier):
            mapping = {"LHS": lhs_val, "RHS": rhs_val, "FMA_RHS": fma_rhs}
            if node.name in mapping:
                builder.result_stack.append(mapping[node.name])
                return
        return original_visit(node, *args, **kwargs)

    builder.visit = mock_visit  # type: ignore[method-assign]
    return builder


def _run_vector_binop(
    op_code: str,
    lhs_val: ir.Value,
    rhs_val: ir.Value,
    unsigned: bool | None = None,
    fma_rhs: ir.Value | None = None,
) -> ir.Value:
    """
    title: Drive a BinaryOp through the visitor and return the result.
    parameters:
      op_code:
        type: str
      lhs_val:
        type: ir.Value
      rhs_val:
        type: ir.Value
      unsigned:
        type: bool | None
      fma_rhs:
        type: ir.Value | None
    returns:
      type: ir.Value
    """
    builder = _make_binop_visitor(lhs_val, rhs_val, fma_rhs)
    bin_op = astx.BinaryOp(
        op_code, astx.Identifier("LHS"), astx.Identifier("RHS")
    )
    if unsigned is not None:
        bin_op.unsigned = unsigned  # type: ignore[attr-defined]
    if fma_rhs is not None:
        bin_op.fma = True  # type: ignore[attr-defined]
        bin_op.fma_rhs = astx.Identifier("FMA_RHS")  # type: ignore[attr-defined]
    builder.visit(bin_op)
    return builder.result_stack.pop()


_ARITH_CASES = [
    # float32 x4
    ("FLOAT_TYPE", VEC4, [4.0] * VEC4, [2.0] * VEC4, "+", "fadd", True),
    ("FLOAT_TYPE", VEC4, [4.0] * VEC4, [2.0] * VEC4, "-", "fsub", True),
    ("FLOAT_TYPE", VEC4, [4.0] * VEC4, [2.0] * VEC4, "*", "fmul", True),
    ("FLOAT_TYPE", VEC4, [4.0] * VEC4, [2.0] * VEC4, "/", "fdiv", True),
    # float64 x2
    ("DOUBLE_TYPE", VEC2, [1.0, 2.0], [3.0, 4.0], "+", "fadd", True),
    ("DOUBLE_TYPE", VEC2, [1.0, 2.0], [3.0, 4.0], "-", "fsub", True),
    ("DOUBLE_TYPE", VEC2, [1.0, 2.0], [3.0, 4.0], "*", "fmul", True),
    ("DOUBLE_TYPE", VEC2, [1.0, 2.0], [3.0, 4.0], "/", "fdiv", True),
    # int32 x4
    ("INT32_TYPE", VEC4, [10] * VEC4, [3] * VEC4, "+", "add", False),
    ("INT32_TYPE", VEC4, [10] * VEC4, [3] * VEC4, "-", "sub", False),
    ("INT32_TYPE", VEC4, [10] * VEC4, [3] * VEC4, "*", "mul", False),
]


def _arith_id(case: tuple[Any, ...]) -> str:
    elem, count, _, _, op, *_ = case
    return f"{elem.replace('_TYPE', '').lower()}x{count}_{op}"


@pytest.mark.parametrize(
    "elem_attr, count, lhs_vals, rhs_vals, op, mnemonic, is_fp",
    _ARITH_CASES,
    ids=[_arith_id(c) for c in _ARITH_CASES],
)
def test_vector_arithmetic(
    elem_attr: str,
    count: int,
    lhs_vals: list[Any],
    rhs_vals: list[Any],
    op: str,
    mnemonic: str,
    is_fp: bool,
) -> None:
    """
    title: >-
      Vector arithmetic emits the correct instruction and preserves element
      type and lane count for all supported numeric types.
    parameters:
      elem_attr:
        type: str
      count:
        type: int
      lhs_vals:
        type: list[Any]
      rhs_vals:
        type: list[Any]
      op:
        type: str
      mnemonic:
        type: str
      is_fp:
        type: bool
    """
    builder = setup_builder()
    elem_ty = getattr(builder._llvm, elem_attr)
    vec_ty = ir.VectorType(elem_ty, count)
    result = _run_vector_binop(
        op, ir.Constant(vec_ty, lhs_vals), ir.Constant(vec_ty, rhs_vals)
    )

    ir_str = str(result)
    assert mnemonic in ir_str, f"Expected '{mnemonic}' in IR, got: {ir_str}"
    if not is_fp:
        assert "f" + mnemonic not in ir_str, (
            f"Integer op must not emit FP variant 'f{mnemonic}'"
        )
    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == count
    assert result.type.element == elem_ty


@pytest.mark.parametrize(
    "unsigned, want, reject",
    [(False, "sdiv", "udiv"), (True, "udiv", "sdiv")],
    ids=["signed", "unsigned"],
)
def test_int_vector_division(unsigned: bool, want: str, reject: str) -> None:
    """
    title: >-
      Integer vector division emits sdiv for signed and udiv for unsigned, and
      never emits the opposite variant.
    parameters:
      unsigned:
        type: bool
      want:
        type: str
      reject:
        type: str
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
    result = _run_vector_binop(
        "/",
        ir.Constant(vec_ty, [10] * VEC4),
        ir.Constant(vec_ty, [2] * VEC4),
        unsigned=unsigned,
    )
    ir_str = str(result)
    assert want in ir_str
    assert reject not in ir_str
    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == VEC4


# (elem_attr, count, vec_vals, scalar_val, lhs_is_vec, mnemonic)
_SPLAT_CASES = [
    ("FLOAT_TYPE", VEC4, [1.0] * VEC4, 2.0, True, "fadd"),
    ("FLOAT_TYPE", VEC4, [1.0] * VEC4, 2.0, False, "fadd"),
    ("INT32_TYPE", VEC4, [1] * VEC4, 2, True, "add"),
    ("INT32_TYPE", VEC4, [1] * VEC4, 2, False, "add"),
]


def _splat_id(case: tuple[Any, ...]) -> str:
    elem, count, _, _, lhs_is_vec, *_ = case
    order = "vec+scalar" if lhs_is_vec else "scalar+vec"
    return f"{elem.replace('_TYPE', '').lower()}x{count}_{order}"


@pytest.mark.parametrize(
    "elem_attr, count, vec_vals, scalar_val, lhs_is_vec, mnemonic",
    _SPLAT_CASES,
    ids=[_splat_id(c) for c in _SPLAT_CASES],
)
def test_scalar_splatted_to_vector(
    elem_attr: str,
    count: int,
    vec_vals: list[Any],
    scalar_val: float | int,
    lhs_is_vec: bool,
    mnemonic: str,
) -> None:
    """
    title: >-
      A scalar operand is broadcast to all lanes of the partner vector
      regardless of operand order.
    parameters:
      elem_attr:
        type: str
      count:
        type: int
      vec_vals:
        type: list[Any]
      scalar_val:
        type: float | int
      lhs_is_vec:
        type: bool
      mnemonic:
        type: str
    """
    builder = setup_builder()
    elem_ty = getattr(builder._llvm, elem_attr)
    vec = ir.Constant(ir.VectorType(elem_ty, count), vec_vals)
    scalar = ir.Constant(elem_ty, scalar_val)
    lhs, rhs = (vec, scalar) if lhs_is_vec else (scalar, vec)

    result = _run_vector_binop("+", lhs, rhs)

    assert mnemonic in str(result)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == count
    assert result.type.element == elem_ty


_CROSS_FP_CASES = [
    (
        "FLOAT_TYPE",
        "DOUBLE_TYPE",
        True,
        "FLOAT_TYPE",
    ),  # float vec + double scalar -> float
    (
        "FLOAT_TYPE",
        "DOUBLE_TYPE",
        False,
        "FLOAT_TYPE",
    ),  # double scalar + float vec -> float
    (
        "DOUBLE_TYPE",
        "FLOAT_TYPE",
        True,
        "DOUBLE_TYPE",
    ),  # double vec + float scalar -> double
    (
        "DOUBLE_TYPE",
        "FLOAT_TYPE",
        False,
        "DOUBLE_TYPE",
    ),  # float scalar + double vec -> double
]


def _cross_id(case: tuple[Any, ...]) -> str:
    vec_e, sc_e, lhs_is_vec, *_ = case
    v = vec_e.replace("_TYPE", "").lower()
    s = sc_e.replace("_TYPE", "").lower()
    return f"{v}_vec_{s}_scalar_{'lhs' if lhs_is_vec else 'rhs'}"


@pytest.mark.parametrize(
    "vec_elem_attr, scalar_attr, lhs_is_vec, expected_elem_attr",
    _CROSS_FP_CASES,
    ids=[_cross_id(c) for c in _CROSS_FP_CASES],
)
def test_fp_scalar_vector_cross_precision(
    vec_elem_attr: str,
    scalar_attr: str,
    lhs_is_vec: bool,
    expected_elem_attr: str,
) -> None:
    """
    title: >-
      Cross-precision FP scalar+vector ops produce the vector's element type
      (scalar is cast to match).
    parameters:
      vec_elem_attr:
        type: str
      scalar_attr:
        type: str
      lhs_is_vec:
        type: bool
      expected_elem_attr:
        type: str
    """
    builder = setup_builder()
    vec_elem_ty = getattr(builder._llvm, vec_elem_attr)
    scalar_ty = getattr(builder._llvm, scalar_attr)
    expected_elem_ty = getattr(builder._llvm, expected_elem_attr)

    vec = ir.Constant(ir.VectorType(vec_elem_ty, VEC4), [1.0] * VEC4)
    scalar = ir.Constant(scalar_ty, 2.0)
    lhs, rhs = (vec, scalar) if lhs_is_vec else (scalar, vec)

    result = _run_vector_binop("+", lhs, rhs)

    assert "fadd" in str(result)
    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == VEC4
    assert result.type.element == expected_elem_ty


@pytest.mark.parametrize(
    "elem_attr, count",
    [("FLOAT_TYPE", VEC4), ("DOUBLE_TYPE", VEC2)],
    ids=["float32x4", "float64x2"],
)
def test_vector_fma_direct(elem_attr: str, count: int) -> None:
    """
    title: >-
      _emit_fma produces a vector result of the correct type and registers the
      llvm.fma intrinsic in the module.
    parameters:
      elem_attr:
        type: str
      count:
        type: int
    """
    builder = setup_builder()
    elem_ty = getattr(builder._llvm, elem_attr)
    vec_ty = ir.VectorType(elem_ty, count)
    v = ir.Constant(vec_ty, [2.0] * count)

    result = builder._emit_fma(v, v, v)

    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == count
    assert result.type.element == elem_ty
    assert "llvm.fma" in str(builder._llvm.module)


def test_fma_via_binop_visitor() -> None:
    """
    title: >-
      FMA driven through the BinaryOp visitor produces a vector result and
      emits the llvm.fma intrinsic.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v = ir.Constant(vec_ty, [2.0] * VEC4)

    result = _run_vector_binop("*", v, v, fma_rhs=v)

    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == VEC4
    assert result.type.element == builder._llvm.FLOAT_TYPE
    assert "llvm.fma" in str(result.operands[0]) or "llvm.fma" in str(result)


def test_fma_missing_fma_rhs_raises() -> None:
    """
    title: FMA without fma_rhs operand raises an exception.
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v = ir.Constant(vec_ty, [1.0] * VEC4)
    patched = _make_binop_visitor(v, v)

    bin_op = astx.BinaryOp("*", astx.Identifier("LHS"), astx.Identifier("RHS"))
    bin_op.fma = True  # type: ignore[attr-defined]

    with pytest.raises(Exception, match="FMA requires a third operand"):
        patched.visit(bin_op)


@pytest.mark.parametrize(
    "op, raises",
    [("+", False), ("%", True)],
    ids=["success", "failure"],
)
def test_fast_math_flag_always_cleared(op: str, raises: bool) -> None:
    """
    title: >-
      _fast_math_enabled is reset to False after the op regardless of whether
      it succeeds or raises.
    parameters:
      op:
        type: str
      raises:
        type: bool
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v = ir.Constant(vec_ty, [1.0] * VEC4)
    patched = _make_binop_visitor(v, v)

    bin_op = astx.BinaryOp(op, astx.Identifier("LHS"), astx.Identifier("RHS"))
    bin_op.fast_math = True  # type: ignore[attr-defined]

    if raises:
        with pytest.raises(Exception):
            patched.visit(bin_op)
    else:
        patched.visit(bin_op)
        result = patched.result_stack.pop()
        assert "fadd" in str(result)
        assert isinstance(result.type, ir.VectorType)

    assert patched._fast_math_enabled is False


@pytest.mark.parametrize(
    "lhs_count, rhs_count, match",
    [(VEC4, VEC2, "Vector size mismatch")],
    ids=["size_mismatch"],
)
def test_vector_size_mismatch_raises(
    lhs_count: int, rhs_count: int, match: str
) -> None:
    """
    title: >-
      Mismatched vector sizes raise an exception with a descriptive message.
    parameters:
      lhs_count:
        type: int
      rhs_count:
        type: int
      match:
        type: str
    """
    builder = setup_builder()
    elem_ty = builder._llvm.INT32_TYPE
    v1 = ir.Constant(ir.VectorType(elem_ty, lhs_count), [1] * lhs_count)
    v2 = ir.Constant(ir.VectorType(elem_ty, rhs_count), [1] * rhs_count)
    with pytest.raises(Exception, match=match):
        _run_vector_binop("+", v1, v2)


def test_vector_element_type_mismatch_raises() -> None:
    """
    title: Mismatched vector element types raise an exception.
    """
    builder = setup_builder()
    v1 = ir.Constant(ir.VectorType(builder._llvm.INT32_TYPE, VEC2), [1] * VEC2)
    v2 = ir.Constant(ir.VectorType(builder._llvm.INT64_TYPE, VEC2), [1] * VEC2)
    with pytest.raises(Exception, match="Vector element type mismatch"):
        _run_vector_binop("+", v1, v2)


@pytest.mark.parametrize(
    "op, match",
    [
        ("%", r"Vector binop .* not implemented"),
        ("==", r"Vector binop .* not implemented"),
        ("!=", r"Vector binop .* not implemented"),
        ("<", r"Vector binop .* not implemented"),
        ("<=", r"Vector binop .* not implemented"),
        (">", r"Vector binop .* not implemented"),
        (">=", r"Vector binop .* not implemented"),
    ],
    ids=[
        "unsupported_%",
        "cmp_eq",
        "cmp_ne",
        "cmp_lt",
        "cmp_le",
        "cmp_gt",
        "cmp_ge",
    ],
)
def test_unsupported_vector_op_raises(op: str, match: str) -> None:
    """
    title: >-
      Unsupported and unimplemented comparison operators all raise an
      exception.
    parameters:
      op:
        type: str
      match:
        type: str
    """
    builder = setup_builder()
    vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
    v = ir.Constant(vec_ty, [1.0] * VEC4)
    with pytest.raises(Exception, match=match):
        _run_vector_binop(op, v, v)


@pytest.mark.parametrize(
    "elem_attr, count, vals, other_val, op, mnemonic, is_fp",
    [
        (
            "FLOAT_TYPE",
            VEC4,
            [0.0] * VEC4,
            1.0,
            "+",
            "fadd",
            True,
        ),  # zero float vec
        (
            "INT32_TYPE",
            VEC4,
            [-1] * VEC4,
            1,
            "+",
            "add",
            False,
        ),  # negative int vec
    ],
    ids=["zero_float_vec", "negative_int_vec"],
)
def test_vector_edge_values(
    elem_attr: str,
    count: int,
    vals: list[Any],
    other_val: float | int,
    op: str,
    mnemonic: str,
    is_fp: bool,
) -> None:
    """
    title: >-
      Zero and negative-valued vectors are lowered without error and produce
      the expected instruction and type.
    parameters:
      elem_attr:
        type: str
      count:
        type: int
      vals:
        type: list[Any]
      other_val:
        type: float | int
      op:
        type: str
      mnemonic:
        type: str
      is_fp:
        type: bool
    """
    builder = setup_builder()
    elem_ty = getattr(builder._llvm, elem_attr)
    vec_ty = ir.VectorType(elem_ty, count)
    v = ir.Constant(vec_ty, vals)
    other = ir.Constant(vec_ty, [other_val] * count)

    result = _run_vector_binop(op, v, other)

    ir_str = str(result)
    assert mnemonic in ir_str
    if not is_fp:
        assert "f" + mnemonic not in ir_str
    assert isinstance(result.type, ir.VectorType)
    assert result.type.count == count
    assert result.type.element == elem_ty

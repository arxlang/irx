"""
title: Tests for heterogeneous vector-vector numeric unification (#270).
"""

from typing import Any

import astx
import pytest

from irx.builder import Builder, Visitor
from llvmlite import ir

VEC4 = 4
VEC2 = 2


def setup_builder() -> Visitor:
    """
    title: Return a visitor with a live IRBuilder positioned inside main().
    returns:
      type: Visitor
    """
    main_builder = Builder()
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
) -> Visitor:
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
      type: Visitor
    """
    builder = setup_builder()
    original_visit = builder.visit

    def mock_visit(node: Any, *args: Any, **kwargs: Any) -> Any:
        """
        title: Mock visit.
        parameters:
          node:
            type: Any
          args:
            type: Any
            variadic: positional
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: Any
        """
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


# ---------------------------------------------------------------------------
# Gap 1: Vector-vector element-type promotion
# ---------------------------------------------------------------------------


class TestVectorVectorPromotion:
    """
    title: Tests for heterogeneous vector-vector element-type promotion.
    """

    def test_int16_plus_int32_vector(self) -> None:
        """
        title: vector<i16> + vector<i32> promotes both to vector<i32>.
        """
        builder = setup_builder()
        v1 = ir.Constant(
            ir.VectorType(builder._llvm.INT16_TYPE, VEC4), [1] * VEC4
        )
        v2 = ir.Constant(
            ir.VectorType(builder._llvm.INT32_TYPE, VEC4), [2] * VEC4
        )
        result = _run_vector_binop("+", v1, v2)
        assert isinstance(result.type, ir.VectorType)
        assert result.type.element == builder._llvm.INT32_TYPE
        assert result.type.count == VEC4

    def test_float_plus_double_vector(self) -> None:
        """
        title: vector<float> + vector<double> promotes both to vector<double>.
        """
        builder = setup_builder()
        v1 = ir.Constant(
            ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4), [1.0] * VEC4
        )
        v2 = ir.Constant(
            ir.VectorType(builder._llvm.DOUBLE_TYPE, VEC4), [2.0] * VEC4
        )
        result = _run_vector_binop("+", v1, v2)
        assert isinstance(result.type, ir.VectorType)
        assert result.type.element == builder._llvm.DOUBLE_TYPE
        assert result.type.count == VEC4

    def test_unsigned_int_vector_widening_uses_zext(self) -> None:
        """
        title: >-
          Unsigned vector<i16> + vector<i32> uses zext (not sext) for widening.
        """
        builder = setup_builder()
        v1 = ir.Constant(
            ir.VectorType(builder._llvm.INT16_TYPE, VEC2), [1] * VEC2
        )
        v2 = ir.Constant(
            ir.VectorType(builder._llvm.INT32_TYPE, VEC2), [2] * VEC2
        )
        lhs, _rhs = builder._unify_numeric_operands(v1, v2, unsigned=True)
        assert lhs.type.element == builder._llvm.INT32_TYPE
        assert getattr(lhs, "opname", "") == "zext"

    def test_signed_int_vector_widening_uses_sext(self) -> None:
        """
        title: Signed vector<i16> + vector<i32> uses sext for widening.
        """
        builder = setup_builder()
        v1 = ir.Constant(
            ir.VectorType(builder._llvm.INT16_TYPE, VEC2), [1] * VEC2
        )
        v2 = ir.Constant(
            ir.VectorType(builder._llvm.INT32_TYPE, VEC2), [2] * VEC2
        )
        lhs, _rhs = builder._unify_numeric_operands(v1, v2, unsigned=False)
        assert lhs.type.element == builder._llvm.INT32_TYPE
        assert getattr(lhs, "opname", "") == "sext"

    def test_unsigned_int_to_float_vector_uses_uitofp(self) -> None:
        """
        title: >-
          Unsigned vector<i32> + vector<float> converts int to float via
          uitofp.
        """
        builder = setup_builder()
        v1 = ir.Constant(
            ir.VectorType(builder._llvm.INT32_TYPE, VEC4), [1] * VEC4
        )
        v2 = ir.Constant(
            ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4), [2.0] * VEC4
        )
        lhs, _rhs = builder._unify_numeric_operands(v1, v2, unsigned=True)
        assert lhs.type.element == builder._llvm.FLOAT_TYPE
        assert getattr(lhs, "opname", "") == "uitofp"

    def test_signed_int_to_float_vector_uses_sitofp(self) -> None:
        """
        title: >-
          Signed vector<i32> + vector<float> converts int to float via sitofp.
        """
        builder = setup_builder()
        v1 = ir.Constant(
            ir.VectorType(builder._llvm.INT32_TYPE, VEC4), [1] * VEC4
        )
        v2 = ir.Constant(
            ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4), [2.0] * VEC4
        )
        lhs, _rhs = builder._unify_numeric_operands(v1, v2, unsigned=False)
        assert lhs.type.element == builder._llvm.FLOAT_TYPE
        assert getattr(lhs, "opname", "") == "sitofp"

    def test_lane_count_mismatch_still_raises(self) -> None:
        """
        title: >-
          Vectors with different lane counts still raise even when element
          types differ.
        """
        builder = setup_builder()
        v1 = ir.Constant(
            ir.VectorType(builder._llvm.INT16_TYPE, VEC4), [1] * VEC4
        )
        v2 = ir.Constant(
            ir.VectorType(builder._llvm.INT32_TYPE, VEC2), [2] * VEC2
        )
        with pytest.raises(Exception, match="Vector size mismatch"):
            builder._unify_numeric_operands(v1, v2)

    def test_same_element_type_returns_unchanged(self) -> None:
        """
        title: Vectors with identical element types are returned unchanged.
        """
        builder = setup_builder()
        v1 = ir.Constant(
            ir.VectorType(builder._llvm.INT32_TYPE, VEC4), [1] * VEC4
        )
        v2 = ir.Constant(
            ir.VectorType(builder._llvm.INT32_TYPE, VEC4), [2] * VEC4
        )
        lhs, rhs = builder._unify_numeric_operands(v1, v2)
        assert lhs is v1
        assert rhs is v2


# ---------------------------------------------------------------------------
# Gap 2: Vector compare operations
# ---------------------------------------------------------------------------


_VEC_CMP_OPS = ["<", ">", "<=", ">=", "==", "!="]


class TestVectorCompare:
    """
    title: Tests for vector compare operations via the BinaryOp visitor.
    """

    @pytest.mark.parametrize("op", _VEC_CMP_OPS, ids=_VEC_CMP_OPS)
    def test_float_vector_compare(self, op: str) -> None:
        """
        title: Float vector compares emit fcmp_ordered.
        parameters:
          op:
            type: str
        """
        builder = setup_builder()
        vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
        v1 = ir.Constant(vec_ty, [1.0] * VEC4)
        v2 = ir.Constant(vec_ty, [2.0] * VEC4)
        result = _run_vector_binop(op, v1, v2)
        assert isinstance(result.type, ir.VectorType)
        assert result.type.count == VEC4
        assert "fcmp" in str(result)

    @pytest.mark.parametrize("op", _VEC_CMP_OPS, ids=_VEC_CMP_OPS)
    def test_int_vector_compare_signed(self, op: str) -> None:
        """
        title: Signed int vector compares emit icmp.
        parameters:
          op:
            type: str
        """
        builder = setup_builder()
        vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
        v1 = ir.Constant(vec_ty, [1] * VEC4)
        v2 = ir.Constant(vec_ty, [2] * VEC4)
        result = _run_vector_binop(op, v1, v2)
        assert isinstance(result.type, ir.VectorType)
        assert result.type.count == VEC4
        assert "icmp" in str(result)

    @pytest.mark.parametrize("op", _VEC_CMP_OPS, ids=_VEC_CMP_OPS)
    def test_int_vector_compare_unsigned(self, op: str) -> None:
        """
        title: Unsigned int vector compares emit icmp with unsigned predicates.
        parameters:
          op:
            type: str
        """
        builder = setup_builder()
        vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
        v1 = ir.Constant(vec_ty, [1] * VEC4)
        v2 = ir.Constant(vec_ty, [2] * VEC4)
        result = _run_vector_binop(op, v1, v2, unsigned=True)
        assert isinstance(result.type, ir.VectorType)
        assert result.type.count == VEC4
        assert "icmp" in str(result)

    def test_heterogeneous_vector_compare(self) -> None:
        """
        title: >-
          vector<float> < vector<double> promotes to vector<double> then
          compares.
        """
        builder = setup_builder()
        v1 = ir.Constant(
            ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4), [1.0] * VEC4
        )
        v2 = ir.Constant(
            ir.VectorType(builder._llvm.DOUBLE_TYPE, VEC4), [2.0] * VEC4
        )
        result = _run_vector_binop("<", v1, v2)
        assert isinstance(result.type, ir.VectorType)
        assert result.type.count == VEC4
        assert "fcmp" in str(result)

    def test_scalar_vector_compare(self) -> None:
        """
        title: Scalar < vector promotes and splats the scalar, then compares.
        """
        builder = setup_builder()
        vec_ty = ir.VectorType(builder._llvm.INT32_TYPE, VEC4)
        v = ir.Constant(vec_ty, [1] * VEC4)
        s = ir.Constant(builder._llvm.INT32_TYPE, 2)
        result = _run_vector_binop("<", s, v)
        assert isinstance(result.type, ir.VectorType)
        assert result.type.count == VEC4
        assert "icmp" in str(result)


# ---------------------------------------------------------------------------
# Gap 3: FMA unification
# ---------------------------------------------------------------------------


class TestFMAUnification:
    """
    title: Tests for FMA with heterogeneous operand types.
    """

    def test_fma_float_float_double_unifies(self) -> None:
        """
        title: >-
          FMA with vector<float> * vector<float> + vector<double> promotes all
          to vector<double>.
        """
        builder = setup_builder()
        vf = ir.Constant(
            ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4), [1.0] * VEC4
        )
        vd = ir.Constant(
            ir.VectorType(builder._llvm.DOUBLE_TYPE, VEC4), [2.0] * VEC4
        )
        result = _run_vector_binop("*", vf, vf, fma_rhs=vd)
        assert isinstance(result.type, ir.VectorType)
        assert result.type.element == builder._llvm.DOUBLE_TYPE
        assert result.type.count == VEC4

    def test_fma_same_type_still_works(self) -> None:
        """
        title: >-
          FMA with matching types still works after replacing the hard error
          with unification.
        """
        builder = setup_builder()
        vec_ty = ir.VectorType(builder._llvm.FLOAT_TYPE, VEC4)
        v = ir.Constant(vec_ty, [2.0] * VEC4)
        result = _run_vector_binop("*", v, v, fma_rhs=v)
        assert isinstance(result.type, ir.VectorType)
        assert result.type.element == builder._llvm.FLOAT_TYPE
        assert result.type.count == VEC4

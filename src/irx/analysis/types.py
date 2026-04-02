"""
title: Type helpers for semantic analysis.
"""

from __future__ import annotations

from public import public

from irx import astx

INT_TYPES = (astx.Int8, astx.Int16, astx.Int32, astx.Int64)
UINT_TYPES = (astx.UInt8, astx.UInt16, astx.UInt32, astx.UInt64, astx.UInt128)
FLOAT_TYPES = (astx.Float16, astx.Float32, astx.Float64)
STRING_TYPES = (astx.String, astx.UTF8String, astx.UTF8Char)
TEMPORAL_TYPES = (astx.Time, astx.Timestamp, astx.DateTime)
BIT_WIDTH_8 = 8
BIT_WIDTH_16 = 16
BIT_WIDTH_32 = 32
BIT_WIDTH_64 = 64
BIT_WIDTH_128 = 128

_BIT_WIDTHS: dict[type[astx.DataType], int] = {
    astx.Int8: BIT_WIDTH_8,
    astx.UInt8: BIT_WIDTH_8,
    astx.Int16: BIT_WIDTH_16,
    astx.UInt16: BIT_WIDTH_16,
    astx.Int32: BIT_WIDTH_32,
    astx.UInt32: BIT_WIDTH_32,
    astx.Int64: BIT_WIDTH_64,
    astx.UInt64: BIT_WIDTH_64,
    astx.UInt128: BIT_WIDTH_128,
    astx.Float16: BIT_WIDTH_16,
    astx.Float32: BIT_WIDTH_32,
    astx.Float64: BIT_WIDTH_64,
}


@public
def clone_type(type_: astx.DataType) -> astx.DataType:
    """
    title: Clone an AST type by class.
    parameters:
      type_:
        type: astx.DataType
    returns:
      type: astx.DataType
    """
    return type_.__class__()


@public
def same_type(lhs: astx.DataType | None, rhs: astx.DataType | None) -> bool:
    """
    title: Return whether two AST types share the same class.
    parameters:
      lhs:
        type: astx.DataType | None
      rhs:
        type: astx.DataType | None
    returns:
      type: bool
    """
    if lhs is None or rhs is None:
        return False
    return lhs.__class__ is rhs.__class__


@public
def is_integer_type(type_: astx.DataType | None) -> bool:
    """
    title: Is integer type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return isinstance(type_, INT_TYPES + UINT_TYPES)


@public
def is_signed_integer_type(type_: astx.DataType | None) -> bool:
    """
    title: Is signed integer type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return isinstance(type_, INT_TYPES)


@public
def is_unsigned_type(type_: astx.DataType | None) -> bool:
    """
    title: Is unsigned type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return isinstance(type_, UINT_TYPES)


@public
def is_float_type(type_: astx.DataType | None) -> bool:
    """
    title: Is float type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return isinstance(type_, FLOAT_TYPES)


@public
def is_numeric_type(type_: astx.DataType | None) -> bool:
    """
    title: Is numeric type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return is_integer_type(type_) or is_float_type(type_)


@public
def is_boolean_type(type_: astx.DataType | None) -> bool:
    """
    title: Is boolean type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return isinstance(type_, astx.Boolean)


@public
def is_string_type(type_: astx.DataType | None) -> bool:
    """
    title: Is string type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return isinstance(type_, STRING_TYPES)


@public
def is_temporal_type(type_: astx.DataType | None) -> bool:
    """
    title: Is temporal type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return isinstance(type_, TEMPORAL_TYPES)


@public
def is_none_type(type_: astx.DataType | None) -> bool:
    """
    title: Is none type.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: bool
    """
    return isinstance(type_, astx.NoneType)


@public
def bit_width(type_: astx.DataType | None) -> int:
    """
    title: Return the nominal bit width for numeric types.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: int
    """
    if type_ is None:
        return 0
    return _BIT_WIDTHS.get(type(type_), 0)


@public
def common_numeric_type(
    lhs: astx.DataType | None,
    rhs: astx.DataType | None,
) -> astx.DataType | None:
    """
    title: Return a widened numeric type shared by both operands.
    parameters:
      lhs:
        type: astx.DataType | None
      rhs:
        type: astx.DataType | None
    returns:
      type: astx.DataType | None
    """
    if lhs is None or rhs is None:
        return None
    if not is_numeric_type(lhs) or not is_numeric_type(rhs):
        return None

    if is_float_type(lhs) or is_float_type(rhs):
        widest = max(bit_width(lhs), bit_width(rhs))
        if widest <= BIT_WIDTH_16:
            return astx.Float16()
        if widest <= BIT_WIDTH_32:
            return astx.Float32()
        return astx.Float64()

    width = max(bit_width(lhs), bit_width(rhs))
    use_unsigned = is_unsigned_type(lhs) or is_unsigned_type(rhs)
    if use_unsigned:
        if width <= BIT_WIDTH_8:
            return astx.UInt8()
        if width <= BIT_WIDTH_16:
            return astx.UInt16()
        if width <= BIT_WIDTH_32:
            return astx.UInt32()
        if width <= BIT_WIDTH_64:
            return astx.UInt64()
        return astx.UInt128()

    if width <= BIT_WIDTH_8:
        return astx.Int8()
    if width <= BIT_WIDTH_16:
        return astx.Int16()
    if width <= BIT_WIDTH_32:
        return astx.Int32()
    return astx.Int64()


@public
def is_assignable(
    target: astx.DataType | None,
    value: astx.DataType | None,
) -> bool:
    """
    title: Return whether a value type can be assigned to a target type.
    parameters:
      target:
        type: astx.DataType | None
      value:
        type: astx.DataType | None
    returns:
      type: bool
    """
    if target is None or value is None:
        return True
    if same_type(target, value):
        return True
    if is_numeric_type(target) and is_numeric_type(value):
        common = common_numeric_type(target, value)
        return common is not None and same_type(target, common)
    if is_string_type(target) and is_string_type(value):
        return True
    if is_none_type(target) and is_none_type(value):
        return True
    return False

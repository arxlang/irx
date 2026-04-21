"""
title: Type helpers for semantic analysis.
summary: >-
  Provide the small AST-type predicates and promotion helpers that the semantic
  analyzer reuses across many node visitors.
"""

from __future__ import annotations

from public import public

from irx import astx
from irx.typecheck import typechecked

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
_SIGNED_INTEGERS_BY_WIDTH: dict[int, type[astx.DataType]] = {
    BIT_WIDTH_8: astx.Int8,
    BIT_WIDTH_16: astx.Int16,
    BIT_WIDTH_32: astx.Int32,
    BIT_WIDTH_64: astx.Int64,
}
_UNSIGNED_INTEGERS_BY_WIDTH: dict[int, type[astx.DataType]] = {
    BIT_WIDTH_8: astx.UInt8,
    BIT_WIDTH_16: astx.UInt16,
    BIT_WIDTH_32: astx.UInt32,
    BIT_WIDTH_64: astx.UInt64,
    BIT_WIDTH_128: astx.UInt128,
}
_FLOATS_BY_WIDTH: dict[int, type[astx.DataType]] = {
    BIT_WIDTH_16: astx.Float16,
    BIT_WIDTH_32: astx.Float32,
    BIT_WIDTH_64: astx.Float64,
}


@public
@typechecked
def clone_type(type_: astx.DataType) -> astx.DataType:
    """
    title: Clone an AST type by class.
    parameters:
      type_:
        type: astx.DataType
    returns:
      type: astx.DataType
    """
    if isinstance(type_, astx.UnionType):
        return astx.UnionType(
            tuple(clone_type(member) for member in type_.members),
            alias_name=type_.alias_name,
        )
    if isinstance(type_, astx.TemplateTypeVar):
        return astx.TemplateTypeVar(
            type_.name,
            bound=clone_type(type_.bound),
        )
    if isinstance(type_, astx.StructType):
        return astx.StructType(
            type_.name,
            resolved_name=type_.resolved_name,
            module_key=type_.module_key,
            qualified_name=type_.qualified_name,
        )
    if isinstance(type_, astx.ClassType):
        return astx.ClassType(
            type_.name,
            resolved_name=type_.resolved_name,
            module_key=type_.module_key,
            qualified_name=type_.qualified_name,
            ancestor_qualified_names=type_.ancestor_qualified_names,
        )
    if isinstance(type_, astx.NamespaceType):
        return astx.NamespaceType(
            type_.namespace_key,
            namespace_kind=type_.namespace_kind,
            display_name=type_.display_name,
        )
    if isinstance(type_, astx.PointerType):
        pointee_type = (
            clone_type(type_.pointee_type)
            if type_.pointee_type is not None
            else None
        )
        return astx.PointerType(pointee_type)
    if isinstance(type_, astx.BufferOwnerType):
        return type_.__class__()
    if isinstance(type_, astx.OpaqueHandleType):
        return astx.OpaqueHandleType(type_.handle_name)
    if isinstance(type_, astx.BufferViewType):
        element_type = (
            clone_type(type_.element_type)
            if type_.element_type is not None
            else None
        )
        return astx.BufferViewType(element_type)
    if isinstance(type_, astx.NdarrayType):
        element_type = (
            clone_type(type_.element_type)
            if type_.element_type is not None
            else None
        )
        return astx.NdarrayType(element_type)
    return type_.__class__()


@public
@typechecked
def display_type_name(type_: astx.DataType | None) -> str:
    """
    title: Return one stable human-facing type name.
    parameters:
      type_:
        type: astx.DataType | None
    returns:
      type: str
    """
    if type_ is None:
        return "<unknown>"
    if isinstance(type_, astx.UnionType):
        if type_.alias_name is not None:
            return type_.alias_name
        return " | ".join(
            display_type_name(member) for member in type_.members
        )
    if isinstance(type_, astx.TemplateTypeVar):
        return type_.name
    if isinstance(type_, astx.StructType):
        return type_.qualified_name or type_.name
    if isinstance(type_, astx.ClassType):
        return type_.qualified_name or type_.name
    if isinstance(type_, astx.NamespaceType):
        visible_name = type_.display_name or type_.namespace_key
        return f"{type_.namespace_kind.value} namespace '{visible_name}'"
    if isinstance(type_, astx.PointerType):
        if type_.pointee_type is None:
            return "PointerType"
        return f"PointerType[{display_type_name(type_.pointee_type)}]"
    if isinstance(type_, astx.OpaqueHandleType):
        return type_.handle_name
    if isinstance(type_, astx.BufferViewType):
        if type_.element_type is None:
            return "BufferViewType"
        return f"BufferViewType[{display_type_name(type_.element_type)}]"
    if isinstance(type_, astx.NdarrayType):
        if type_.element_type is None:
            return "NdarrayType"
        return f"NdarrayType[{display_type_name(type_.element_type)}]"
    return str(type_.__class__.__name__)


@public
@typechecked
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
    if isinstance(lhs, astx.UnionType) and isinstance(rhs, astx.UnionType):
        if lhs.alias_name != rhs.alias_name:
            return False
        if len(lhs.members) != len(rhs.members):
            return False
        return all(
            same_type(left_member, right_member)
            for left_member, right_member in zip(lhs.members, rhs.members)
        )
    if isinstance(lhs, astx.TemplateTypeVar) and isinstance(
        rhs,
        astx.TemplateTypeVar,
    ):
        return lhs.name == rhs.name and same_type(lhs.bound, rhs.bound)
    if isinstance(lhs, astx.StructType) and isinstance(rhs, astx.StructType):
        lhs_identity = lhs.qualified_name or lhs.name
        rhs_identity = rhs.qualified_name or rhs.name
        return lhs_identity == rhs_identity
    if isinstance(lhs, astx.ClassType) and isinstance(rhs, astx.ClassType):
        lhs_identity = lhs.qualified_name or lhs.name
        rhs_identity = rhs.qualified_name or rhs.name
        return lhs_identity == rhs_identity
    if isinstance(lhs, astx.NamespaceType) and isinstance(
        rhs,
        astx.NamespaceType,
    ):
        return (
            lhs.namespace_key == rhs.namespace_key
            and lhs.namespace_kind is rhs.namespace_kind
        )
    if isinstance(lhs, astx.PointerType) and isinstance(rhs, astx.PointerType):
        if lhs.pointee_type is None or rhs.pointee_type is None:
            return lhs.pointee_type is None and rhs.pointee_type is None
        return same_type(lhs.pointee_type, rhs.pointee_type)
    if isinstance(lhs, astx.OpaqueHandleType) and isinstance(
        rhs,
        astx.OpaqueHandleType,
    ):
        return lhs.handle_name == rhs.handle_name
    if isinstance(lhs, astx.BufferViewType) and isinstance(
        rhs,
        astx.BufferViewType,
    ):
        if lhs.element_type is None or rhs.element_type is None:
            return True
        return same_type(lhs.element_type, rhs.element_type)
    if isinstance(lhs, astx.NdarrayType) and isinstance(
        rhs,
        astx.NdarrayType,
    ):
        if lhs.element_type is None or rhs.element_type is None:
            return True
        return same_type(lhs.element_type, rhs.element_type)
    return lhs.__class__ is rhs.__class__


@public
@typechecked
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
@typechecked
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
@typechecked
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
@typechecked
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
@typechecked
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
@typechecked
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
@typechecked
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
@typechecked
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
@typechecked
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
@typechecked
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


@typechecked
def _type_for_width(
    width: int,
    table: dict[int, type[astx.DataType]],
) -> astx.DataType | None:
    """
    title: Instantiate a type family member for one width.
    parameters:
      width:
        type: int
      table:
        type: dict[int, type[astx.DataType]]
    returns:
      type: astx.DataType | None
    """
    type_cls = table.get(width)
    if type_cls is None:
        return None
    return type_cls()


@public
@typechecked
def float_promotion_width_for_integer_width(width: int) -> int:
    """
    title: Return the float width floor used when integers promote with floats.
    parameters:
      width:
        type: int
    returns:
      type: int
    """
    if width <= BIT_WIDTH_16:
        return BIT_WIDTH_16
    if width <= BIT_WIDTH_32:
        return BIT_WIDTH_32
    return BIT_WIDTH_64


@typechecked
def _common_integer_type(
    lhs: astx.DataType,
    rhs: astx.DataType,
) -> astx.DataType | None:
    """
    title: Return the canonical promoted type for two integers.
    parameters:
      lhs:
        type: astx.DataType
      rhs:
        type: astx.DataType
    returns:
      type: astx.DataType | None
    """
    lhs_width = bit_width(lhs)
    rhs_width = bit_width(rhs)

    if is_signed_integer_type(lhs) and is_signed_integer_type(rhs):
        return _type_for_width(
            max(lhs_width, rhs_width),
            _SIGNED_INTEGERS_BY_WIDTH,
        )

    if is_unsigned_type(lhs) and is_unsigned_type(rhs):
        return _type_for_width(
            max(lhs_width, rhs_width),
            _UNSIGNED_INTEGERS_BY_WIDTH,
        )

    signed_width = lhs_width if is_signed_integer_type(lhs) else rhs_width
    unsigned_width = lhs_width if is_unsigned_type(lhs) else rhs_width

    if signed_width > unsigned_width:
        return _type_for_width(signed_width, _SIGNED_INTEGERS_BY_WIDTH)
    return _type_for_width(
        max(lhs_width, rhs_width),
        _UNSIGNED_INTEGERS_BY_WIDTH,
    )


@typechecked
def _common_float_type(
    lhs: astx.DataType,
    rhs: astx.DataType,
) -> astx.DataType | None:
    """
    title: Return the canonical promoted type for operands with floats.
    parameters:
      lhs:
        type: astx.DataType
      rhs:
        type: astx.DataType
    returns:
      type: astx.DataType | None
    """
    float_width = max(
        bit_width(type_) for type_ in (lhs, rhs) if is_float_type(type_)
    )
    integer_width = max(
        (bit_width(type_) for type_ in (lhs, rhs) if is_integer_type(type_)),
        default=0,
    )
    target_width = max(
        float_width,
        float_promotion_width_for_integer_width(integer_width),
    )
    target_width = min(target_width, BIT_WIDTH_64)
    return _type_for_width(target_width, _FLOATS_BY_WIDTH)


@public
@typechecked
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
        return _common_float_type(lhs, rhs)
    return _common_integer_type(lhs, rhs)


@typechecked
def _is_safe_integer_assignment(
    target: astx.DataType,
    value: astx.DataType,
) -> bool:
    """
    title: Return whether one integer can implicitly promote into another.
    parameters:
      target:
        type: astx.DataType
      value:
        type: astx.DataType
    returns:
      type: bool
    """
    target_width = bit_width(target)
    value_width = bit_width(value)

    if is_signed_integer_type(target) and is_signed_integer_type(value):
        return target_width >= value_width
    if is_unsigned_type(target) and is_unsigned_type(value):
        return target_width >= value_width
    if is_signed_integer_type(target) and is_unsigned_type(value):
        return target_width > value_width
    return False


@typechecked
def _is_safe_float_assignment(
    target: astx.DataType,
    value: astx.DataType,
) -> bool:
    """
    title: Return whether a value can implicitly promote to a float target.
    parameters:
      target:
        type: astx.DataType
      value:
        type: astx.DataType
    returns:
      type: bool
    """
    target_width = bit_width(target)
    if is_float_type(value):
        return target_width >= bit_width(value)
    if is_integer_type(value):
        return target_width >= float_promotion_width_for_integer_width(
            bit_width(value)
        )
    return False


@public
@typechecked
def is_explicitly_castable(
    source: astx.DataType | None,
    target: astx.DataType | None,
) -> bool:
    """
    title: Return whether an explicit Cast expression is allowed.
    parameters:
      source:
        type: astx.DataType | None
      target:
        type: astx.DataType | None
    returns:
      type: bool
    """
    if source is None or target is None:
        return True
    if is_assignable(target, source):
        return True
    if (is_numeric_type(source) or is_boolean_type(source)) and (
        is_numeric_type(target) or is_boolean_type(target)
    ):
        return True
    if isinstance(target, (astx.String, astx.UTF8String)):
        return (
            is_string_type(source)
            or is_numeric_type(source)
            or (is_boolean_type(source))
        )
    return False


@public
@typechecked
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
    if isinstance(target, astx.UnionType):
        return any(is_assignable(member, value) for member in target.members)
    if isinstance(value, astx.UnionType):
        return all(is_assignable(target, member) for member in value.members)
    if isinstance(target, astx.TemplateTypeVar):
        return is_assignable(target.bound, value)
    if isinstance(value, astx.TemplateTypeVar):
        return is_assignable(target, value.bound)
    if isinstance(target, astx.ClassType) and isinstance(
        value, astx.ClassType
    ):
        target_identity = target.qualified_name or target.name
        value_identity = value.qualified_name or value.name
        if value_identity == target_identity:
            return True
        return target_identity in value.ancestor_qualified_names
    if is_integer_type(target) and is_integer_type(value):
        return _is_safe_integer_assignment(target, value)
    if is_float_type(target) and is_numeric_type(value):
        return _is_safe_float_assignment(target, value)
    if is_string_type(target) and is_string_type(value):
        return True
    if is_none_type(target) and is_none_type(value):
        return True
    return False

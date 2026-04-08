"""
title: Runtime type-checking helpers.
"""

from typing import Any, Callable, TypeVar

from public import public
from typeguard import (
    CollectionCheckStrategy,
    ForwardRefPolicy,
)
from typeguard import (
    typechecked as _typechecked,
)
from typeguard._config import global_config

_T = TypeVar("_T")

typechecked = _typechecked(
    forward_ref_policy=ForwardRefPolicy.IGNORE,
    collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS,
)

global_config.forward_ref_policy = ForwardRefPolicy.IGNORE
global_config.collection_check_strategy = CollectionCheckStrategy.ALL_ITEMS

__all__ = ["copy_type", "skip_unused", "typechecked"]


@public
@typechecked
def skip_unused(*args: Any, **kwargs: Any) -> None:
    """
    title: Referencing variables to pacify static analyzers.
    parameters:
      args:
        type: Any
        variadic: positional
      kwargs:
        type: Any
        variadic: keyword
    """
    for _arg in args:
        pass
    for _key in kwargs:
        pass


@public
@typechecked
def copy_type(f: _T) -> Callable[[Any], _T]:
    """
    title: Copy types for args, kwargs from parent class.
    parameters:
      f:
        type: _T
    returns:
      type: Callable[[Any], _T]
    """
    skip_unused(f)
    return lambda x: x

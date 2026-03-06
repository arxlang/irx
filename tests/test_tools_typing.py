"""
title: Tests for typing helper utilities.
"""

from irx.tools import typing as typing_tools


def test_skip_unused_accepts_args_and_kwargs() -> None:
    """
    title: skip_unused should accept positional and keyword values.
    """
    typing_tools.skip_unused(1, "two", alpha=3, beta="four")


def test_copy_type_returns_identity_callable() -> None:
    """
    title: copy_type should return a callable that preserves the argument.
    """
    sentinel = object()
    identity = typing_tools.copy_type(int)
    assert identity(sentinel) is sentinel

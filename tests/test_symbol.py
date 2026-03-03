"""
title: Test RegisterTable functionality.
"""

import pytest

from irx.symbol_table import RegisterTable

# Constants for test values used in each test
INITIAL_VALUE = 0
TEST_INCREMENT_1 = 3
TEST_INCREMENT_2 = 1
TEST_VALUE_1 = 2
TEST_VALUE_2 = 5
TEST_REDEFINE = 10
TEST_RESET_VALUE = 7


@pytest.fixture
def register_table() -> RegisterTable:
    """
    title: Fixture providing a fresh RegisterTable instance.

    Returns
    -------
      type: RegisterTable
    """
    return RegisterTable()


def test_initial_state(register_table: RegisterTable) -> None:
    """
    title: Test initialization creates empty stack.

    Parameters
    ----------
      register_table:
        type: RegisterTable
    """
    assert register_table.stack == []


def test_append_operation(register_table: RegisterTable) -> None:
    """
    title: Test append adds new context level.

    Parameters
    ----------
      register_table:
        type: RegisterTable
    """
    register_table.append()
    assert register_table.stack == [INITIAL_VALUE]

    register_table.append()
    assert register_table.stack == [INITIAL_VALUE, INITIAL_VALUE]


def test_increase_operation(register_table: RegisterTable) -> None:
    """
    title: Test increase modifies current context.

    Parameters
    ----------
      register_table:
        type: RegisterTable
    """
    register_table.append()
    new_value = register_table.increase(TEST_INCREMENT_1)
    assert new_value == TEST_INCREMENT_1
    assert register_table.stack == [TEST_INCREMENT_1]

    new_value = register_table.increase()
    assert new_value == TEST_INCREMENT_1 + TEST_INCREMENT_2
    assert register_table.stack == [TEST_INCREMENT_1 + TEST_INCREMENT_2]


def test_last_property(register_table: RegisterTable) -> None:
    """
    title: Test last property returns top value.

    Parameters
    ----------
      register_table:
        type: RegisterTable
    """
    register_table.append()
    register_table.increase(TEST_VALUE_1)
    assert register_table.last == TEST_VALUE_1

    register_table.append()
    assert register_table.last == INITIAL_VALUE
    register_table.increase(TEST_VALUE_2)
    assert register_table.last == TEST_VALUE_2


def test_pop_operation(register_table: RegisterTable) -> None:
    """
    title: Test pop removes current context.

    Parameters
    ----------
      register_table:
        type: RegisterTable
    """
    register_table.append()
    register_table.append()
    register_table.increase(3)
    assert register_table.stack == [0, 3]

    register_table.pop()
    assert register_table.stack == [0]


def test_redefine_operation(register_table: RegisterTable) -> None:
    """
    title: Test redefine overwrites current context.

    Parameters
    ----------
      register_table:
        type: RegisterTable
    """
    register_table.append()
    register_table.increase(2)
    register_table.redefine(10)
    assert register_table.stack == [10]

    register_table.append()
    register_table.redefine(5)
    assert register_table.stack == [10, 5]


def test_reset_operation(register_table: RegisterTable) -> None:
    """
    title: Test reset clears current context.

    Parameters
    ----------
      register_table:
        type: RegisterTable
    """
    register_table.append()
    register_table.increase(7)
    register_table.reset()
    assert register_table.stack == [0]

    register_table.append()
    register_table.reset()
    assert register_table.stack == [0, 0]


def test_nested_context_operations(register_table: RegisterTable) -> None:
    """
    title: Test complex nested context scenarios.

    Parameters
    ----------
      register_table:
        type: RegisterTable
    """
    # Enter global scope
    register_table.append()
    register_table.increase(TEST_VALUE_1)

    # Enter function scope
    register_table.append()
    register_table.increase(TEST_INCREMENT_1)

    assert register_table.last == TEST_INCREMENT_1
    assert register_table.stack == [TEST_VALUE_1, TEST_INCREMENT_1]

    # Modify function scope
    register_table.redefine(TEST_REDEFINE // 2)
    assert register_table.stack == [TEST_VALUE_1, 5]

    # Exit function scope
    register_table.pop()
    assert register_table.stack == [TEST_VALUE_1]

    # Reset global scope
    register_table.reset()
    assert register_table.stack == [INITIAL_VALUE]

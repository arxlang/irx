"""
title: Machine-readable assertion failure parsing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

from public import public

from irx.typecheck import typechecked

ASSERT_FAILURE_PREFIX = "ARX_ASSERT_FAIL"
ASSERT_FAILURE_FIELD_COUNT = 5


@public
@typechecked
@dataclass(frozen=True)
class AssertionFailureReport:
    """
    title: Parsed machine-readable assertion failure report.
    attributes:
      source:
        type: str
      line:
        type: int
      col:
        type: int
      message:
        type: str
    """

    source: str
    line: int
    col: int
    message: str


@public
@typechecked
def parse_assert_failure_line(line: str) -> AssertionFailureReport | None:
    """
    title: Parse one machine-readable assertion failure line.
    parameters:
      line:
        type: str
    returns:
      type: AssertionFailureReport | None
    """
    stripped = line.strip()
    prefix = f"{ASSERT_FAILURE_PREFIX}|"
    if not stripped.startswith(prefix):
        return None

    parts = stripped.split("|", 4)
    if len(parts) != ASSERT_FAILURE_FIELD_COUNT:
        return None

    _, source, line_text, col_text, message = parts
    try:
        line_number = int(line_text)
        col_number = int(col_text)
    except ValueError:
        return None

    return AssertionFailureReport(
        source=source,
        line=line_number,
        col=col_number,
        message=message,
    )


@public
@typechecked
def parse_assert_failure_output(stderr: str) -> AssertionFailureReport | None:
    """
    title: Parse the first machine-readable assertion failure from stderr.
    parameters:
      stderr:
        type: str
    returns:
      type: AssertionFailureReport | None
    """
    for line in stderr.splitlines():
        parsed = parse_assert_failure_line(line)
        if parsed is not None:
            return parsed
    return None

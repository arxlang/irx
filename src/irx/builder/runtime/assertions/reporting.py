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


@typechecked
def _decode_assert_failure_field(text: str) -> str:
    """
    title: Decode one escaped assertion failure protocol field.
    parameters:
      text:
        type: str
    returns:
      type: str
    """
    decoded: list[str] = []
    index = 0
    escapes = {
        "\\": "\\",
        "n": "\n",
        "p": "|",
        "r": "\r",
        "t": "\t",
    }

    while index < len(text):
        char = text[index]
        if char != "\\":
            decoded.append(char)
            index += 1
            continue

        if index + 1 >= len(text):
            decoded.append("\\")
            break

        escaped = text[index + 1]
        replacement = escapes.get(escaped)
        if replacement is None:
            decoded.append("\\")
            decoded.append(escaped)
        else:
            decoded.append(replacement)
        index += 2

    return "".join(decoded)


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

    parts = stripped.split("|", ASSERT_FAILURE_FIELD_COUNT - 1)
    if len(parts) != ASSERT_FAILURE_FIELD_COUNT:
        return None

    _, source, line_text, col_text, message = parts
    try:
        line_number = int(line_text)
        col_number = int(col_text)
    except ValueError:
        return None

    return AssertionFailureReport(
        source=_decode_assert_failure_field(source),
        line=line_number,
        col=col_number,
        message=_decode_assert_failure_field(message),
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

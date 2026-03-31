"""
title: General configuration module for pytest.
"""

import os
import tempfile

from difflib import SequenceMatcher
from pathlib import Path

import astx

from irx.builders.base import Builder

TEST_DATA_PATH = Path(__file__).parent / "data"


def similarity(text_a: str, text_b: str) -> float:
    """
    title: Calculate the similarity between two strings.
    parameters:
      text_a:
        type: str
      text_b:
        type: str
    returns:
      type: float
    """
    return SequenceMatcher(None, text_a, text_b).ratio()


def check_result(
    action: str,
    builder: Builder,
    module: astx.Module,
    expected_file: str = "",
    expected_output: str | None = None,
    similarity_factor: float = 0.35,  # TODO: change it to 0.95
    tolerance: float = 1e-4,
) -> None:
    """
    title: Check the result for translation or build.
    parameters:
      action:
        type: str
      builder:
        type: Builder
      module:
        type: astx.Module
      expected_file:
        type: str
      expected_output:
        type: str | None
      similarity_factor:
        type: float
      tolerance:
        type: float
    """
    if action == "build":
        filename_exe = ""
        with tempfile.NamedTemporaryFile(
            suffix=".exe",
            prefix="arx",
            dir="/tmp",
            delete=False,
        ) as fp:
            filename_exe = fp.name
            builder.build(module, output_file=filename_exe)

        result = builder.run(raise_on_error=False)
        exe_result = result.stdout.strip() or str(result.returncode)

        if expected_output:
            message = (
                f"Expected `{expected_output}`, "
                f"but the result is `{exe_result}`"
            )
            assert expected_output == exe_result, message

        os.unlink(filename_exe)

    elif action == "translate":
        with open(TEST_DATA_PATH / expected_file, "r") as f:
            expected = f.read()
        ir_result = builder.translate(module)
        print(" TEST ".center(80, "="))
        print("==== EXPECTED =====")
        print(f"\n{expected}\n")
        print("==== results =====")
        print(f"\n{ir_result}\n")
        print("=" * 80)
        assert similarity(ir_result, expected) >= similarity_factor

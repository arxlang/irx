# """General configuration module for pytest."""

# import os
# import tempfile

# from difflib import SequenceMatcher
# from pathlib import Path
# from typing import Optional

# import astx

# from irx.builders.base import Builder

# TEST_DATA_PATH = Path(__file__).parent / "data"


# def similarity(text_a: str, text_b: str) -> float:
#     """Calculate the similarity between two strings."""
#     return SequenceMatcher(None, text_a, text_b).ratio()


# def check_result(
#     action: str,
#     builder: Builder,
#     module: astx.Module,
#     expected_file: str = "",
#     expected_output: Optional[str] = None,
#     similarity_factor: float = 0.35,  # TODO: change it to 0.95
# ) -> None:
#     """Check the result for translation or build."""
#     if action == "build":
#         filename_exe = ""
#         with tempfile.NamedTemporaryFile(
#             suffix=".exe",
#             prefix="arx",
#             dir="/tmp",
#             delete=False,
#         ) as fp:
#             filename_exe = fp.name
#             builder.build(module, output_file=filename_exe)
#         exe_result = builder.run()

#         if expected_output:
#             message = f"Expected {expected_output}, but result is {exe_result}"
#             assert expected_output == exe_result, message
#         os.unlink(filename_exe)
#     elif action == "translate":
#         with open(TEST_DATA_PATH / expected_file, "r") as f:
#             expected = f.read()
#         result = builder.translate(module)
#         print(" TEST ".center(80, "="))
#         print("==== EXPECTED =====")
#         print(f"\n{expected}\n")
#         print("==== results =====")
#         print(f"\n{result}\n")
#         print("=" * 80)
#         assert similarity(result, expected) >= similarity_factor

"""General configuration module for pytest."""

import math
import os
import subprocess
import tempfile

from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import astx

from irx.builders.base import Builder

TEST_DATA_PATH = Path(__file__).parent / "data"


def similarity(text_a: str, text_b: str) -> float:
    """Calculate the similarity between two strings."""
    return SequenceMatcher(None, text_a, text_b).ratio()


def check_result(
    action: str,
    builder: Builder,
    module: astx.Module,
    expected_file: str = "",
    expected_output: Optional[str] = None,
    similarity_factor: float = 0.35,
    tolerance: float = 1e-4,
) -> None:
    """Check the result for translation or build."""
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

        try:
            result = (
                subprocess.check_output([filename_exe]).decode("utf-8").strip()
            )
        except subprocess.CalledProcessError as e:
            raise AssertionError(
                f"Program failed with exit code {e.returncode}"
            )

        os.unlink(filename_exe)

        if expected_output is not None:
            try:
                expected_val = float(expected_output)
                result_val = float(result)
                assert math.isclose(
                    result_val, expected_val, rel_tol=tolerance
                ), f"Expected {expected_val}, got {result_val}"
            except ValueError:
                assert result == expected_output, (
                    f"Expected '{expected_output}', got '{result}'"
                )

    elif action == "translate":
        with open(TEST_DATA_PATH / expected_file, "r") as f:
            expected = f.read()
        result = builder.translate(module)
        print(" TEST ".center(80, "="))
        print("==== EXPECTED =====")
        print(f"\n{expected}\n")
        print("==== results =====")
        print(f"\n{result}\n")
        print("=" * 80)
        assert similarity(result, expected) >= similarity_factor

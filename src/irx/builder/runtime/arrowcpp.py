"""
title: Arrow C++ runtime build helpers.
"""

from __future__ import annotations

import sys

from pathlib import Path

import pyarrow

from arx_arrowcpp_sources import (
    bundled_arrowcpp_version,
)
from arx_arrowcpp_sources import (
    get_include_dir as get_arrowcpp_source_include_dir,
)

from irx.typecheck import typechecked


@typechecked
def arrowcpp_include_dirs() -> tuple[Path, ...]:
    """
    title: Return Arrow C++ include directories for native runtime builds.
    returns:
      type: tuple[Path, Ellipsis]
    """
    return (
        get_arrowcpp_source_include_dir(),
        Path(pyarrow.get_include()),
    )


@typechecked
def arrowcpp_compile_flags() -> tuple[str, ...]:
    """
    title: Return compiler flags required by Arrow C++ headers.
    returns:
      type: tuple[str, Ellipsis]
    """
    return ("-std=c++20",)


@typechecked
def arrowcpp_linker_flags() -> tuple[str, ...]:
    """
    title: Return linker flags for the bundled Arrow C++ runtime.
    returns:
      type: tuple[str, Ellipsis]
    """
    library = _find_pyarrow_library("arrow")
    library_dir = library.parent
    flags = [str(library)]

    if sys.platform != "win32":
        flags.append(f"-Wl,-rpath,{library_dir}")

    return tuple(flags)


@typechecked
def arrowcpp_runtime_metadata() -> dict[str, object]:
    """
    title: Return Arrow C++ runtime implementation metadata.
    returns:
      type: dict[str, object]
    """
    return {
        "implementation": "arrow-cpp",
        "arrowcpp_version": bundled_arrowcpp_version(),
        "arrowcpp_include_dirs": tuple(
            str(path) for path in arrowcpp_include_dirs()
        ),
        "arrowcpp_libraries": tuple(arrowcpp_linker_flags()),
    }


@typechecked
def _find_pyarrow_library(name: str) -> Path:
    """
    title: Find one PyArrow-shipped Arrow C++ shared library.
    parameters:
      name:
        type: str
    returns:
      type: Path
    """
    prefixes = _library_prefixes(name)
    suffixes = _library_suffixes()

    for directory_name in pyarrow.get_library_dirs():
        directory = Path(directory_name)
        for prefix in prefixes:
            for suffix in suffixes:
                candidate = directory / f"{prefix}{suffix}"
                if candidate.exists():
                    return candidate

            versioned = sorted(
                directory.glob(f"{prefix}{_versioned_library_glob()}")
            )
            if versioned:
                return versioned[0]

    raise RuntimeError(
        f"PyArrow did not provide a linkable lib{name!s} shared library"
    )


@typechecked
def _library_prefixes(name: str) -> tuple[str, ...]:
    """
    title: Return platform-specific library name prefixes.
    parameters:
      name:
        type: str
    returns:
      type: tuple[str, Ellipsis]
    """
    if sys.platform == "win32":
        return (name, f"lib{name}")
    return (f"lib{name}",)


@typechecked
def _library_suffixes() -> tuple[str, ...]:
    """
    title: Return platform-specific exact shared library suffixes.
    returns:
      type: tuple[str, Ellipsis]
    """
    if sys.platform == "darwin":
        return (".dylib",)
    if sys.platform == "win32":
        return (".lib", ".dll")
    return (".so",)


@typechecked
def _versioned_library_glob() -> str:
    """
    title: Return platform-specific versioned shared library glob.
    returns:
      type: str
    """
    if sys.platform == "darwin":
        return "*.dylib"
    if sys.platform == "win32":
        return "*.dll"
    return ".so*"


__all__ = [
    "arrowcpp_compile_flags",
    "arrowcpp_include_dirs",
    "arrowcpp_linker_flags",
    "arrowcpp_runtime_metadata",
]

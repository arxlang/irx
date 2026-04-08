"""
title: Conditional native linking helpers for runtime features.
"""

from __future__ import annotations

import hashlib
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from irx.runtime.features import NativeArtifact
from irx.typecheck import typechecked


@typechecked
@dataclass(frozen=True)
class NativeLinkInputs:
    """
    title: Collected native linker inputs for one build.
    attributes:
      objects:
        type: tuple[Path, Ellipsis]
      linker_flags:
        type: tuple[str, Ellipsis]
    """

    objects: tuple[Path, ...]
    linker_flags: tuple[str, ...]


def compile_native_artifacts(
    artifacts: Sequence[NativeArtifact],
    build_dir: Path,
    clang_binary: str = "clang",
) -> NativeLinkInputs:
    """
    title: Compile or collect native artifacts for linking.
    parameters:
      artifacts:
        type: Sequence[NativeArtifact]
      build_dir:
        type: Path
      clang_binary:
        type: str
    returns:
      type: NativeLinkInputs
    """
    objects: list[Path] = []
    linker_flags: list[str] = []

    for artifact in artifacts:
        linker_flags.extend(artifact.link_flags)
        if artifact.kind == "c_source":
            objects.append(
                _compile_c_source(
                    artifact=artifact,
                    build_dir=build_dir,
                    clang_binary=clang_binary,
                )
            )
            continue

        if artifact.kind in {"object", "static_library"}:
            objects.append(artifact.path)
            continue

        raise ValueError(f"Unsupported native artifact kind '{artifact.kind}'")

    return NativeLinkInputs(tuple(objects), tuple(linker_flags))


def link_executable(
    primary_object: Path,
    output_file: Path,
    artifacts: Sequence[NativeArtifact],
    linker_flags: Sequence[str] = (),
    clang_binary: str = "clang",
) -> None:
    """
    title: Link the main object file plus optional runtime artifacts.
    parameters:
      primary_object:
        type: Path
      output_file:
        type: Path
      artifacts:
        type: Sequence[NativeArtifact]
      linker_flags:
        type: Sequence[str]
      clang_binary:
        type: str
    """
    build_dir = primary_object.parent
    link_inputs = compile_native_artifacts(
        artifacts=artifacts,
        build_dir=build_dir,
        clang_binary=clang_binary,
    )

    command = [clang_binary, str(primary_object)]
    command.extend(str(obj) for obj in link_inputs.objects)
    command.extend(link_inputs.linker_flags)
    command.extend(linker_flags)
    command.extend(["-o", str(output_file)])
    _run_checked(command)


def _compile_c_source(
    artifact: NativeArtifact,
    build_dir: Path,
    clang_binary: str,
) -> Path:
    """
    title: Compile c source.
    parameters:
      artifact:
        type: NativeArtifact
      build_dir:
        type: Path
      clang_binary:
        type: str
    returns:
      type: Path
    """
    digest = hashlib.sha256(str(artifact.path).encode("utf8")).hexdigest()[:12]
    object_path = build_dir / f"{artifact.path.stem}_{digest}.o"

    command = [
        clang_binary,
        "-c",
        str(artifact.path),
        "-o",
        str(object_path),
        "-fPIC",
    ]

    for include_dir in artifact.include_dirs:
        command.extend(["-I", str(include_dir)])

    command.extend(artifact.compile_flags)
    _run_checked(command)
    return object_path


def _run_checked(command: Sequence[str]) -> None:
    """
    title: Run checked.
    parameters:
      command:
        type: Sequence[str]
    """
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        details = stderr or stdout or str(exc.returncode)
        raise RuntimeError(details) from exc

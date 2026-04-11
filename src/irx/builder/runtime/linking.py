"""
title: Conditional native linking helpers for runtime features.
"""

from __future__ import annotations

import hashlib
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from irx.builder.runtime.features import NativeArtifact
from irx.diagnostics import (
    Diagnostic,
    DiagnosticCodes,
    LinkingError,
    NativeCompileError,
    RuntimeFeatureError,
)
from irx.typecheck import typechecked

MAX_COMMAND_OUTPUT_LINES = 8
MAX_COMMAND_OUTPUT_CHARS = 400


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


@typechecked
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

        raise RuntimeFeatureError(
            Diagnostic(
                message=(
                    f"runtime artifact '{artifact.path}' uses unsupported "
                    f"kind '{artifact.kind}'"
                ),
                code=DiagnosticCodes.RUNTIME_ARTIFACT_KIND_INVALID,
                phase="runtime",
                notes=(
                    "supported artifact kinds: c_source, object, "
                    "static_library",
                ),
            )
        )

    return NativeLinkInputs(tuple(objects), tuple(linker_flags))


@typechecked
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
    _run_checked(
        command,
        error_type=LinkingError,
        phase="link",
        code=DiagnosticCodes.LINK_FAILED,
        message=f"link failed while producing '{output_file.name}'",
        notes=(f"output path: {output_file}",),
        hint=(
            "install clang or pass a valid clang_binary value"
            if clang_binary == "clang"
            else None
        ),
    )


@typechecked
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
    _run_checked(
        command,
        error_type=NativeCompileError,
        phase="native-compile",
        code=DiagnosticCodes.NATIVE_COMPILE_FAILED,
        message=(
            f"native compile failed for runtime artifact '{artifact.path}'"
        ),
        notes=(f"object path: {object_path}",),
        hint=(
            "install clang or pass a valid clang_binary value"
            if clang_binary == "clang"
            else None
        ),
    )
    return object_path


@typechecked
def _command_excerpt(text: str) -> str:
    """
    title: Return one compact stdout or stderr excerpt.
    parameters:
      text:
        type: str
    returns:
      type: str
    """
    stripped = text.strip()
    if not stripped:
        return ""
    lines = stripped.splitlines()[:MAX_COMMAND_OUTPUT_LINES]
    excerpt = "\n".join(lines)
    if len(excerpt) > MAX_COMMAND_OUTPUT_CHARS:
        return f"{excerpt[:MAX_COMMAND_OUTPUT_CHARS].rstrip()}..."
    if len(lines) < len(stripped.splitlines()):
        return f"{excerpt}\n..."
    return excerpt


@typechecked
def _format_command(command: Sequence[str]) -> str:
    """
    title: Render one command line for diagnostics.
    parameters:
      command:
        type: Sequence[str]
    returns:
      type: str
    """
    return " ".join(command)


@typechecked
def _run_checked(
    command: Sequence[str],
    *,
    error_type: type[NativeCompileError] | type[LinkingError],
    phase: str,
    code: str,
    message: str,
    notes: Sequence[str] = (),
    hint: str | None = None,
) -> None:
    """
    title: Run checked.
    parameters:
      command:
        type: Sequence[str]
      error_type:
        type: type[NativeCompileError] | type[LinkingError]
      phase:
        type: str
      code:
        type: str
      message:
        type: str
      notes:
        type: Sequence[str]
      hint:
        type: str | None
    """
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise error_type(
            Diagnostic(
                message=message,
                code=code,
                phase=phase,
                notes=(
                    *tuple(notes),
                    f"command: {_format_command(command)}",
                ),
                hint=hint,
                cause=exc,
            )
        ) from exc

    if result.returncode == 0:
        return

    diagnostic_notes = [*notes, f"command: {_format_command(command)}"]
    stderr_excerpt = _command_excerpt(result.stderr)
    stdout_excerpt = _command_excerpt(result.stdout)
    if stderr_excerpt:
        diagnostic_notes.append(f"stderr: {stderr_excerpt}")
    if stdout_excerpt:
        diagnostic_notes.append(f"stdout: {stdout_excerpt}")
    if not stderr_excerpt and not stdout_excerpt:
        diagnostic_notes.append(f"exit code: {result.returncode}")

    raise error_type(
        Diagnostic(
            message=message,
            code=code,
            phase=phase,
            notes=tuple(diagnostic_notes),
            hint=hint,
        )
    )

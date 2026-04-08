"""
title: Runtime feature specifications and LLVM symbol helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Mapping, TypeAlias, cast

from llvmlite import ir

from irx.typecheck import typechecked

if TYPE_CHECKING:
    from irx.builder.protocols import VisitorProtocol

NativeArtifactKind = Literal["c_source", "object", "static_library"]
if TYPE_CHECKING:
    RuntimeSymbolFactory: TypeAlias = Callable[[VisitorProtocol], ir.Function]
else:
    RuntimeSymbolFactory: TypeAlias = Callable[[object], ir.Function]


@typechecked
@dataclass(frozen=True)
class NativeArtifact:
    """
    title: Describe one native artifact required by a runtime feature.
    attributes:
      kind:
        type: NativeArtifactKind
      path:
        type: Path
      include_dirs:
        type: tuple[Path, Ellipsis]
      compile_flags:
        type: tuple[str, Ellipsis]
      link_flags:
        type: tuple[str, Ellipsis]
    """

    kind: NativeArtifactKind
    path: Path
    include_dirs: tuple[Path, ...] = ()
    compile_flags: tuple[str, ...] = ()
    link_flags: tuple[str, ...] = ()


@typechecked
@dataclass(frozen=True)
class ExternalSymbolSpec:
    """
    title: Describe one external symbol exposed by a runtime feature.
    attributes:
      name:
        type: str
      declare:
        type: RuntimeSymbolFactory
    """

    name: str
    declare: RuntimeSymbolFactory


@typechecked
@dataclass(frozen=True)
class RuntimeFeature:
    """
    title: Describe one optional native runtime feature.
    attributes:
      name:
        type: str
      symbols:
        type: Mapping[str, ExternalSymbolSpec]
      artifacts:
        type: tuple[NativeArtifact, Ellipsis]
      linker_flags:
        type: tuple[str, Ellipsis]
      metadata:
        type: Mapping[str, object]
    """

    name: str
    symbols: Mapping[str, ExternalSymbolSpec] = field(default_factory=dict)
    artifacts: tuple[NativeArtifact, ...] = ()
    linker_flags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


def declare_external_function(
    module: ir.Module, name: str, fn_type: ir.FunctionType
) -> ir.Function:
    """
    title: Declare or reuse an external function in an LLVM module.
    parameters:
      module:
        type: ir.Module
      name:
        type: str
      fn_type:
        type: ir.FunctionType
    returns:
      type: ir.Function
    """
    existing = module.globals.get(name)
    if existing is not None:
        if not isinstance(existing, ir.Function):
            raise TypeError(f"Global '{name}' is not a function")
        if existing.function_type != fn_type:
            raise TypeError(
                f"Function '{name}' already exists with a mismatch"
            )
        return cast(ir.Function, existing)

    fn = ir.Function(module, fn_type, name=name)
    fn.linkage = "external"
    return fn

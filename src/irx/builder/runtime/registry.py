"""
title: Runtime feature registry and per-module activation state.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Iterable

from llvmlite import ir

from irx.builder.runtime.arrow.feature import build_arrow_runtime_feature
from irx.builder.runtime.buffer.feature import build_buffer_runtime_feature
from irx.builder.runtime.feature_libc import build_libc_runtime_feature
from irx.builder.runtime.feature_libm import build_libm_runtime_feature
from irx.builder.runtime.features import NativeArtifact, RuntimeFeature
from irx.diagnostics import (
    Diagnostic,
    DiagnosticCodes,
    RuntimeFeatureError,
)
from irx.typecheck import typechecked

if TYPE_CHECKING:
    from irx.builder.protocols import VisitorProtocol


@typechecked
class RuntimeFeatureRegistry:
    """
    title: Registry of named runtime features.
    attributes:
      _features:
        type: dict[str, RuntimeFeature]
    """

    def __init__(self) -> None:
        """
        title: Initialize RuntimeFeatureRegistry.
        """
        self._features: dict[str, RuntimeFeature] = {}

    def register(self, feature: RuntimeFeature) -> None:
        """
        title: Register a runtime feature by name.
        parameters:
          feature:
            type: RuntimeFeature
        """
        if feature.name in self._features:
            raise ValueError(
                f"Runtime feature '{feature.name}' already exists"
            )
        self._features[feature.name] = feature

    def get(self, name: str) -> RuntimeFeature:
        """
        title: Return a runtime feature by name.
        parameters:
          name:
            type: str
        returns:
          type: RuntimeFeature
        """
        try:
            return self._features[name]
        except KeyError as exc:
            available = self.names()
            raise RuntimeFeatureError(
                Diagnostic(
                    message=f"runtime feature '{name}' is not registered",
                    code=DiagnosticCodes.RUNTIME_FEATURE_UNKNOWN,
                    phase="runtime",
                    notes=(
                        f"available runtime features: {', '.join(available)}",
                    )
                    if available
                    else (),
                    cause=KeyError(name),
                )
            ) from exc

    def names(self) -> tuple[str, ...]:
        """
        title: Return the registered runtime feature names.
        returns:
          type: tuple[str, Ellipsis]
        """
        return tuple(sorted(self._features))


@typechecked
class RuntimeFeatureState:
    """
    title: Track feature activation and symbol declarations for one module.
    attributes:
      _owner:
        type: VisitorProtocol
      _registry:
        type: RuntimeFeatureRegistry
      _active_features:
        type: set[str]
      _declared_symbols:
        type: dict[tuple[str, str], ir.Function]
    """

    _owner: VisitorProtocol
    _registry: RuntimeFeatureRegistry
    _active_features: set[str]
    _declared_symbols: dict[tuple[str, str], ir.Function]

    def __init__(
        self,
        owner: "VisitorProtocol",
        registry: RuntimeFeatureRegistry,
        active_features: Iterable[str] | None = None,
    ) -> None:
        """
        title: Initialize RuntimeFeatureState.
        parameters:
          owner:
            type: VisitorProtocol
          registry:
            type: RuntimeFeatureRegistry
          active_features:
            type: Iterable[str] | None
        """
        self._owner = owner
        self._registry = registry
        self._active_features: set[str] = set()
        self._declared_symbols: dict[tuple[str, str], ir.Function] = {}

        if active_features is None:
            return

        for feature_name in active_features:
            self.activate(feature_name)

    def activate(self, feature_name: str) -> None:
        """
        title: Activate a runtime feature for the current module.
        parameters:
          feature_name:
            type: str
        """
        self._registry.get(feature_name)
        self._active_features.add(feature_name)

    def is_active(self, feature_name: str) -> bool:
        """
        title: Return whether a feature is active for the current module.
        parameters:
          feature_name:
            type: str
        returns:
          type: bool
        """
        return feature_name in self._active_features

    def active_feature_names(self) -> tuple[str, ...]:
        """
        title: Return the active feature names.
        returns:
          type: tuple[str, Ellipsis]
        """
        return tuple(sorted(self._active_features))

    def feature(self, feature_name: str) -> RuntimeFeature:
        """
        title: Return a registered feature by name.
        parameters:
          feature_name:
            type: str
        returns:
          type: RuntimeFeature
        """
        return self._registry.get(feature_name)

    def feature_declares_symbol(
        self,
        feature_name: str,
        symbol_name: str,
    ) -> bool:
        """
        title: Return whether one feature declares one symbol.
        parameters:
          feature_name:
            type: str
          symbol_name:
            type: str
        returns:
          type: bool
        """
        return symbol_name in self.feature(feature_name).symbols

    def require_symbol(
        self,
        feature_name: str,
        symbol_name: str,
    ) -> ir.Function:
        """
        title: Activate a feature and declare one of its external symbols.
        parameters:
          feature_name:
            type: str
          symbol_name:
            type: str
        returns:
          type: ir.Function
        """
        self.activate(feature_name)
        cache_key = (feature_name, symbol_name)
        cached = self._declared_symbols.get(cache_key)
        if cached is not None:
            return cached

        feature = self.feature(feature_name)
        try:
            symbol_spec = feature.symbols[symbol_name]
        except KeyError as exc:
            raise RuntimeFeatureError(
                Diagnostic(
                    message=(
                        f"runtime feature '{feature_name}' does not declare "
                        f"symbol '{symbol_name}'"
                    ),
                    code=DiagnosticCodes.RUNTIME_FEATURE_SYMBOL_MISSING,
                    phase="runtime",
                    notes=(
                        "declared symbols: "
                        f"{', '.join(sorted(feature.symbols))}",
                    )
                    if feature.symbols
                    else (),
                    cause=KeyError(symbol_name),
                )
            ) from exc

        declared = symbol_spec.declare(self._owner)
        self._declared_symbols[cache_key] = declared
        return declared

    def native_artifacts(self) -> tuple[NativeArtifact, ...]:
        """
        title: Return the native artifacts required by active features.
        returns:
          type: tuple[NativeArtifact, Ellipsis]
        """
        artifacts: list[NativeArtifact] = []
        seen: set[tuple[str, str]] = set()

        for feature_name in self.active_feature_names():
            feature = self.feature(feature_name)
            for artifact in feature.artifacts:
                dedupe_key = (artifact.kind, str(artifact.path))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                artifacts.append(artifact)

        return tuple(artifacts)

    def linker_flags(self) -> tuple[str, ...]:
        """
        title: Return the linker flags required by active features.
        returns:
          type: tuple[str, Ellipsis]
        """
        flags: list[str] = []
        seen: set[str] = set()

        for feature_name in self.active_feature_names():
            feature = self.feature(feature_name)
            for flag in feature.linker_flags:
                if flag in seen:
                    continue
                seen.add(flag)
                flags.append(flag)

        return tuple(flags)


@lru_cache(maxsize=1)
@typechecked
def get_default_runtime_feature_registry() -> RuntimeFeatureRegistry:
    """
    title: Build the default runtime feature registry.
    returns:
      type: RuntimeFeatureRegistry
    """
    registry = RuntimeFeatureRegistry()
    registry.register(build_libc_runtime_feature())
    registry.register(build_libm_runtime_feature())
    registry.register(build_buffer_runtime_feature())
    registry.register(build_arrow_runtime_feature())
    return registry

"""
title: Canonical array runtime feature support for IRx.
"""

from irx.builder.runtime.array.feature import (
    ARRAY_PRIMITIVE_TYPE_SPECS,
    ARROW_PRIMITIVE_TYPE_SPECS,
    build_array_runtime_feature,
    build_arrow_runtime_feature,
)

__all__ = [
    "ARRAY_PRIMITIVE_TYPE_SPECS",
    "ARROW_PRIMITIVE_TYPE_SPECS",
    "build_array_runtime_feature",
    "build_arrow_runtime_feature",
]

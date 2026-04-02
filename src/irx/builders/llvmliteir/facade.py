"""
title: Public llvmliteir facade and composed backend classes.
"""

from __future__ import annotations

import os
import tempfile

from pathlib import Path

import astx

from llvmlite import binding as llvm
from public import public

from irx.builders.base import Builder as BaseBuilder
from irx.builders.llvmliteir.core import _VisitorCore
from irx.builders.llvmliteir.visitors import (
    ArrowVisitorMixin,
    BinaryOpVisitorMixin,
    ControlFlowVisitorMixin,
    FunctionVisitorMixin,
    LiteralVisitorMixin,
    ModuleVisitorMixin,
    SystemVisitorMixin,
    TemporalVisitorMixin,
    UnaryOpVisitorMixin,
    VariableVisitorMixin,
)
from irx.runtime.linking import link_executable


@public
class Visitor(
    LiteralVisitorMixin,
    VariableVisitorMixin,
    UnaryOpVisitorMixin,
    BinaryOpVisitorMixin,
    ControlFlowVisitorMixin,
    FunctionVisitorMixin,
    TemporalVisitorMixin,
    ArrowVisitorMixin,
    SystemVisitorMixin,
    ModuleVisitorMixin,
    _VisitorCore,
):
    pass


@public
class Builder(BaseBuilder):
    translator: Visitor

    def __init__(self) -> None:
        super().__init__()
        self.translator = self._new_translator()

    def _new_translator(self) -> Visitor:
        return Visitor(active_runtime_features=set(self.runtime_feature_names))

    def translate(self, expr: astx.AST) -> str:
        self.translator = self._new_translator()
        return self.translator.translate(expr)

    def build(self, node: astx.AST, output_file: str) -> None:
        result = self.translate(node)
        result_mod = llvm.parse_assembly(result)
        result_object = self.translator.target_machine.emit_object(result_mod)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.tmp_path = temp_dir
            file_path_o = Path(temp_dir) / "irx_module.o"

            with open(file_path_o, "wb") as handle:
                handle.write(result_object)

            self.output_file = output_file
            link_executable(
                primary_object=file_path_o,
                output_file=Path(self.output_file),
                artifacts=self.translator.runtime_features.native_artifacts(),
                linker_flags=self.translator.runtime_features.linker_flags(),
            )

        os.chmod(self.output_file, 0o755)

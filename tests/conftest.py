"""
title: General configuration module for pytest.
"""

import os
import tempfile

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import pytest

from irx import astx
from irx.analysis import ModuleKey, ParsedModule
from irx.builders.base import Builder, CommandResult
from irx.builders.llvmliteir import Builder as LLVMBuilder
from irx.builders.llvmliteir import Visitor as LLVMVisitor
from llvmlite import binding as llvm
from llvmlite import ir

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
        if expected_output is not None:
            assert_build_output(builder, module, expected_output)
        else:
            build_and_run(builder, module)

    elif action == "translate":
        with open(TEST_DATA_PATH / expected_file, "r") as f:
            expected = f.read()
        ir_result = translate_ir(builder, module)
        print(" TEST ".center(80, "="))
        print("==== EXPECTED =====")
        print(f"\n{expected}\n")
        print("==== results =====")
        print(f"\n{ir_result}\n")
        print("=" * 80)
        assert similarity(ir_result, expected) >= similarity_factor


def build_and_run(builder: Builder, module: astx.Module) -> CommandResult:
    """
    title: Build a module and run the resulting executable.
    parameters:
      builder:
        type: Builder
      module:
        type: astx.Module
    returns:
      type: CommandResult
    """
    output_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".exe",
            prefix="arx",
            dir="/tmp",
            delete=False,
        ) as fp:
            output_path = fp.name

        builder.build(module, output_file=output_path)
        return builder.run(raise_on_error=False)
    finally:
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)


def assert_build_output(
    builder: Builder,
    module: astx.Module,
    expected_output: str,
) -> None:
    """
    title: Assert that building and running a module yields exact output.
    parameters:
      builder:
        type: Builder
      module:
        type: astx.Module
      expected_output:
        type: str
    """
    result = build_and_run(builder, module)
    actual_output = result.stdout.strip() or str(result.returncode)
    assert actual_output == expected_output, (
        f"Expected `{expected_output}`, but got `{actual_output}` "
        f"(stderr={result.stderr.strip()!r})"
    )


def assert_build_succeeds(builder: Builder, module: astx.Module) -> None:
    """
    title: Assert that a module builds and runs successfully.
    parameters:
      builder:
        type: Builder
      module:
        type: astx.Module
    """
    result = build_and_run(builder, module)
    assert result.returncode == 0, (
        f"Expected build/run success, got exit {result.returncode} "
        f"with stderr {result.stderr.strip()!r}"
    )


def translate_ir(builder: Builder, module: astx.Module) -> str:
    """
    title: Translate a module to LLVM IR text.
    parameters:
      builder:
        type: Builder
      module:
        type: astx.Module
    returns:
      type: str
    """
    return builder.translate(module)


def assert_ir_parses(ir_text: str) -> None:
    """
    title: Assert that LLVM can parse generated IR.
    parameters:
      ir_text:
        type: str
    """
    llvm.parse_assembly(ir_text)


def make_main_module(
    *nodes: astx.AST,
    return_type: astx.DataType | None = None,
) -> astx.Module:
    """
    title: Build a small module with a single main function.
    parameters:
      return_type:
        type: astx.DataType | None
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    module = astx.Module()
    prototype = astx.FunctionPrototype(
        "main",
        args=astx.Arguments(),
        return_type=cast(Any, return_type or astx.Int32()),
    )
    body = astx.Block()
    for node in nodes:
        body.append(node)
    module.block.append(astx.FunctionDef(prototype=prototype, body=body))
    return module


def make_module(name: str, *nodes: astx.AST) -> astx.Module:
    """
    title: Build a named module with the provided top-level nodes.
    parameters:
      name:
        type: str
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    module = astx.Module(name=name)
    for node in nodes:
        module.block.append(node)
    return module


def make_parsed_module(
    key: str,
    *nodes: astx.AST,
    module_name: str | None = None,
    display_name: str | None = None,
) -> ParsedModule:
    """
    title: Build a ParsedModule test fixture.
    parameters:
      key:
        type: str
      module_name:
        type: str | None
      display_name:
        type: str | None
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: ParsedModule
    """
    module = make_module(module_name or key, *nodes)
    return ParsedModule(
        key=ModuleKey(key),
        ast=module,
        display_name=display_name or key,
    )


class StaticImportResolver:
    """
    title: Small test resolver backed by an in-memory module mapping.
    attributes:
      modules:
        type: dict[str, ParsedModule]
    """

    modules: dict[str, ParsedModule]

    def __init__(self, modules: dict[str, ParsedModule]) -> None:
        """
        title: Initialize StaticImportResolver.
        parameters:
          modules:
            type: dict[str, ParsedModule]
        """
        self.modules = modules

    def __call__(
        self,
        requesting_module_key: ModuleKey,
        import_node: astx.ImportStmt | astx.ImportFromStmt,
        requested_specifier: str,
    ) -> ParsedModule:
        """
        title: Resolve one requested specifier.
        parameters:
          requesting_module_key:
            type: ModuleKey
          import_node:
            type: astx.ImportStmt | astx.ImportFromStmt
          requested_specifier:
            type: str
        returns:
          type: ParsedModule
        """
        _ = requesting_module_key
        _ = import_node
        if requested_specifier not in self.modules:
            raise LookupError(requested_specifier)
        return self.modules[requested_specifier]


def translate_modules_ir(
    builder: LLVMBuilder,
    root: ParsedModule,
    resolver: StaticImportResolver,
) -> str:
    """
    title: Translate a parsed module graph to LLVM IR text.
    parameters:
      builder:
        type: LLVMBuilder
      root:
        type: ParsedModule
      resolver:
        type: StaticImportResolver
    returns:
      type: str
    """
    return builder.translate_modules(root, resolver)


@pytest.fixture
def llvm_builder() -> LLVMBuilder:
    """
    title: Return a fresh llvmliteir builder.
    returns:
      type: LLVMBuilder
    """
    return LLVMBuilder()


@pytest.fixture
def llvm_visitor() -> LLVMVisitor:
    """
    title: Return a fresh llvmliteir visitor with an empty result stack.
    returns:
      type: LLVMVisitor
    """
    builder = LLVMBuilder()
    visitor = builder.translator
    visitor.result_stack.clear()
    return visitor


@pytest.fixture
def llvm_visitor_in_function() -> LLVMVisitor:
    """
    title: Return a fresh llvmliteir visitor inside a live basic block.
    summary: >-
      Some lowering helpers require a valid insertion point. This fixture
      creates a dummy function and positions the IRBuilder at its entry block.
    returns:
      type: LLVMVisitor
    """
    builder = LLVMBuilder()
    visitor = builder.translator
    visitor.result_stack.clear()
    fn_type = ir.FunctionType(visitor._llvm.VOID_TYPE, [])
    function = ir.Function(visitor._llvm.module, fn_type, name="_test_dummy")
    block = function.append_basic_block("entry")
    visitor._llvm.ir_builder = ir.IRBuilder(block)
    return visitor

"""
title: Context-manager statement tests.
"""

from __future__ import annotations

import re

from typing import cast

import pytest

from irx import astx
from irx.analysis import SemanticError, SemanticInfo, analyze
from irx.builder import Builder as LLVMBuilder
from irx.builder.base import Builder

from tests.conftest import assert_ir_parses, assert_jit_int_main_result

ENTER_VALUE = 7
ENTER_STATE = 1
EXIT_STATE = 2


def _semantic(node: astx.AST) -> SemanticInfo:
    """
    title: Return semantic sidecar information for a node.
    parameters:
      node:
        type: astx.AST
    returns:
      type: SemanticInfo
    """
    return cast(SemanticInfo, getattr(node, "semantic"))


def _class_type(name: str) -> astx.ClassType:
    """
    title: Build one named class type reference.
    parameters:
      name:
        type: str
    returns:
      type: astx.ClassType
    """
    return astx.ClassType(name)


def _attribute(
    name: str,
    type_: astx.DataType,
    *,
    value: astx.AST | None = None,
    is_static: bool = False,
) -> astx.VariableDeclaration:
    """
    title: Build one class attribute declaration.
    parameters:
      name:
        type: str
      type_:
        type: astx.DataType
      value:
        type: astx.AST | None
      is_static:
        type: bool
    returns:
      type: astx.VariableDeclaration
    """
    declaration = astx.VariableDeclaration(
        name=name,
        type_=type_,
        mutability=astx.MutabilityKind.mutable,
        scope=(astx.ScopeKind.global_ if is_static else astx.ScopeKind.local),
        value=value if value is not None else astx.Undefined(),
    )
    if is_static:
        declaration.is_static = True
    return declaration


def _method(
    name: str,
    return_type: astx.DataType,
    *body_nodes: astx.AST,
) -> astx.FunctionDef:
    """
    title: Build one instance method from body nodes.
    parameters:
      name:
        type: str
      return_type:
        type: astx.DataType
      body_nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.FunctionDef
    """
    body = astx.Block()
    for node in body_nodes:
        body.append(node)
    return astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name,
            args=astx.Arguments(),
            return_type=return_type,
        ),
        body=body,
    )


def _main(*body_nodes: astx.AST) -> astx.FunctionDef:
    """
    title: Build an Int32 main function.
    parameters:
      body_nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.FunctionDef
    """
    body = astx.Block()
    for node in body_nodes:
        body.append(node)
    return astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=body,
    )


def _module(*nodes: astx.AST) -> astx.Module:
    """
    title: Build one named module.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    module = astx.Module(name="main")
    for node in nodes:
        module.block.append(node)
    return module


def _manager_class(*methods: astx.FunctionDef) -> astx.ClassDefStmt:
    """
    title: Build a context-manager class declaration.
    parameters:
      methods:
        type: astx.FunctionDef
        variadic: positional
    returns:
      type: astx.ClassDefStmt
    """
    return astx.ClassDefStmt(
        name="Manager",
        attributes=[
            _attribute(
                "state",
                astx.Int32(),
                is_static=True,
                value=astx.LiteralInt32(0),
            )
        ],
        methods=list(methods),
    )


def _enter_method() -> astx.FunctionDef:
    """
    title: Build a context __enter__ method.
    returns:
      type: astx.FunctionDef
    """
    return _method(
        "__enter__",
        astx.Int32(),
        astx.BinaryOp(
            "=",
            astx.StaticFieldAccess("Manager", "state"),
            astx.LiteralInt32(ENTER_STATE),
        ),
        astx.FunctionReturn(astx.LiteralInt32(ENTER_VALUE)),
    )


def _exit_method() -> astx.FunctionDef:
    """
    title: Build a context __exit__ method.
    returns:
      type: astx.FunctionDef
    """
    return _method(
        "__exit__",
        astx.NoneType(),
        astx.BinaryOp(
            "=",
            astx.StaticFieldAccess("Manager", "state"),
            astx.LiteralInt32(EXIT_STATE),
        ),
    )


def _exit_with_parameter_method() -> astx.FunctionDef:
    """
    title: Build an invalid context __exit__ method.
    returns:
      type: astx.FunctionDef
    """
    return astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "__exit__",
            args=astx.Arguments(astx.Argument("code", astx.Int32())),
            return_type=astx.NoneType(),
        ),
        body=astx.Block(),
    )


def _main_ir(ir_text: str) -> str:
    """
    title: Return the emitted LLVM body for main.
    parameters:
      ir_text:
        type: str
    returns:
      type: str
    """
    start = ir_text.index('define i32 @"main"(')
    end = ir_text.find("\ndefine ", start + 1)
    if end < 0:
        return ir_text[start:]
    return ir_text[start:end]


def test_analyze_with_resolves_context_protocol_and_target() -> None:
    """
    title: With statements resolve enter, exit, and target metadata.
    """
    body = astx.Block()
    body.append(astx.FunctionReturn(astx.Identifier("value")))
    with_stmt = astx.WithStmt(
        astx.ClassConstruct("Manager"),
        body,
        target=astx.Identifier("value"),
    )
    module = _module(
        _manager_class(_enter_method(), _exit_method()),
        _main(with_stmt),
    )

    analyze(module)

    resolved = _semantic(with_stmt).resolved_context_manager
    assert resolved is not None
    assert resolved.enter.member.name == "__enter__"
    assert resolved.exit.member.name == "__exit__"
    assert resolved.target_symbol is not None
    assert isinstance(resolved.target_symbol.type_, astx.Int32)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_with_statement_calls_exit_on_normal_fallthrough(
    builder_class: type[Builder],
) -> None:
    """
    title: Normal with fallthrough should run __exit__ before following code.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    body = astx.Block()
    with_stmt = astx.WithStmt(astx.ClassConstruct("Manager"), body)
    module = _module(
        _manager_class(_enter_method(), _exit_method()),
        _main(
            with_stmt,
            astx.FunctionReturn(astx.StaticFieldAccess("Manager", "state")),
        ),
    )

    ir_text = builder.translate(module)

    assert "__enter__" in ir_text
    assert "__exit__" in ir_text
    assert_ir_parses(ir_text)
    assert_jit_int_main_result(builder, module, EXIT_STATE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_with_statement_binds_enter_result_to_target(
    builder_class: type[Builder],
) -> None:
    """
    title: With targets should expose the __enter__ result inside the body.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    body = astx.Block()
    body.append(astx.FunctionReturn(astx.Identifier("value")))
    with_stmt = astx.WithStmt(
        astx.ClassConstruct("Manager"),
        body,
        target=astx.Identifier("value"),
    )
    module = _module(
        _manager_class(_enter_method(), _exit_method()),
        _main(with_stmt),
    )

    assert_jit_int_main_result(builder, module, ENTER_VALUE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_break_inside_with_calls_exit(
    builder_class: type[Builder],
) -> None:
    """
    title: Break inside a with body should run __exit__.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    with_body = astx.Block()
    with_body.append(astx.BreakStmt())
    loop_body = astx.Block()
    loop_body.append(astx.WithStmt(astx.ClassConstruct("Manager"), with_body))
    loop = astx.WhileStmt(astx.LiteralBoolean(True), loop_body)
    module = _module(
        _manager_class(_enter_method(), _exit_method()),
        _main(
            loop,
            astx.FunctionReturn(astx.StaticFieldAccess("Manager", "state")),
        ),
    )

    assert_jit_int_main_result(builder, module, EXIT_STATE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_continue_inside_with_calls_exit(
    builder_class: type[Builder],
) -> None:
    """
    title: Continue inside a with body should run __exit__.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    with_body = astx.Block()
    with_body.append(astx.ContinueStmt())
    loop_body = astx.Block()
    loop_body.append(astx.WithStmt(astx.ClassConstruct("Manager"), with_body))
    loop = astx.ForCountLoopStmt(
        initializer=astx.InlineVariableDeclaration(
            "i",
            type_=astx.Int32(),
            value=astx.LiteralInt32(0),
            mutability=astx.MutabilityKind.mutable,
        ),
        condition=astx.BinaryOp(
            "<",
            astx.Identifier("i"),
            astx.LiteralInt32(1),
        ),
        update=astx.UnaryOp("++", astx.Identifier("i")),
        body=loop_body,
    )
    module = _module(
        _manager_class(_enter_method(), _exit_method()),
        _main(
            loop,
            astx.FunctionReturn(astx.StaticFieldAccess("Manager", "state")),
        ),
    )

    assert_jit_int_main_result(builder, module, EXIT_STATE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_break_inside_loop_keeps_outer_with_active(
    builder_class: type[Builder],
) -> None:
    """
    title: Break should only clean up with scopes nested inside its loop.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    loop_body = astx.Block()
    loop_body.append(astx.BreakStmt())
    with_body = astx.Block()
    with_body.append(astx.WhileStmt(astx.LiteralBoolean(True), loop_body))
    with_body.append(
        astx.FunctionReturn(astx.StaticFieldAccess("Manager", "state"))
    )
    with_stmt = astx.WithStmt(astx.ClassConstruct("Manager"), with_body)
    module = _module(
        _manager_class(_enter_method(), _exit_method()),
        _main(with_stmt),
    )

    assert_jit_int_main_result(builder, module, ENTER_STATE)


@pytest.mark.parametrize("builder_class", [LLVMBuilder])
def test_return_inside_with_emits_exit_before_ret(
    builder_class: type[Builder],
) -> None:
    """
    title: Return inside a with body should emit __exit__ before ret.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    with_body = astx.Block()
    with_body.append(
        astx.FunctionReturn(astx.StaticFieldAccess("Manager", "state"))
    )
    with_stmt = astx.WithStmt(astx.ClassConstruct("Manager"), with_body)
    module = _module(
        _manager_class(_enter_method(), _exit_method()),
        _main(with_stmt),
    )

    main_ir = _main_ir(builder.translate(module))
    ret_index = main_ir.index("ret i32")
    exit_before_return = main_ir[:ret_index]

    assert re.search(r"call void .*__exit__", exit_before_return) is not None


def test_with_target_is_scoped_to_body() -> None:
    """
    title: With targets should not be visible after the body.
    """
    body = astx.Block()
    with_stmt = astx.WithStmt(
        astx.ClassConstruct("Manager"),
        body,
        target=astx.Identifier("value"),
    )
    module = _module(
        _manager_class(_enter_method(), _exit_method()),
        _main(with_stmt, astx.FunctionReturn(astx.Identifier("value"))),
    )

    with pytest.raises(SemanticError, match="cannot resolve name 'value'"):
        analyze(module)


def test_analyze_with_rejects_exit_parameters() -> None:
    """
    title: Context __exit__ should currently be zero-argument.
    """
    body = astx.Block()
    with_stmt = astx.WithStmt(astx.ClassConstruct("Manager"), body)
    module = _module(
        _manager_class(_enter_method(), _exit_with_parameter_method()),
        _main(with_stmt, astx.FunctionReturn(astx.LiteralInt32(0))),
    )

    with pytest.raises(SemanticError, match="expects 1 arguments but got 0"):
        analyze(module)


def test_analyze_with_requires_exit_method() -> None:
    """
    title: With statements require an __exit__ method.
    """
    body = astx.Block()
    with_stmt = astx.WithStmt(astx.ClassConstruct("Manager"), body)
    module = _module(_manager_class(_enter_method()), _main(with_stmt))

    with pytest.raises(SemanticError, match="has no method '__exit__'"):
        analyze(module)


def test_analyze_with_requires_class_manager() -> None:
    """
    title: With manager expressions must be class values.
    """
    body = astx.Block()
    with_stmt = astx.WithStmt(astx.LiteralInt32(1), body)
    module = _module(
        _main(with_stmt, astx.FunctionReturn(astx.LiteralInt32(0)))
    )

    with pytest.raises(SemanticError, match="with manager requires"):
        analyze(module)

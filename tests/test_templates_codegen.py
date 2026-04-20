"""
title: Tests for template-specialization LLVM lowering.
"""

from __future__ import annotations

from irx import astx
from irx.analysis import analyze_modules
from irx.analysis.module_symbols import mangle_function_name
from irx.builder import Builder

from tests.conftest import (
    StaticImportResolver,
    assert_ir_parses,
    make_parsed_module,
    translate_modules_ir,
)


def _number_bound() -> astx.UnionType:
    """
    title: Build the small Number template domain used in tests.
    returns:
      type: astx.UnionType
    """
    return astx.UnionType(
        (astx.Int32(), astx.Float64()),
        alias_name="Number",
    )


def _template_var(name: str = "T") -> astx.TemplateTypeVar:
    """
    title: Build one Number-bounded template type variable.
    parameters:
      name:
        type: str
    returns:
      type: astx.TemplateTypeVar
    """
    return astx.TemplateTypeVar(name, bound=_number_bound())


def _template_param(name: str = "T") -> astx.TemplateParam:
    """
    title: Build one Number-bounded template parameter.
    parameters:
      name:
        type: str
    returns:
      type: astx.TemplateParam
    """
    return astx.TemplateParam(name, _number_bound())


def _templated_function(
    name: str,
    return_value: astx.AST,
    *args: astx.Argument,
) -> astx.FunctionDef:
    """
    title: Build one templated function definition.
    parameters:
      name:
        type: str
      return_value:
        type: astx.AST
      args:
        type: astx.Argument
        variadic: positional
    returns:
      type: astx.FunctionDef
    """
    prototype = astx.FunctionPrototype(
        name,
        args=astx.Arguments(*args),
        return_type=_template_var(),
    )
    astx.set_template_params(prototype, (_template_param(),))
    body = astx.Block()
    body.append(astx.FunctionReturn(return_value))
    return astx.FunctionDef(prototype=prototype, body=body)


def _templated_static_method(
    name: str,
    return_value: astx.AST,
    *args: astx.Argument,
) -> astx.FunctionDef:
    """
    title: Build one templated static method definition.
    parameters:
      name:
        type: str
      return_value:
        type: astx.AST
      args:
        type: astx.Argument
        variadic: positional
    returns:
      type: astx.FunctionDef
    """
    method = _templated_function(name, return_value, *args)
    method.prototype.is_static = True
    return method


def _int_function(name: str, *body_nodes: astx.AST) -> astx.FunctionDef:
    """
    title: Build a small Int32-returning function.
    parameters:
      name:
        type: str
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
            return_type=astx.Int32(),
        ),
        body=body,
    )


def test_translate_modules_lowers_template_function_specialization() -> None:
    """
    title: Template calls lower through concrete specialized functions.
    """
    add = _templated_function(
        "add",
        astx.BinaryOp("+", astx.Identifier("lhs"), astx.Identifier("rhs")),
        astx.Argument("lhs", _template_var()),
        astx.Argument("rhs", _template_var()),
    )
    call = astx.FunctionCall(
        "add",
        [astx.LiteralInt32(1), astx.LiteralInt32(2)],
    )
    root = make_parsed_module(
        "app.main",
        add,
        _int_function("main", astx.FunctionReturn(call)),
    )

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({}),
    )
    specialized_symbol = mangle_function_name("app.main", "add__Int32")
    generic_symbol = mangle_function_name("app.main", "add")

    assert_ir_parses(ir_text)
    assert f'define i32 @"{specialized_symbol}"(' in ir_text
    assert f'call i32 @"{specialized_symbol}"(' in ir_text
    assert f'define i32 @"{generic_symbol}"(' not in ir_text


def test_translate_modules_lowers_static_template_method_specialization() -> (
    None
):
    """
    title: Static template methods lower through specialized method symbols.
    """
    identity = _templated_static_method(
        "identity",
        astx.Identifier("value"),
        astx.Argument("value", _template_var()),
    )
    math = astx.ClassDefStmt(name="Math", methods=[identity])
    call = astx.StaticMethodCall("Math", "identity", [astx.LiteralInt32(5)])
    root = make_parsed_module(
        "app.main",
        math,
        _int_function("main", astx.FunctionReturn(call)),
    )

    analyze_modules(root, StaticImportResolver({}))
    resolved_method = getattr(call, "semantic").resolved_method_call
    assert resolved_method is not None
    specialized_base = resolved_method.function.signature.symbol_name
    generic_base = identity.semantic.resolved_function.signature.symbol_name

    ir_text = translate_modules_ir(
        Builder(),
        root,
        StaticImportResolver({}),
    )
    specialized_symbol = mangle_function_name("app.main", specialized_base)
    generic_symbol = mangle_function_name("app.main", generic_base)

    assert_ir_parses(ir_text)
    assert f'define i32 @"{specialized_symbol}"(' in ir_text
    assert f'call i32 @"{specialized_symbol}"(' in ir_text
    assert f'define i32 @"{generic_symbol}"(' not in ir_text

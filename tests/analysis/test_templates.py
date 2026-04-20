"""
title: Tests for template-function semantic analysis.
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.analysis import MethodDispatchKind, SemanticError, analyze
from irx.analysis.resolved_nodes import SemanticInfo

from tests.conftest import make_module


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


def _templated_function(
    name: str,
    return_value: astx.AST,
    *args: astx.Argument,
    return_type: astx.DataType | None = None,
    template_params: tuple[astx.TemplateParam, ...] | None = None,
) -> astx.FunctionDef:
    """
    title: Build one templated function definition.
    parameters:
      name:
        type: str
      return_value:
        type: astx.AST
      return_type:
        type: astx.DataType | None
      template_params:
        type: tuple[astx.TemplateParam, Ellipsis] | None
      args:
        type: astx.Argument
        variadic: positional
    returns:
      type: astx.FunctionDef
    """
    prototype = astx.FunctionPrototype(
        name,
        args=astx.Arguments(*args),
        return_type=return_type or _template_var(),
    )
    astx.set_template_params(
        prototype, template_params or (_template_param(),)
    )
    body = astx.Block()
    body.append(astx.FunctionReturn(return_value))
    return astx.FunctionDef(prototype=prototype, body=body)


def _templated_method(
    name: str,
    return_value: astx.AST,
    *args: astx.Argument,
    is_static: bool = False,
) -> astx.FunctionDef:
    """
    title: Build one templated class method definition.
    parameters:
      name:
        type: str
      return_value:
        type: astx.AST
      is_static:
        type: bool
      args:
        type: astx.Argument
        variadic: positional
    returns:
      type: astx.FunctionDef
    """
    method = _templated_function(name, return_value, *args)
    if is_static:
        method.prototype.is_static = True
    return method


def _main_returning(value: astx.AST) -> astx.FunctionDef:
    """
    title: Build a simple Int32-returning main function.
    parameters:
      value:
        type: astx.AST
    returns:
      type: astx.FunctionDef
    """
    body = astx.Block()
    body.append(astx.FunctionReturn(value))
    return astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=body,
    )


def test_analyze_template_function_specializes_inferred_call() -> None:
    """
    title: Inferred template calls bind to one concrete specialization.
    """
    template_fn = _templated_function(
        "add",
        astx.BinaryOp("+", astx.Identifier("lhs"), astx.Identifier("rhs")),
        astx.Argument("lhs", _template_var()),
        astx.Argument("rhs", _template_var()),
    )
    call = astx.FunctionCall(
        "add",
        [astx.LiteralInt32(1), astx.LiteralInt32(2)],
    )
    module = make_module(
        "app.main",
        template_fn,
        _main_returning(call),
    )

    analyze(module)

    resolved_call = _semantic(call).resolved_call
    resolved_function = _semantic(call).resolved_function
    generated_names = {
        node.name
        for node in astx.generated_template_nodes(module)
        if isinstance(node, astx.FunctionDef)
    }

    assert resolved_call is not None
    assert resolved_function is not None
    assert resolved_function.name == "add__Int32"
    assert isinstance(resolved_call.result_type, astx.Int32)
    assert generated_names == {"add__Int32", "add__Float64"}


def test_analyze_template_function_uses_explicit_template_args() -> None:
    """
    title: Explicit template arguments select the requested specialization.
    """
    template_fn = _templated_function(
        "identity",
        astx.Identifier("value"),
        astx.Argument("value", _template_var()),
    )
    call = astx.FunctionCall("identity", [astx.LiteralInt32(7)])
    astx.set_template_args(call, (astx.Float64(),))
    body = astx.Block()
    body.append(call)
    body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=body,
    )
    module = make_module("app.main", template_fn, main)

    analyze(module)

    resolved_call = _semantic(call).resolved_call
    resolved_function = _semantic(call).resolved_function

    assert resolved_call is not None
    assert resolved_function is not None
    assert resolved_function.name == "identity__Float64"
    assert isinstance(resolved_call.result_type, astx.Float64)
    assert isinstance(resolved_call.resolved_argument_types[0], astx.Float64)


def test_analyze_template_function_rejects_out_of_bound_explicit_arg() -> None:
    """
    title: Explicit template arguments must satisfy their finite bound.
    """
    template_fn = _templated_function(
        "identity",
        astx.Identifier("value"),
        astx.Argument("value", _template_var()),
    )
    call = astx.FunctionCall("identity", [astx.LiteralString("bad")])
    astx.set_template_args(call, (astx.String(),))
    body = astx.Block()
    body.append(call)
    body.append(astx.FunctionReturn(astx.LiteralInt32(0)))
    main = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=body,
    )
    module = make_module("app.main", template_fn, main)

    with pytest.raises(SemanticError) as excinfo:
        analyze(module)

    assert "does not satisfy bound 'Number'" in str(excinfo.value)


def test_analyze_template_definition_reports_failing_substitution() -> None:
    """
    title: Template validation reports the substitution that fails.
    """
    template_fn = _templated_function(
        "bit_and",
        astx.BinaryOp("&", astx.Identifier("lhs"), astx.Identifier("rhs")),
        astx.Argument("lhs", _template_var()),
        astx.Argument("rhs", _template_var()),
    )
    module = make_module(
        "app.main",
        template_fn,
        _main_returning(astx.LiteralInt32(0)),
    )

    with pytest.raises(SemanticError) as excinfo:
        analyze(module)

    message = str(excinfo.value)
    assert "Template function 'bit_and' is invalid for T = Float64" in message
    assert "Invalid operator '&' for operand types" in message


def test_analyze_template_instance_method_uses_direct_specialization() -> None:
    """
    title: Instance template methods lower through direct specializations.
    """
    echo = _templated_method(
        "echo",
        astx.Identifier("value"),
        astx.Argument("value", _template_var()),
    )
    box = astx.ClassDefStmt(name="Box", methods=[echo])
    call = astx.MethodCall(
        astx.Identifier("box"),
        "echo",
        [astx.LiteralInt32(7)],
    )
    probe_body = astx.Block()
    probe_body.append(astx.FunctionReturn(call))
    probe = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name="probe",
            args=astx.Arguments(
                astx.Argument("box", astx.ClassType("Box")),
            ),
            return_type=astx.Int32(),
        ),
        body=probe_body,
    )
    module = make_module(
        "app.main",
        box,
        probe,
        _main_returning(astx.LiteralInt32(0)),
    )

    analyze(module)

    resolved_method = _semantic(call).resolved_method_call
    generated_names = {
        node.name
        for node in astx.generated_template_nodes(module)
        if isinstance(node, astx.FunctionDef)
    }

    assert resolved_method is not None
    assert resolved_method.dispatch_kind is MethodDispatchKind.DIRECT
    assert resolved_method.function.name == "echo__Int32"
    assert "echo__Int32" in generated_names


def test_analyze_static_template_method_specializes_call() -> None:
    """
    title: Static template methods specialize like top-level template funcs.
    """
    identity = _templated_method(
        "identity",
        astx.Identifier("value"),
        astx.Argument("value", _template_var()),
        is_static=True,
    )
    math = astx.ClassDefStmt(name="Math", methods=[identity])
    call = astx.StaticMethodCall("Math", "identity", [astx.LiteralInt32(11)])
    module = make_module("app.main", math, _main_returning(call))

    analyze(module)

    resolved_method = _semantic(call).resolved_method_call

    assert resolved_method is not None
    assert resolved_method.function.name == "identity__Int32"
    assert resolved_method.dispatch_kind is MethodDispatchKind.DIRECT

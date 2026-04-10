"""
title: Tests for hardened function signatures and call semantics.
"""

from __future__ import annotations

from typing import cast

import pytest

from irx import astx
from irx.analysis import SemanticError, analyze
from irx.analysis.resolved_nodes import (
    CallingConvention,
    SemanticInfo,
)


def _semantic(node: astx.AST) -> SemanticInfo:
    """
    title: Return one node's attached semantic info.
    parameters:
      node:
        type: astx.AST
    returns:
      type: SemanticInfo
    """
    return cast(SemanticInfo, getattr(node, "semantic"))


def _block(*nodes: astx.AST) -> astx.Block:
    """
    title: Build one AST block from the provided nodes.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Block
    """
    block = astx.Block()
    for node in nodes:
        block.append(node)
    return block


def _main_module(*nodes: astx.AST) -> astx.Module:
    """
    title: Build one small Int32 main module.
    parameters:
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.Module
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "main",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(*nodes),
        )
    )
    return module


def _int_function(name: str, *nodes: astx.AST) -> astx.FunctionDef:
    """
    title: Build one Int32-returning helper function.
    parameters:
      name:
        type: str
      nodes:
        type: astx.AST
        variadic: positional
    returns:
      type: astx.FunctionDef
    """
    return astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            name,
            args=astx.Arguments(),
            return_type=astx.Int32(),
        ),
        body=_block(*nodes),
    )


def test_analyze_attaches_canonical_function_signature() -> None:
    """
    title: >-
      Function declarations should expose one canonical semantic signature.
    """
    helper = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(astx.Argument("value", astx.Int32())),
            return_type=astx.Int32(),
        ),
        body=_block(astx.FunctionReturn(astx.Identifier("value"))),
    )
    module = astx.Module()
    module.block.append(helper)
    analyze(module)

    resolved_function = _semantic(helper).resolved_function

    assert resolved_function is not None
    assert resolved_function.signature.name == "helper"
    assert resolved_function.signature.calling_convention is (
        CallingConvention.IRX_DEFAULT
    )
    assert resolved_function.signature.is_extern is False
    assert resolved_function.signature.is_variadic is False
    assert resolved_function.signature.symbol_name == "helper"
    assert len(resolved_function.signature.parameters) == 1
    assert resolved_function.signature.parameters[0].name == "value"
    assert isinstance(
        resolved_function.signature.parameters[0].type_,
        astx.Int32,
    )


def test_analyze_accepts_repeated_identical_prototypes() -> None:
    """
    title: Repeated identical function declarations should be accepted.
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(astx.Argument("value", astx.Int32())),
            return_type=astx.Int32(),
        )
    )
    module.block.append(
        astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(astx.Argument("value", astx.Int32())),
            return_type=astx.Int32(),
        )
    )

    analyze(module)


def test_analyze_rejects_duplicate_parameter_names() -> None:
    """
    title: Duplicate parameter names should fail semantic validation.
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "helper",
                args=astx.Arguments(
                    astx.Argument("value", astx.Int32()),
                    astx.Argument("value", astx.Int32()),
                ),
                return_type=astx.Int32(),
            ),
            body=_block(astx.FunctionReturn(astx.LiteralInt32(0))),
        )
    )

    with pytest.raises(SemanticError, match="repeats parameter 'value'"):
        analyze(module)


def test_analyze_rejects_conflicting_function_declarations() -> None:
    """
    title: Conflicting declarations should be rejected before lowering.
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(astx.Argument("value", astx.Int32())),
            return_type=astx.Int32(),
        )
    )
    module.block.append(
        astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(astx.Argument("value", astx.Int64())),
            return_type=astx.Int32(),
        )
    )

    with pytest.raises(
        SemanticError,
        match="Conflicting declaration for function 'helper'",
    ):
        analyze(module)


def test_analyze_rejects_invalid_main_signature() -> None:
    """
    title: Main must use the hardened deterministic Int32 signature.
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "main",
                args=astx.Arguments(),
                return_type=astx.Boolean(),
            ),
            body=_block(astx.FunctionReturn(astx.LiteralBoolean(True))),
        )
    )

    with pytest.raises(
        SemanticError, match="Function 'main' must return Int32"
    ):
        analyze(module)


def test_analyze_rejects_invalid_calling_convention() -> None:
    """
    title: Unknown calling conventions should fail semantically.
    """
    prototype = astx.FunctionPrototype(
        "helper",
        args=astx.Arguments(),
        return_type=astx.Int32(),
    )
    prototype.calling_convention = "bogus"
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=prototype,
            body=_block(astx.FunctionReturn(astx.LiteralInt32(0))),
        )
    )

    with pytest.raises(
        SemanticError,
        match="invalid calling convention",
    ):
        analyze(module)


def test_analyze_rejects_variadic_irx_defined_function() -> None:
    """
    title: >-
      Variadic support should stay limited to explicit extern declarations.
    """
    prototype = astx.FunctionPrototype(
        "helper",
        args=astx.Arguments(astx.Argument("value", astx.Int32())),
        return_type=astx.Int32(),
    )
    prototype.is_variadic = True
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=prototype,
            body=_block(astx.FunctionReturn(astx.Identifier("value"))),
        )
    )

    with pytest.raises(
        SemanticError,
        match="may be variadic only when declared extern",
    ):
        analyze(module)


def test_analyze_accepts_extern_c_variadic_signature() -> None:
    """
    title: Extern varargs should be accepted only on the narrow C path.
    """
    prototype = astx.FunctionPrototype(
        "printf",
        args=astx.Arguments(astx.Argument("format", astx.UTF8String())),
        return_type=astx.Int32(),
    )
    prototype.is_extern = True
    prototype.is_variadic = True
    prototype.calling_convention = "c"
    prototype.symbol_name = "printf"

    module = astx.Module()
    module.block.append(prototype)

    analyze(module)

    resolved_function = _semantic(prototype).resolved_function

    assert resolved_function is not None
    assert resolved_function.signature.is_extern is True
    assert resolved_function.signature.is_variadic is True
    assert (
        resolved_function.signature.calling_convention is CallingConvention.C
    )
    assert resolved_function.signature.symbol_name == "printf"


def test_analyze_attaches_call_resolution_with_implicit_widening() -> None:
    """
    title: Calls should carry resolved argument conversions from semantics.
    """
    helper = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(astx.Argument("value", astx.Int32())),
            return_type=astx.Int32(),
        ),
        body=_block(astx.FunctionReturn(astx.Identifier("value"))),
    )
    call = astx.FunctionCall("helper", [astx.LiteralInt16(7)])
    module = _main_module(astx.FunctionReturn(call))
    module.block.insert(0, helper)

    analyze(module)

    resolution = _semantic(call).resolved_call

    assert resolution is not None
    assert isinstance(resolution.result_type, astx.Int32)
    assert isinstance(resolution.resolved_argument_types[0], astx.Int32)
    assert resolution.implicit_conversions[0] is not None
    assert isinstance(
        resolution.implicit_conversions[0].source_type,
        astx.Int16,
    )
    assert isinstance(
        resolution.implicit_conversions[0].target_type,
        astx.Int32,
    )


def test_analyze_rejects_too_many_call_arguments() -> None:
    """
    title: Fixed-arity calls should reject extra arguments.
    """
    helper = _int_function("helper", astx.FunctionReturn(astx.LiteralInt32(1)))
    call = astx.FunctionCall(
        "helper",
        [astx.LiteralInt32(1)],
    )
    module = _main_module(call, astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.insert(0, helper)

    with pytest.raises(SemanticError, match="Incorrect # arguments passed"):
        analyze(module)


def test_analyze_rejects_implicit_call_narrowing() -> None:
    """
    title: Calls should reject narrowing implicit argument conversions.
    """
    helper = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(astx.Argument("value", astx.Int16())),
            return_type=astx.Int32(),
        ),
        body=_block(astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    module = _main_module(
        astx.FunctionCall("helper", [astx.LiteralInt32(7)]),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )
    module.block.insert(0, helper)

    with pytest.raises(
        SemanticError,
        match="Argument 0 for 'helper' has incompatible type",
    ):
        analyze(module)


def test_analyze_rejects_sign_changing_call_cast() -> None:
    """
    title: Calls should reject unsafe sign-changing implicit casts.
    """
    helper = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(astx.Argument("value", astx.UInt32())),
            return_type=astx.Int32(),
        ),
        body=_block(astx.FunctionReturn(astx.LiteralInt32(0))),
    )
    module = _main_module(
        astx.FunctionCall("helper", [astx.LiteralInt32(7)]),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )
    module.block.insert(0, helper)

    with pytest.raises(
        SemanticError,
        match="Argument 0 for 'helper' has incompatible type",
    ):
        analyze(module)


def test_void_call_as_statement_is_allowed() -> None:
    """
    title: Void calls may still appear as standalone statements.
    """
    noop = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "noop",
            args=astx.Arguments(),
            return_type=astx.NoneType(),
        ),
        body=_block(astx.FunctionReturn(astx.LiteralNone())),
    )
    call = astx.FunctionCall("noop", [])
    module = _main_module(call, astx.FunctionReturn(astx.LiteralInt32(0)))
    module.block.insert(0, noop)

    analyze(module)

    assert isinstance(_semantic(call).resolved_type, astx.NoneType)
    assert _semantic(call).resolved_call is not None


def test_void_call_as_value_is_rejected() -> None:
    """
    title: Void calls must not be consumed as value expressions.
    """
    noop = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "noop",
            args=astx.Arguments(),
            return_type=astx.NoneType(),
        ),
        body=_block(astx.FunctionReturn(astx.LiteralNone())),
    )
    module = _main_module(
        astx.VariableDeclaration(
            name="value",
            type_=astx.Int32(),
            value=astx.FunctionCall("noop", []),
        ),
        astx.FunctionReturn(astx.LiteralInt32(0)),
    )
    module.block.insert(0, noop)

    with pytest.raises(
        SemanticError,
        match="cannot use the result of void call as a value",
    ):
        analyze(module)


def test_void_function_return_value_is_rejected() -> None:
    """
    title: Void functions must not return a value.
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "helper",
                args=astx.Arguments(),
                return_type=astx.NoneType(),
            ),
            body=_block(astx.FunctionReturn(astx.LiteralInt32(1))),
        )
    )

    with pytest.raises(
        SemanticError,
        match="cannot return a value",
    ):
        analyze(module)


def test_nonvoid_function_bare_return_is_rejected() -> None:
    """
    title: Non-void functions must return a value-bearing expression.
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "helper",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(astx.FunctionReturn(astx.LiteralNone())),
        )
    )

    with pytest.raises(
        SemanticError,
        match="must return a value",
    ):
        analyze(module)


def test_nonvoid_function_fallthrough_is_rejected() -> None:
    """
    title: Non-void functions must not fall through.
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                "helper",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=_block(
                astx.IfStmt(
                    condition=astx.LiteralBoolean(True),
                    then=_block(astx.FunctionReturn(astx.LiteralInt32(1))),
                    else_=None,
                )
            ),
        )
    )

    with pytest.raises(SemanticError, match="missing a return statement"):
        analyze(module)


def test_nonvoid_function_both_branches_return_is_allowed() -> None:
    """
    title: Structured analysis should accept both-returning if branches.
    """
    helper = astx.FunctionDef(
        prototype=astx.FunctionPrototype(
            "helper",
            args=astx.Arguments(astx.Argument("flag", astx.Boolean())),
            return_type=astx.Int32(),
        ),
        body=_block(
            astx.IfStmt(
                condition=astx.Identifier("flag"),
                then=_block(astx.FunctionReturn(astx.LiteralInt32(1))),
                else_=_block(astx.FunctionReturn(astx.LiteralInt32(2))),
            )
        ),
    )
    module = astx.Module()
    module.block.append(helper)

    analyze(module)


def test_main_requires_definition() -> None:
    """
    title: Main declarations must resolve to a definition.
    """
    module = astx.Module()
    module.block.append(
        astx.FunctionPrototype(
            "main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        )
    )

    with pytest.raises(
        SemanticError,
        match="Function 'main' must have a definition",
    ):
        analyze(module)

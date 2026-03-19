"""
title: Test FunctionDef with void return type
"""
from __future__ import annotations

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
class TestVoidFunctionNoReturn:
    def test_void_function_no_return_compiles(
        self, builder_class: type[Builder]
    ) -> None:
        """
        title: Void function with no return statement must compile cleanly.
        """
        builder = builder_class()
        module = builder.module()

        proto = astx.FunctionPrototype(
            name="do_nothing",
            args=astx.Arguments(),
            return_type=astx.NoneType(),
        )
        body = astx.Block()

        void_fn = astx.FunctionDef(prototype=proto, body=body)
        module.block.append(void_fn)

        main_proto = astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        )
        main_body = astx.Block()
        main_body.append(astx.FunctionCall(void_fn, []))
        main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))

        main_fn = astx.FunctionDef(prototype=main_proto, body=main_body)
        module.block.append(main_fn)

        check_result("build", builder, module)

    def test_void_function_with_print_no_explicit_return(
        self, builder_class: type[Builder]
    ) -> None:
        """
        title: Void function with statements but no return node must compile cleanly.
        """
        builder = builder_class()
        module = builder.module()

        proto = astx.FunctionPrototype(
            name="greet",
            args=astx.Arguments(),
            return_type=astx.NoneType(),
        )
        body = astx.Block()
        body.append(PrintExpr(astx.LiteralUTF8String("hello")))

        greet_fn = astx.FunctionDef(prototype=proto, body=body)
        module.block.append(greet_fn)

        main_proto = astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        )
        main_body = astx.Block()
        main_body.append(astx.FunctionCall(greet_fn, []))
        main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))

        main_fn = astx.FunctionDef(prototype=main_proto, body=main_body)
        module.block.append(main_fn)

        check_result("build", builder, module, expected_output="hello")


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
class TestVoidFunctionWithEarlyReturn:
    def test_void_function_with_explicit_return_no_duplicate_terminator(
        self, builder_class: type[Builder]
    ) -> None:
        """
        title: Void function with explicit return must not get a duplicate terminator.
        """
        builder = builder_class()
        module = builder.module()

        proto = astx.FunctionPrototype(
            name="explicit_void",
            args=astx.Arguments(),
            return_type=astx.NoneType(),
        )
        body = astx.Block()
        body.append(PrintExpr(astx.LiteralUTF8String("explicit")))
        body.append(astx.FunctionReturn(astx.LiteralNone()))

        void_fn = astx.FunctionDef(prototype=proto, body=body)
        module.block.append(void_fn)

        main_proto = astx.FunctionPrototype(
            name="main",
            args=astx.Arguments(),
            return_type=astx.Int32(),
        )
        main_body = astx.Block()
        main_body.append(astx.FunctionCall(void_fn, []))
        main_body.append(astx.FunctionReturn(astx.LiteralInt32(0)))

        main_fn = astx.FunctionDef(prototype=main_proto, body=main_body)
        module.block.append(main_fn)

        check_result("build", builder, module, expected_output="explicit")


@pytest.mark.parametrize(
    "int_type, literal_type",
    [
        (astx.Int32, astx.LiteralInt32),
        (astx.Int64, astx.LiteralInt64),
    ],
)
@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMLiteIR,
    ],
)
def test_non_void_function_missing_return_gets_zero_fallback(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Non-void function with no return must not produce malformed IR.
    """
    builder = builder_class()
    module = builder.module()

    proto = astx.FunctionPrototype(
        name="returns_nothing",
        args=astx.Arguments(),
        return_type=int_type(),
    )
    body = astx.Block()

    missing_return_fn = astx.FunctionDef(prototype=proto, body=body)
    module.block.append(missing_return_fn)

    main_proto = astx.FunctionPrototype(
        name="main",
        args=astx.Arguments(),
        return_type=int_type(),
    )
    main_body = astx.Block()
    main_body.append(astx.FunctionCall(missing_return_fn, []))
    main_body.append(astx.FunctionReturn(literal_type(0)))

    main_fn = astx.FunctionDef(prototype=main_proto, body=main_body)
    module.block.append(main_fn)

    check_result("build", builder, module)

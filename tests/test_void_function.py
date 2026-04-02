"""
title: Test FunctionDef with void return type
"""

from __future__ import annotations

import pytest

from irx import astx
from irx.analysis import SemanticError
from irx.builders.base import Builder
from irx.builders.llvmliteir import Builder as LLVMBuilder
from irx.system import PrintExpr

from .conftest import check_result


@pytest.mark.parametrize(
    "builder_class",
    [
        LLVMBuilder,
    ],
)
class TestVoidFunctionNoReturn:
    def test_void_function_no_return_compiles(
        self, builder_class: type[Builder]
    ) -> None:
        """
        title: Void function with no return statement must compile cleanly.
        parameters:
          builder_class:
            type: type[Builder]
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

        ir_text = builder.translate(module)
        assert "ret void" in ir_text, (
            "Expected implicit 'ret void' in IR for void function "
            f"with no return statement, but got:\n{ir_text}"
        )

    def test_void_function_with_print_no_explicit_return(
        self, builder_class: type[Builder]
    ) -> None:
        """
        title: Void function with no return node compiles cleanly.
        parameters:
          builder_class:
            type: type[Builder]
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
        LLVMBuilder,
    ],
)
class TestVoidFunctionWithEarlyReturn:
    def test_void_function_with_explicit_return_no_duplicate_terminator(
        self, builder_class: type[Builder]
    ) -> None:
        """
        title: Void function with explicit return, no duplicate terminator.
        parameters:
          builder_class:
            type: type[Builder]
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
        LLVMBuilder,
    ],
)
def test_non_void_function_missing_return_raises_error(
    builder_class: type[Builder],
    int_type: type,
    literal_type: type,
) -> None:
    """
    title: Non-void function with no return must raise an error.
    parameters:
      builder_class:
        type: type[Builder]
      int_type:
        type: type
      literal_type:
        type: type
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

    with pytest.raises(SemanticError, match="missing a return statement"):
        builder.translate(module)

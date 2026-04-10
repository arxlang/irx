"""
title: Semantic registration helpers.
summary: >-
  Centralize semantic declaration and top-level registration policy so the
  analyzer delegates identity and duplicate handling to a smaller subsystem.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from public import public

from irx import astx
from irx.analysis.context import SemanticContext
from irx.analysis.factories import SemanticEntityFactory
from irx.analysis.module_interfaces import ModuleKey
from irx.analysis.resolved_nodes import (
    CallingConvention,
    FunctionSignature,
    ParameterSpec,
    SemanticFunction,
    SemanticStruct,
    SemanticSymbol,
)
from irx.analysis.types import same_type
from irx.typecheck import typechecked

MAIN_FUNCTION_NAME = "main"


@public
@typechecked
class SemanticRegistry:
    """
    title: Semantic entity registration policy.
    summary: >-
      Register locals, functions, and structs while enforcing the duplicate
      declaration rules that semantic analysis currently exposes.
    attributes:
      context:
        type: SemanticContext
      factory:
        type: SemanticEntityFactory
    """

    context: SemanticContext
    factory: SemanticEntityFactory

    def __init__(
        self,
        context: SemanticContext,
        factory: SemanticEntityFactory,
    ) -> None:
        """
        title: Initialize SemanticRegistry.
        parameters:
          context:
            type: SemanticContext
          factory:
            type: SemanticEntityFactory
        """
        self.context = context
        self.factory = factory

    def _current_module_key(self) -> ModuleKey:
        """
        title: Return the active module key.
        returns:
          type: ModuleKey
        """
        return self.context.current_module_key or "<root>"

    def _prototype_is_extern(
        self,
        prototype: astx.FunctionPrototype,
    ) -> bool:
        """
        title: Return whether one prototype is an explicit extern declaration.
        parameters:
          prototype:
            type: astx.FunctionPrototype
        returns:
          type: bool
        """
        return bool(getattr(prototype, "is_extern", False))

    def _prototype_is_variadic(
        self,
        prototype: astx.FunctionPrototype,
    ) -> bool:
        """
        title: Return whether one prototype is marked variadic.
        parameters:
          prototype:
            type: astx.FunctionPrototype
        returns:
          type: bool
        """
        return bool(getattr(prototype, "is_variadic", False))

    def _prototype_symbol_name(
        self,
        prototype: astx.FunctionPrototype,
    ) -> str:
        """
        title: Return the normalized semantic symbol name for one prototype.
        parameters:
          prototype:
            type: astx.FunctionPrototype
        returns:
          type: str
        """
        prototype_name = cast(str, prototype.name)
        raw_symbol_name = getattr(prototype, "symbol_name", prototype_name)
        if raw_symbol_name is None:
            return prototype_name
        if isinstance(raw_symbol_name, str) and raw_symbol_name.strip():
            return raw_symbol_name
        self.context.diagnostics.add(
            f"Function '{prototype_name}' has an invalid symbol_name",
            node=prototype,
        )
        return prototype_name

    def _prototype_calling_convention(
        self,
        prototype: astx.FunctionPrototype,
        *,
        is_extern: bool,
    ) -> CallingConvention:
        """
        title: Return the normalized calling convention for one prototype.
        parameters:
          prototype:
            type: astx.FunctionPrototype
          is_extern:
            type: bool
        returns:
          type: CallingConvention
        """
        default = (
            CallingConvention.C if is_extern else CallingConvention.IRX_DEFAULT
        )
        raw_value = getattr(prototype, "calling_convention", None)
        if isinstance(raw_value, CallingConvention):
            return raw_value
        if raw_value is None:
            return default
        if isinstance(raw_value, str):
            for convention in CallingConvention:
                if raw_value == convention.value:
                    return convention
        self.context.diagnostics.add(
            f"Function '{prototype.name}' uses an invalid calling convention",
            node=prototype,
        )
        return default

    def _same_parameter_spec(
        self,
        lhs: ParameterSpec,
        rhs: ParameterSpec,
    ) -> bool:
        """
        title: Return whether two parameter specs match exactly.
        parameters:
          lhs:
            type: ParameterSpec
          rhs:
            type: ParameterSpec
        returns:
          type: bool
        """
        return (
            lhs.name == rhs.name
            and lhs.passing_kind is rhs.passing_kind
            and same_type(lhs.type_, rhs.type_)
        )

    def signatures_match(
        self,
        lhs: FunctionSignature,
        rhs: FunctionSignature,
    ) -> bool:
        """
        title: Return whether two canonical function signatures match.
        parameters:
          lhs:
            type: FunctionSignature
          rhs:
            type: FunctionSignature
        returns:
          type: bool
        """
        if lhs.name != rhs.name:
            return False
        if lhs.calling_convention is not rhs.calling_convention:
            return False
        if lhs.is_variadic != rhs.is_variadic:
            return False
        if lhs.is_extern != rhs.is_extern:
            return False
        if lhs.symbol_name != rhs.symbol_name:
            return False
        if not same_type(lhs.return_type, rhs.return_type):
            return False
        if len(lhs.parameters) != len(rhs.parameters):
            return False
        return all(
            self._same_parameter_spec(lhs_param, rhs_param)
            for lhs_param, rhs_param in zip(lhs.parameters, rhs.parameters)
        )

    def normalize_function_signature(
        self,
        prototype: astx.FunctionPrototype,
        *,
        definition: astx.FunctionDef | None = None,
    ) -> FunctionSignature:
        """
        title: Normalize and validate one semantic function signature.
        parameters:
          prototype:
            type: astx.FunctionPrototype
          definition:
            type: astx.FunctionDef | None
        returns:
          type: FunctionSignature
        """
        seen_parameter_names: set[str] = set()
        for argument in prototype.args.nodes:
            if argument.name in seen_parameter_names:
                self.context.diagnostics.add(
                    f"Function '{prototype.name}' repeats parameter "
                    f"'{argument.name}'",
                    node=argument,
                )
                continue
            seen_parameter_names.add(argument.name)

        is_extern = self._prototype_is_extern(prototype)
        is_variadic = self._prototype_is_variadic(prototype)
        symbol_name = self._prototype_symbol_name(prototype)
        calling_convention = self._prototype_calling_convention(
            prototype,
            is_extern=is_extern,
        )

        if definition is not None and is_extern:
            self.context.diagnostics.add(
                f"Extern function '{prototype.name}' cannot define a body",
                node=definition,
            )
        if is_variadic and not is_extern:
            self.context.diagnostics.add(
                f"Function '{prototype.name}' may be variadic only when "
                "declared extern",
                node=prototype,
            )
        if is_extern and calling_convention is not CallingConvention.C:
            self.context.diagnostics.add(
                f"Extern function '{prototype.name}' must use calling "
                "convention 'c'",
                node=prototype,
            )
        if (
            not is_extern
            and calling_convention is not CallingConvention.IRX_DEFAULT
        ):
            self.context.diagnostics.add(
                f"IRx-defined function '{prototype.name}' must use calling "
                "convention 'irx_default'",
                node=prototype,
            )
        if not is_extern and symbol_name != prototype.name:
            self.context.diagnostics.add(
                f"Function '{prototype.name}' may override symbol_name only "
                "for extern declarations",
                node=prototype,
            )

        signature = self.factory.make_function_signature(
            prototype,
            calling_convention=calling_convention,
            is_variadic=is_variadic,
            is_extern=is_extern,
            symbol_name=symbol_name,
        )

        if signature.name == MAIN_FUNCTION_NAME:
            if signature.is_extern:
                self.context.diagnostics.add(
                    "Function 'main' cannot be extern",
                    node=prototype,
                )
            if signature.is_variadic:
                self.context.diagnostics.add(
                    "Function 'main' must not be variadic",
                    node=prototype,
                )
            if len(signature.parameters) != 0:
                self.context.diagnostics.add(
                    "Function 'main' must not declare parameters",
                    node=prototype,
                )
            if (
                signature.calling_convention
                is not CallingConvention.IRX_DEFAULT
            ):
                self.context.diagnostics.add(
                    "Function 'main' must use calling convention "
                    "'irx_default'",
                    node=prototype,
                )
            if not same_type(signature.return_type, astx.Int32()):
                self.context.diagnostics.add(
                    "Function 'main' must return Int32",
                    node=prototype,
                )

        return signature

    def declare_local(
        self,
        name: str,
        type_: astx.DataType,
        *,
        is_mutable: bool,
        declaration: astx.AST,
        kind: str = "variable",
    ) -> SemanticSymbol:
        """
        title: Declare one lexical symbol.
        parameters:
          name:
            type: str
          type_:
            type: astx.DataType
          is_mutable:
            type: bool
          declaration:
            type: astx.AST
          kind:
            type: str
        returns:
          type: SemanticSymbol
        """
        symbol = self.factory.make_variable_symbol(
            self._current_module_key(),
            name,
            type_,
            is_mutable=is_mutable,
            declaration=declaration,
            kind=kind,
        )
        if not self.context.scopes.declare(symbol):
            self.context.diagnostics.add(
                f"Identifier already declared: {name}",
                node=declaration,
            )
        return symbol

    def register_function(
        self,
        prototype: astx.FunctionPrototype,
        *,
        definition: astx.FunctionDef | None = None,
    ) -> SemanticFunction:
        """
        title: Register one top-level function.
        parameters:
          prototype:
            type: astx.FunctionPrototype
          definition:
            type: astx.FunctionDef | None
        returns:
          type: SemanticFunction
        """
        signature = self.normalize_function_signature(
            prototype,
            definition=definition,
        )
        module_key = self._current_module_key()
        existing = self.context.get_function(module_key, prototype.name)
        if existing is not None:
            if not self.signatures_match(existing.signature, signature):
                self.context.diagnostics.add(
                    f"Conflicting declaration for function '{prototype.name}'",
                    node=definition or prototype,
                )
                return existing
            if definition is not None and existing.definition is not None:
                self.context.diagnostics.add(
                    f"Function '{prototype.name}' already defined",
                    node=definition,
                )
            if definition is not None and existing.definition is None:
                updated = replace(existing, definition=definition)
                self.context.register_function(updated)
                return updated
            return existing

        function = self.factory.make_function(
            module_key,
            prototype,
            signature=signature,
            definition=definition,
        )
        self.context.register_function(function)
        return function

    def register_struct(
        self,
        node: astx.StructDefStmt,
    ) -> SemanticStruct:
        """
        title: Register one top-level struct.
        parameters:
          node:
            type: astx.StructDefStmt
        returns:
          type: SemanticStruct
        """
        module_key = self._current_module_key()
        existing = self.context.get_struct(module_key, node.name)
        if existing is not None:
            if existing.declaration is not node:
                self.context.diagnostics.add(
                    f"Struct '{node.name}' already defined.",
                    node=node,
                )
            return existing

        struct = self.factory.make_struct(module_key, node)
        self.context.register_struct(struct)
        return struct

    def resolve_function(
        self,
        name: str,
        *,
        module_key: ModuleKey | None = None,
    ) -> SemanticFunction | None:
        """
        title: Resolve one registered function.
        parameters:
          name:
            type: str
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticFunction | None
        """
        lookup_module_key = module_key or self._current_module_key()
        return self.context.get_function(lookup_module_key, name)

    def resolve_struct(
        self,
        name: str,
        *,
        module_key: ModuleKey | None = None,
    ) -> SemanticStruct | None:
        """
        title: Resolve one registered struct.
        parameters:
          name:
            type: str
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticStruct | None
        """
        lookup_module_key = module_key or self._current_module_key()
        return self.context.get_struct(lookup_module_key, name)

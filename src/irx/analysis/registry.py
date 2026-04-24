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
from irx.analysis.ffi import (
    build_ffi_callable_info,
    normalize_runtime_features,
)
from irx.analysis.module_interfaces import ModuleKey
from irx.analysis.resolved_nodes import (
    CallingConvention,
    FunctionSignature,
    ParameterSpec,
    SemanticClass,
    SemanticFunction,
    SemanticStruct,
    SemanticSymbol,
)
from irx.analysis.types import display_type_name, same_type
from irx.diagnostics import (
    DiagnosticCodes,
    DiagnosticRelatedInformation,
)
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
            code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
        )
        return prototype_name

    def _argument_has_default(
        self,
        argument: astx.Argument,
    ) -> bool:
        """
        title: Return whether one parameter declares a default value.
        parameters:
          argument:
            type: astx.Argument
        returns:
          type: bool
        """
        return not isinstance(argument.default, astx.Undefined)

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
            code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
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

    def _signature_mismatch_detail(
        self,
        lhs: FunctionSignature,
        rhs: FunctionSignature,
        *,
        abi_only: bool,
    ) -> str:
        """
        title: Describe the first incompatible function-signature field.
        parameters:
          lhs:
            type: FunctionSignature
          rhs:
            type: FunctionSignature
          abi_only:
            type: bool
        returns:
          type: str
        """
        if not abi_only and lhs.name != rhs.name:
            return f"name differs ('{lhs.name}' vs '{rhs.name}')"
        if lhs.calling_convention is not rhs.calling_convention:
            return (
                "calling_convention differs "
                f"('{lhs.calling_convention.value}' vs "
                f"'{rhs.calling_convention.value}')"
            )
        if lhs.is_variadic != rhs.is_variadic:
            return (
                f"is_variadic differs ({lhs.is_variadic} vs {rhs.is_variadic})"
            )
        if lhs.is_extern != rhs.is_extern:
            return f"is_extern differs ({lhs.is_extern} vs {rhs.is_extern})"
        if lhs.symbol_name != rhs.symbol_name:
            return (
                f"symbol_name differs ('{lhs.symbol_name}' vs "
                f"'{rhs.symbol_name}')"
            )
        if lhs.required_runtime_features != rhs.required_runtime_features:
            return (
                "required_runtime_features differ "
                f"({lhs.required_runtime_features} vs "
                f"{rhs.required_runtime_features})"
            )
        if not same_type(lhs.return_type, rhs.return_type):
            return (
                "return_type differs "
                f"('{display_type_name(lhs.return_type)}' vs "
                f"'{display_type_name(rhs.return_type)}')"
            )
        if len(lhs.parameters) != len(rhs.parameters):
            return (
                f"parameter count differs ({len(lhs.parameters)} vs "
                f"{len(rhs.parameters)})"
            )
        for idx, (lhs_param, rhs_param) in enumerate(
            zip(lhs.parameters, rhs.parameters)
        ):
            if not abi_only and lhs_param.name != rhs_param.name:
                return (
                    f"parameter {idx} name differs ('{lhs_param.name}' vs "
                    f"'{rhs_param.name}')"
                )
            if lhs_param.passing_kind is not rhs_param.passing_kind:
                return (
                    f"parameter {idx} passing_kind differs "
                    f"('{lhs_param.passing_kind.value}' vs "
                    f"'{rhs_param.passing_kind.value}')"
                )
            if not same_type(lhs_param.type_, rhs_param.type_):
                return (
                    "parameter "
                    f"{idx} type differs "
                    f"('{display_type_name(lhs_param.type_)}' vs "
                    f"'{display_type_name(rhs_param.type_)}')"
                )
        return "signature differs"

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
        if lhs.required_runtime_features != rhs.required_runtime_features:
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
        validate_ffi: bool = True,
        validate_main: bool = True,
    ) -> FunctionSignature:
        """
        title: Normalize and validate one semantic function signature.
        parameters:
          prototype:
            type: astx.FunctionPrototype
          definition:
            type: astx.FunctionDef | None
          validate_ffi:
            type: bool
          validate_main:
            type: bool
        returns:
          type: FunctionSignature
        """
        seen_parameter_names: set[str] = set()
        seen_default_parameter = False
        for argument in prototype.args.nodes:
            if argument.name in seen_parameter_names:
                self.context.diagnostics.add(
                    f"Function '{prototype.name}' repeats parameter "
                    f"'{argument.name}'",
                    node=argument,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                )
                continue
            seen_parameter_names.add(argument.name)
            if self._argument_has_default(argument):
                seen_default_parameter = True
                continue
            if not seen_default_parameter:
                continue
            self.context.diagnostics.add(
                f"Function '{prototype.name}' parameter '{argument.name}' "
                "without a default cannot follow a parameter with a "
                "default value",
                node=argument,
                code=DiagnosticCodes.SEMANTIC_CALL_ARITY,
            )

        is_extern = self._prototype_is_extern(prototype)
        is_variadic = self._prototype_is_variadic(prototype)
        symbol_name = self._prototype_symbol_name(prototype)
        required_runtime_features = normalize_runtime_features(
            prototype,
            diagnostics=self.context.diagnostics,
        )
        calling_convention = self._prototype_calling_convention(
            prototype,
            is_extern=is_extern,
        )

        if definition is not None and is_extern:
            self.context.diagnostics.add(
                f"Extern function '{prototype.name}' cannot define a body",
                node=definition,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
        if is_variadic and not is_extern:
            self.context.diagnostics.add(
                f"Function '{prototype.name}' may be variadic only when "
                "declared extern",
                node=prototype,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
        if is_extern and calling_convention is not CallingConvention.C:
            self.context.diagnostics.add(
                f"Extern function '{prototype.name}' must use calling "
                "convention 'c'",
                node=prototype,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
        if (
            not is_extern
            and calling_convention is not CallingConvention.IRX_DEFAULT
        ):
            self.context.diagnostics.add(
                f"IRx-defined function '{prototype.name}' must use calling "
                "convention 'irx_default'",
                node=prototype,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
        if not is_extern and symbol_name != prototype.name:
            self.context.diagnostics.add(
                f"Function '{prototype.name}' may override symbol_name only "
                "for extern declarations",
                node=prototype,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
        if not is_extern and required_runtime_features:
            self.context.diagnostics.add(
                f"Function '{prototype.name}' may declare runtime features "
                "only when declared extern",
                node=prototype,
                code=DiagnosticCodes.RUNTIME_FEATURE_UNKNOWN,
            )

        signature = self.factory.make_function_signature(
            prototype,
            calling_convention=calling_convention,
            is_variadic=is_variadic,
            is_extern=is_extern,
            symbol_name=symbol_name,
            required_runtime_features=required_runtime_features,
        )

        if validate_ffi and signature.is_extern:
            ffi = build_ffi_callable_info(
                self.context,
                signature=signature,
                prototype=prototype,
            )
            if ffi is not None:
                signature = replace(signature, ffi=ffi)

        if validate_main and signature.name == MAIN_FUNCTION_NAME:
            if signature.is_extern:
                self.context.diagnostics.add(
                    "Function 'main' cannot be extern",
                    node=prototype,
                    code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
                )
            if signature.is_variadic:
                self.context.diagnostics.add(
                    "Function 'main' must not be variadic",
                    node=prototype,
                    code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
                )
            if len(signature.parameters) != 0:
                self.context.diagnostics.add(
                    "Function 'main' must not declare parameters",
                    node=prototype,
                    code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
                )
            if (
                signature.calling_convention
                is not CallingConvention.IRX_DEFAULT
            ):
                self.context.diagnostics.add(
                    "Function 'main' must use calling convention "
                    "'irx_default'",
                    node=prototype,
                    code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
                )
            if not same_type(signature.return_type, astx.Int32()):
                self.context.diagnostics.add(
                    "Function 'main' must return Int32",
                    node=prototype,
                    code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
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
        current_scope = self.context.scopes.current
        existing = (
            current_scope.symbols.get(name)
            if current_scope is not None
            else None
        )
        if not self.context.scopes.declare(symbol):
            self.context.diagnostics.add(
                f"Identifier already declared: {name}",
                node=declaration,
                code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                related=(
                    ()
                    if existing is None or existing.declaration is None
                    else (
                        DiagnosticRelatedInformation(
                            "previous declaration is here",
                            node=existing.declaration,
                            module_key=existing.module_key,
                        ),
                    )
                ),
            )
        return symbol

    def register_function(
        self,
        prototype: astx.FunctionPrototype,
        *,
        definition: astx.FunctionDef | None = None,
        validate_ffi: bool = True,
    ) -> SemanticFunction:
        """
        title: Register one top-level function.
        parameters:
          prototype:
            type: astx.FunctionPrototype
          definition:
            type: astx.FunctionDef | None
          validate_ffi:
            type: bool
        returns:
          type: SemanticFunction
        """
        signature = self.normalize_function_signature(
            prototype,
            definition=definition,
            validate_ffi=validate_ffi,
        )
        module_key = self._current_module_key()
        existing = self.context.get_function(module_key, prototype.name)
        if existing is not None:
            if not self.signatures_match(existing.signature, signature):
                mismatch = self._signature_mismatch_detail(
                    existing.signature,
                    signature,
                    abi_only=False,
                )
                self.context.diagnostics.add(
                    f"Conflicting declaration for function "
                    f"'{prototype.name}': {mismatch}",
                    node=definition or prototype,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    related=(
                        DiagnosticRelatedInformation(
                            "previous declaration is here",
                            node=existing.definition or existing.prototype,
                            module_key=existing.module_key,
                        ),
                    ),
                )
                return existing
            if definition is not None and existing.definition is not None:
                self.context.diagnostics.add(
                    f"Function '{prototype.name}' already defined",
                    node=definition,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    related=(
                        DiagnosticRelatedInformation(
                            "previous definition is here",
                            node=existing.definition,
                            module_key=existing.module_key,
                        ),
                    ),
                )
            if definition is not None and existing.definition is None:
                updated = replace(existing, definition=definition)
                self.context.register_function(updated)
                return updated
            return existing

        if signature.is_extern:
            for candidate in self.context.functions.values():
                if candidate.module_key != module_key:
                    continue
                candidate_signature = candidate.signature
                if not candidate_signature.is_extern:
                    continue
                if candidate_signature.symbol_name != signature.symbol_name:
                    continue
                if candidate_signature.name == prototype.name:
                    continue
                if self.signatures_match(candidate_signature, signature):
                    continue
                mismatch = self._signature_mismatch_detail(
                    candidate_signature,
                    signature,
                    abi_only=True,
                )
                self.context.diagnostics.add(
                    f"Extern symbol '{signature.symbol_name}' is declared "
                    f"incompatibly by '{candidate.name}' and "
                    f"'{prototype.name}': {mismatch}",
                    node=definition or prototype,
                    code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
                    related=(
                        DiagnosticRelatedInformation(
                            "previous extern declaration is here",
                            node=candidate.definition or candidate.prototype,
                            module_key=candidate.module_key,
                        ),
                    ),
                )
                break

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
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    related=(
                        DiagnosticRelatedInformation(
                            "previous struct definition is here",
                            node=existing.declaration,
                            module_key=existing.module_key,
                        ),
                    ),
                )
            return existing

        struct = self.factory.make_struct(module_key, node)
        self.context.register_struct(struct)
        return struct

    def register_class(
        self,
        node: astx.ClassDefStmt,
    ) -> SemanticClass:
        """
        title: Register one top-level class.
        parameters:
          node:
            type: astx.ClassDefStmt
        returns:
          type: SemanticClass
        """
        module_key = self._current_module_key()
        existing = self.context.get_class(module_key, node.name)
        if existing is not None:
            if existing.declaration is not node:
                self.context.diagnostics.add(
                    f"Class '{node.name}' already defined.",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    related=(
                        DiagnosticRelatedInformation(
                            "previous class definition is here",
                            node=existing.declaration,
                            module_key=existing.module_key,
                        ),
                    ),
                )
            return existing

        class_ = self.factory.make_class(module_key, node)
        self.context.register_class(class_)
        return class_

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

    def resolve_class(
        self,
        name: str,
        *,
        module_key: ModuleKey | None = None,
    ) -> SemanticClass | None:
        """
        title: Resolve one registered class.
        parameters:
          name:
            type: str
          module_key:
            type: ModuleKey | None
        returns:
          type: SemanticClass | None
        """
        lookup_module_key = module_key or self._current_module_key()
        return self.context.get_class(lookup_module_key, name)

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

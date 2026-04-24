"""
title: Declaration class-method helpers.
summary: >-
  Normalize class-method signatures and build the lowered callable metadata
  used during declaration analysis.
"""

from __future__ import annotations

from dataclasses import replace

from irx import astx
from irx.analysis.handlers.base import SemanticVisitorMixinBase
from irx.analysis.module_symbols import (
    class_method_symbol_basename,
    qualified_class_method_name,
)
from irx.analysis.resolved_nodes import (
    FunctionSignature,
    ParameterSpec,
    SemanticClass,
    SemanticClassMember,
    SemanticFunction,
    SemanticSymbol,
)
from irx.analysis.types import clone_type
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked

IMPLICIT_METHOD_RECEIVER_NAME = "self"


@typechecked
class DeclarationClassMethodVisitorMixin(SemanticVisitorMixinBase):
    """
    title: Declaration class-method helpers.
    """

    def _normalize_class_method_signature(
        self,
        class_: SemanticClass,
        declaration: astx.FunctionDef,
    ) -> FunctionSignature:
        """
        title: Normalize one class-method signature.
        parameters:
          class_:
            type: SemanticClass
          declaration:
            type: astx.FunctionDef
        returns:
          type: FunctionSignature
        """
        signature = self.registry.normalize_function_signature(
            declaration.prototype,
            definition=declaration,
            validate_ffi=False,
            validate_main=False,
        )
        if signature.is_extern:
            self.context.diagnostics.add(
                (
                    f"Class method '{class_.name}.{declaration.name}' "
                    "cannot be extern"
                ),
                node=declaration,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
        if signature.is_variadic:
            self.context.diagnostics.add(
                (
                    f"Class method '{class_.name}.{declaration.name}' "
                    "must not be variadic"
                ),
                node=declaration,
                code=DiagnosticCodes.FFI_INVALID_SIGNATURE,
            )
        return signature

    def _make_method_receiver_type(
        self,
        class_: SemanticClass,
    ) -> astx.ClassType:
        """
        title: Return the canonical implicit receiver type for one class.
        parameters:
          class_:
            type: SemanticClass
        returns:
          type: astx.ClassType
        """
        return astx.ClassType(
            class_.name,
            resolved_name=class_.name,
            module_key=class_.module_key,
            qualified_name=class_.qualified_name,
            ancestor_qualified_names=tuple(
                ancestor.qualified_name for ancestor in class_.mro[1:]
            ),
        )

    def _make_visible_method_function(
        self,
        class_: SemanticClass,
        member: SemanticClassMember,
    ) -> SemanticFunction:
        """
        title: Build one visible-signature callable wrapper for a class method.
        parameters:
          class_:
            type: SemanticClass
          member:
            type: SemanticClassMember
        returns:
          type: SemanticFunction
        """
        declaration = member.declaration
        if (
            member.signature is None
            or member.lowered_function is None
            or not isinstance(declaration, astx.FunctionDef)
        ):
            raise TypeError("class method must have a lowered function")
        return SemanticFunction(
            symbol_id=member.lowered_function.symbol_id,
            name=f"{class_.name}.{member.name}",
            return_type=clone_type(member.signature.return_type),
            args=(),
            signature=member.signature,
            prototype=declaration.prototype,
            definition=declaration,
            module_key=class_.module_key,
            qualified_name=qualified_class_method_name(
                class_.module_key,
                class_.name,
                member.name,
                member.signature_key,
            ),
        )

    def _make_lowered_method_function(
        self,
        class_: SemanticClass,
        declaration: astx.FunctionDef,
        signature: FunctionSignature,
        *,
        is_static: bool,
        signature_key: str,
    ) -> SemanticFunction:
        """
        title: Build one lowered semantic function for a class method.
        parameters:
          class_:
            type: SemanticClass
          declaration:
            type: astx.FunctionDef
          signature:
            type: FunctionSignature
          is_static:
            type: bool
          signature_key:
            type: str
        returns:
          type: SemanticFunction
        """
        lowered_signature = signature
        user_args = tuple(
            self.factory.make_parameter_symbol(class_.module_key, argument)
            for argument in declaration.prototype.args.nodes
        )
        receiver_args: tuple[SemanticSymbol, ...] = ()
        if not is_static:
            receiver_type = self._make_method_receiver_type(class_)
            receiver_symbol = self.factory.make_variable_symbol(
                class_.module_key,
                IMPLICIT_METHOD_RECEIVER_NAME,
                receiver_type,
                is_mutable=False,
                declaration=declaration,
                kind="method_receiver",
            )
            receiver_spec = ParameterSpec(
                name=IMPLICIT_METHOD_RECEIVER_NAME,
                type_=clone_type(receiver_type),
            )
            lowered_signature = replace(
                signature,
                parameters=(receiver_spec, *signature.parameters),
                symbol_name=class_method_symbol_basename(
                    class_.name,
                    declaration.name,
                    signature_key,
                ),
                metadata={
                    **signature.metadata,
                    "class_name": class_.name,
                    "method_name": declaration.name,
                    "has_hidden_receiver": True,
                },
            )
            receiver_args = (receiver_symbol,)
        else:
            lowered_signature = replace(
                signature,
                symbol_name=class_method_symbol_basename(
                    class_.name,
                    declaration.name,
                    signature_key,
                ),
                metadata={
                    **signature.metadata,
                    "class_name": class_.name,
                    "method_name": declaration.name,
                    "has_hidden_receiver": False,
                },
            )
        semantic_args = (*receiver_args, *user_args)
        lowered_function = self.factory.make_function(
            class_.module_key,
            declaration.prototype,
            signature=lowered_signature,
            definition=declaration,
            args=semantic_args,
        )
        return replace(
            lowered_function,
            qualified_name=qualified_class_method_name(
                class_.module_key,
                class_.name,
                declaration.name,
                signature_key,
            ),
        )

    def _analyze_class_method_body(
        self,
        class_: SemanticClass,
        member: SemanticClassMember,
    ) -> None:
        """
        title: Analyze one lowered class method body.
        parameters:
          class_:
            type: SemanticClass
          member:
            type: SemanticClassMember
        """
        declaration = member.declaration
        function = member.lowered_function
        if not isinstance(declaration, astx.FunctionDef) or function is None:
            return
        if function.template_params:
            self._prepare_function_template_specializations(function)
            self._analyze_function_template_specializations(function)
            return

        hidden_parameter_count = len(function.args) - len(
            declaration.prototype.args.nodes
        )
        with self.context.in_function(function):
            with self.context.scope("method"):
                for idx, arg_symbol in enumerate(function.args):
                    self.context.scopes.declare(arg_symbol)
                    if idx < hidden_parameter_count:
                        continue
                    arg_node = declaration.prototype.args.nodes[
                        idx - hidden_parameter_count
                    ]
                    self._set_symbol(arg_node, arg_symbol)
                    self._set_type(arg_node, arg_symbol.type_)
                self.visit(declaration.body)
        if not isinstance(function.return_type, astx.NoneType) and not (
            self._guarantees_return(declaration.body)
        ):
            self.context.diagnostics.add(
                f"Function '{class_.name}.{member.name}' with return type "
                f"'{function.return_type}' is missing a return statement",
                node=declaration,
            )

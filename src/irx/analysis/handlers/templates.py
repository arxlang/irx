"""
title: Template-specialization semantic helpers.
summary: >-
  Prepare, validate, and specialize compile-time template functions into
  concrete function definitions that the rest of semantic analysis and lowering
  can consume directly.
"""

from __future__ import annotations

import copy

from dataclasses import replace
from itertools import product
from typing import Iterable

from irx import astx
from irx.analysis.handlers.base import SemanticVisitorMixinBase
from irx.analysis.module_symbols import specialized_function_basename
from irx.analysis.resolved_nodes import (
    SemanticFunction,
    SemanticSymbol,
    TemplateArgumentBinding,
    TemplateSpecializationKey,
)
from irx.analysis.types import clone_type, display_type_name, same_type
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked

_SPECIALIZATION_ANALYZED_ATTR = "irx_template_specialization_analyzed"
_OWNER_MODULE_ATTR = "irx_owner_module"
_TEMPLATE_PREPARED_ATTR = "irx_template_specializations_prepared"


@typechecked
class TemplateVisitorMixin(SemanticVisitorMixinBase):
    def _type_identity_name(self, type_: astx.DataType) -> str:
        """
        title: Return one deterministic type identity string.
        parameters:
          type_:
            type: astx.DataType
        returns:
          type: str
        """
        if isinstance(type_, astx.StructType):
            return type_.qualified_name or type_.name
        if isinstance(type_, astx.ClassType):
            return type_.qualified_name or type_.name
        if isinstance(type_, astx.NamespaceType):
            return type_.module_key
        if isinstance(type_, astx.OpaqueHandleType):
            return type_.handle_name
        if isinstance(type_, astx.UnionType):
            if type_.alias_name is not None:
                return type_.alias_name
            return "_or_".join(
                self._type_identity_name(member) for member in type_.members
            )
        if isinstance(type_, astx.TemplateTypeVar):
            return type_.name
        return str(type_.__class__.__name__)

    def _template_specialization_key(
        self,
        function: SemanticFunction,
        concrete_args: tuple[astx.DataType, ...],
    ) -> TemplateSpecializationKey:
        """
        title: Return one stable specialization key for a template function.
        parameters:
          function:
            type: SemanticFunction
          concrete_args:
            type: tuple[astx.DataType, Ellipsis]
        returns:
          type: TemplateSpecializationKey
        """
        return TemplateSpecializationKey(
            qualified_name=function.qualified_name,
            arg_type_names=tuple(
                self._type_identity_name(type_) for type_ in concrete_args
            ),
        )

    def _template_specialization_name(
        self,
        function: SemanticFunction,
        concrete_args: tuple[astx.DataType, ...],
    ) -> str:
        """
        title: Return one deterministic specialization function basename.
        parameters:
          function:
            type: SemanticFunction
          concrete_args:
            type: tuple[astx.DataType, Ellipsis]
        returns:
          type: str
        """
        key = self._template_specialization_key(function, concrete_args)
        return specialized_function_basename(function.name, key.arg_type_names)

    def _template_specialization_symbol_name(
        self,
        function: SemanticFunction,
        concrete_args: tuple[astx.DataType, ...],
    ) -> str:
        """
        title: Return one deterministic lowered specialization symbol name.
        parameters:
          function:
            type: SemanticFunction
          concrete_args:
            type: tuple[astx.DataType, Ellipsis]
        returns:
          type: str
        """
        key = self._template_specialization_key(function, concrete_args)
        base_name = function.signature.symbol_name or function.name
        return specialized_function_basename(base_name, key.arg_type_names)

    def _template_param_domain(
        self,
        param: astx.TemplateParam,
    ) -> tuple[astx.DataType, ...]:
        """
        title: Return the finite admissible type domain for one template param.
        parameters:
          param:
            type: astx.TemplateParam
        returns:
          type: tuple[astx.DataType, Ellipsis]
        """
        bound = param.bound
        if isinstance(bound, astx.UnionType):
            return tuple(clone_type(member) for member in bound.members)
        return (clone_type(bound),)

    def _type_within_template_bound(
        self,
        type_: astx.DataType,
        bound: astx.DataType,
    ) -> bool:
        """
        title: Return whether one concrete type satisfies one template bound.
        parameters:
          type_:
            type: astx.DataType
          bound:
            type: astx.DataType
        returns:
          type: bool
        """
        if isinstance(bound, astx.UnionType):
            return any(
                self._type_within_template_bound(type_, member)
                for member in bound.members
            )
        return same_type(type_, bound)

    def _substitute_type(
        self,
        type_: astx.DataType,
        bindings: dict[str, astx.DataType],
    ) -> astx.DataType:
        """
        title: Substitute template variables inside one type.
        parameters:
          type_:
            type: astx.DataType
          bindings:
            type: dict[str, astx.DataType]
        returns:
          type: astx.DataType
        """
        if isinstance(type_, astx.TemplateTypeVar):
            return clone_type(bindings.get(type_.name, type_))
        if isinstance(type_, astx.UnionType):
            return astx.UnionType(
                tuple(
                    self._substitute_type(member, bindings)
                    for member in type_.members
                ),
                alias_name=type_.alias_name,
            )
        if isinstance(type_, astx.PointerType):
            pointee_type = (
                self._substitute_type(type_.pointee_type, bindings)
                if type_.pointee_type is not None
                else None
            )
            return astx.PointerType(pointee_type)
        if isinstance(type_, astx.BufferViewType):
            element_type = (
                self._substitute_type(type_.element_type, bindings)
                if type_.element_type is not None
                else None
            )
            return astx.BufferViewType(element_type)
        if isinstance(type_, astx.NdarrayType):
            element_type = (
                self._substitute_type(type_.element_type, bindings)
                if type_.element_type is not None
                else None
            )
            return astx.NdarrayType(element_type)
        return clone_type(type_)

    def _template_bindings_map(
        self,
        function: SemanticFunction,
        concrete_args: tuple[astx.DataType, ...],
    ) -> dict[str, astx.DataType]:
        """
        title: Return the concrete binding map for one specialization.
        parameters:
          function:
            type: SemanticFunction
          concrete_args:
            type: tuple[astx.DataType, Ellipsis]
        returns:
          type: dict[str, astx.DataType]
        """
        return {
            param.name: clone_type(argument)
            for param, argument in zip(function.template_params, concrete_args)
        }

    def _specialization_bindings_map(
        self,
        function: SemanticFunction,
    ) -> dict[str, astx.DataType]:
        """
        title: Return the concrete binding map stored on one specialization.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: dict[str, astx.DataType]
        """
        return {
            binding.name: clone_type(binding.type_)
            for binding in function.template_bindings
        }

    def _specialize_signature(
        self,
        function: SemanticFunction,
        bindings: dict[str, astx.DataType],
    ) -> SemanticFunction:
        """
        title: Return one visible callable wrapper with substituted types.
        parameters:
          function:
            type: SemanticFunction
          bindings:
            type: dict[str, astx.DataType]
        returns:
          type: SemanticFunction
        """
        signature = replace(
            function.signature,
            parameters=tuple(
                replace(
                    parameter,
                    type_=self._substitute_type(parameter.type_, bindings),
                )
                for parameter in function.signature.parameters
            ),
            return_type=self._substitute_type(
                function.signature.return_type,
                bindings,
            ),
        )
        return replace(
            function,
            return_type=clone_type(signature.return_type),
            signature=signature,
        )

    def _clear_semantic_sidecars(self, node: astx.AST) -> None:
        """
        title: Remove semantic sidecars from one cloned AST subtree.
        parameters:
          node:
            type: astx.AST
        """
        seen: set[int] = set()

        def clear(current: astx.AST) -> None:
            """
            title: Clear semantic sidecars from one reachable AST node.
            parameters:
              current:
                type: astx.AST
            """
            current_id = id(current)
            if current_id in seen:
                return
            seen.add(current_id)
            if hasattr(current, "semantic"):
                delattr(current, "semantic")
            for value in vars(current).values():
                if isinstance(value, astx.AST):
                    clear(value)
                    continue
                if isinstance(value, list | tuple):
                    for item in value:
                        if isinstance(item, astx.AST):
                            clear(item)

        clear(node)

    def _reset_template_analysis_state(self, module: astx.Module) -> None:
        """
        title: Reset per-run template state attached to one module AST.
        parameters:
          module:
            type: astx.Module
        """
        astx.clear_generated_template_nodes(module)
        seen: set[int] = set()

        def clear(current: astx.AST) -> None:
            """
            title: Clear template-analysis markers from one reachable AST node.
            parameters:
              current:
                type: astx.AST
            """
            current_id = id(current)
            if current_id in seen:
                return
            seen.add(current_id)
            for attr_name in (
                _SPECIALIZATION_ANALYZED_ATTR,
                _TEMPLATE_PREPARED_ATTR,
            ):
                if hasattr(current, attr_name):
                    delattr(current, attr_name)
            for value in vars(current).values():
                if isinstance(value, astx.AST):
                    clear(value)
                    continue
                if isinstance(value, list | tuple):
                    for item in value:
                        if isinstance(item, astx.AST):
                            clear(item)

        clear(module)

    def _substitute_declared_types(
        self,
        node: astx.AST,
        bindings: dict[str, astx.DataType],
    ) -> None:
        """
        title: >-
          Substitute template types throughout one cloned function subtree.
        parameters:
          node:
            type: astx.AST
          bindings:
            type: dict[str, astx.DataType]
        """
        if isinstance(node, astx.FunctionDef):
            self._substitute_declared_types(node.prototype, bindings)
            self._substitute_declared_types(node.body, bindings)
            explicit_args = astx.get_template_args(node)
            if explicit_args is not None:
                astx.set_template_args(
                    node,
                    tuple(
                        self._substitute_type(arg, bindings)
                        for arg in explicit_args
                    ),
                )
            return
        if isinstance(node, astx.FunctionPrototype):
            for argument in node.args.nodes:
                self._substitute_declared_types(argument, bindings)
            node.return_type = self._substitute_type(
                node.return_type,
                bindings,
            )
            explicit_args = astx.get_template_args(node)
            if explicit_args is not None:
                astx.set_template_args(
                    node,
                    tuple(
                        self._substitute_type(arg, bindings)
                        for arg in explicit_args
                    ),
                )
            return
        if isinstance(node, astx.Argument):
            node.type_ = self._substitute_type(node.type_, bindings)
            return
        if isinstance(node, astx.VariableDeclaration):
            node.type_ = self._substitute_type(node.type_, bindings)
            if node.value is not None:
                self._substitute_declared_types(node.value, bindings)
            return
        if isinstance(node, astx.Cast):
            node.target_type = self._substitute_type(
                node.target_type,
                bindings,
            )
            self._substitute_declared_types(node.value, bindings)
            return

        explicit_args = astx.get_template_args(node)
        if explicit_args is not None:
            astx.set_template_args(
                node,
                tuple(
                    self._substitute_type(arg, bindings)
                    for arg in explicit_args
                ),
            )

        for name, value in vars(node).items():
            if name in {
                "semantic",
                "parent",
                "type_",
                "return_type",
                "target_type",
            }:
                continue
            if isinstance(value, astx.AST):
                self._substitute_declared_types(value, bindings)
                continue
            if isinstance(value, list | tuple):
                for item in value:
                    if isinstance(item, astx.AST):
                        self._substitute_declared_types(item, bindings)

    def _function_owner_module(
        self,
        function: SemanticFunction,
    ) -> astx.Module | None:
        """
        title: Return the owning AST module for one semantic function.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: astx.Module | None
        """
        if (
            self.session is not None
            and function.module_key in self.session.modules
        ):
            return self.session.module(function.module_key).ast
        current_module = getattr(self, "_current_ast_module", None)
        if isinstance(current_module, astx.Module):
            return current_module
        owner = getattr(function.prototype, _OWNER_MODULE_ATTR, None)
        if isinstance(owner, astx.Module):
            return owner
        owner = getattr(function.definition, _OWNER_MODULE_ATTR, None)
        if isinstance(owner, astx.Module):
            return owner
        return None

    def _specialized_function_args(
        self,
        function: SemanticFunction,
        definition: astx.FunctionDef,
        bindings: dict[str, astx.DataType],
    ) -> tuple[SemanticSymbol, ...]:
        """
        title: Return substituted semantic argument symbols.
        parameters:
          function:
            type: SemanticFunction
          definition:
            type: astx.FunctionDef
          bindings:
            type: dict[str, astx.DataType]
        returns:
          type: tuple[SemanticSymbol, Ellipsis]
        """
        visible_args = tuple(definition.prototype.args.nodes)
        hidden_count = len(function.args) - len(visible_args)
        return tuple(
            self.factory.make_variable_symbol(
                function.module_key,
                arg_symbol.name,
                self._substitute_type(arg_symbol.type_, bindings),
                is_mutable=arg_symbol.is_mutable,
                declaration=(
                    definition
                    if index < hidden_count
                    else visible_args[index - hidden_count]
                ),
                kind=arg_symbol.kind,
            )
            for index, arg_symbol in enumerate(function.args)
        )

    def _build_template_specialization(
        self,
        function: SemanticFunction,
        concrete_args: tuple[astx.DataType, ...],
    ) -> SemanticFunction | None:
        """
        title: Build or reuse one concrete specialization for a template func.
        parameters:
          function:
            type: SemanticFunction
          concrete_args:
            type: tuple[astx.DataType, Ellipsis]
        returns:
          type: SemanticFunction | None
        """
        key = self._template_specialization_key(function, concrete_args)
        existing = function.specializations.get(key)
        if existing is not None:
            return existing
        if function.definition is None:
            self.context.diagnostics.add(
                f"Template function '{function.name}' must have a definition",
                node=function.prototype,
                code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
            )
            return None
        owner_module = self._function_owner_module(function)
        if owner_module is None:
            raise TypeError(
                "template specialization requires an owning module"
            )

        bindings = self._template_bindings_map(function, concrete_args)
        clone_def = copy.deepcopy(function.definition)
        self._clear_semantic_sidecars(clone_def)
        specialization_name = self._template_specialization_name(
            function,
            concrete_args,
        )
        specialization_symbol_name = self._template_specialization_symbol_name(
            function,
            concrete_args,
        )
        clone_def.prototype.name = specialization_name
        setattr(clone_def, _OWNER_MODULE_ATTR, owner_module)
        setattr(clone_def.prototype, _OWNER_MODULE_ATTR, owner_module)
        astx.set_template_params(clone_def, ())
        astx.set_template_params(clone_def.prototype, ())
        astx.mark_template_specialization(clone_def, specialization_name)
        astx.mark_template_specialization(
            clone_def.prototype,
            specialization_name,
        )
        self._substitute_declared_types(clone_def, bindings)

        signature = replace(
            function.signature,
            name=specialization_name,
            parameters=tuple(
                replace(
                    parameter,
                    type_=self._substitute_type(parameter.type_, bindings),
                )
                for parameter in function.signature.parameters
            ),
            return_type=self._substitute_type(
                function.signature.return_type,
                bindings,
            ),
            symbol_name=specialization_symbol_name,
        )
        specialized_function = self.factory.make_function(
            function.module_key,
            clone_def.prototype,
            signature=signature,
            definition=clone_def,
            args=self._specialized_function_args(
                function,
                clone_def,
                bindings,
            ),
        )
        specialized_function = replace(
            specialized_function,
            qualified_name=(
                f"{function.qualified_name}::specialization::"
                f"{'__'.join(key.arg_type_names)}"
            ),
            template_bindings=tuple(
                TemplateArgumentBinding(param.name, clone_type(argument))
                for param, argument in zip(
                    function.template_params,
                    concrete_args,
                )
            ),
            template_definition=function,
            specialization_key=key,
        )
        function.specializations[key] = specialized_function
        self.context.register_function(specialized_function)
        self._set_function(clone_def.prototype, specialized_function)
        self._set_function(clone_def, specialized_function)
        self._set_type(clone_def.prototype, None)
        self._set_type(clone_def, None)
        astx.add_generated_template_node(owner_module, clone_def)
        return specialized_function

    def _collect_inferred_template_bindings(
        self,
        expected_type: astx.DataType,
        actual_type: astx.DataType | None,
        inferred: dict[str, astx.DataType],
        *,
        function: SemanticFunction,
        node: astx.AST,
        diagnose: bool = True,
    ) -> bool:
        """
        title: Collect inferred template arguments from one parameter pair.
        parameters:
          expected_type:
            type: astx.DataType
          actual_type:
            type: astx.DataType | None
          inferred:
            type: dict[str, astx.DataType]
          function:
            type: SemanticFunction
          node:
            type: astx.AST
          diagnose:
            type: bool
        returns:
          type: bool
        """
        if isinstance(expected_type, astx.TemplateTypeVar):
            if actual_type is None:
                if diagnose:
                    self.context.diagnostics.add(
                        "Cannot infer template argument "
                        f"'{expected_type.name}' "
                        f"for call to '{function.name}'",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                    )
                return False
            previous = inferred.get(expected_type.name)
            if previous is None:
                inferred[expected_type.name] = actual_type
                return True
            if same_type(previous, actual_type):
                return True
            if diagnose:
                self.context.diagnostics.add(
                    "Conflicting inferences for template argument "
                    f"'{expected_type.name}' in call to '{function.name}'",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            return False
        if isinstance(expected_type, astx.PointerType) and isinstance(
            actual_type,
            astx.PointerType,
        ):
            if (
                expected_type.pointee_type is None
                or actual_type.pointee_type is None
            ):
                return True
            return self._collect_inferred_template_bindings(
                expected_type.pointee_type,
                actual_type.pointee_type,
                inferred,
                function=function,
                node=node,
                diagnose=diagnose,
            )
        if isinstance(expected_type, astx.BufferViewType) and isinstance(
            actual_type,
            astx.BufferViewType,
        ):
            if (
                expected_type.element_type is None
                or actual_type.element_type is None
            ):
                return True
            return self._collect_inferred_template_bindings(
                expected_type.element_type,
                actual_type.element_type,
                inferred,
                function=function,
                node=node,
                diagnose=diagnose,
            )
        if isinstance(expected_type, astx.NdarrayType) and isinstance(
            actual_type,
            astx.NdarrayType,
        ):
            if (
                expected_type.element_type is None
                or actual_type.element_type is None
            ):
                return True
            return self._collect_inferred_template_bindings(
                expected_type.element_type,
                actual_type.element_type,
                inferred,
                function=function,
                node=node,
                diagnose=diagnose,
            )
        return True

    def _resolve_template_arguments(
        self,
        function: SemanticFunction,
        arg_types: list[astx.DataType | None],
        node: astx.AST,
        *,
        diagnose: bool = True,
    ) -> tuple[astx.DataType, ...] | None:
        """
        title: Resolve explicit or inferred template arguments for one call.
        parameters:
          function:
            type: SemanticFunction
          arg_types:
            type: list[astx.DataType | None]
          node:
            type: astx.AST
          diagnose:
            type: bool
        returns:
          type: tuple[astx.DataType, Ellipsis] | None
        """
        explicit_args = astx.get_template_args(node)
        params = function.template_params
        if explicit_args is not None:
            if len(explicit_args) != len(params):
                if diagnose:
                    self.context.diagnostics.add(
                        f"Call to '{function.name}' supplies "
                        f"{len(explicit_args)} template arguments but expects "
                        f"{len(params)}",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_CALL_ARITY,
                    )
                return None
            concrete_args = tuple(clone_type(arg) for arg in explicit_args)
        else:
            inferred: dict[str, astx.DataType] = {}
            for parameter, arg_type in zip(
                function.signature.parameters,
                arg_types,
            ):
                if not self._collect_inferred_template_bindings(
                    parameter.type_,
                    arg_type,
                    inferred,
                    function=function,
                    node=node,
                    diagnose=diagnose,
                ):
                    return None
            missing = [
                param.name for param in params if param.name not in inferred
            ]
            if missing:
                if diagnose:
                    missing_text = ", ".join(missing)
                    self.context.diagnostics.add(
                        f"Cannot infer template argument(s) {missing_text} "
                        f"for call to '{function.name}'",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                    )
                return None
            concrete_args = tuple(
                clone_type(inferred[param.name]) for param in params
            )

        for param, argument in zip(params, concrete_args):
            if self._type_within_template_bound(argument, param.bound):
                continue
            if diagnose:
                self.context.diagnostics.add(
                    f"Template argument '{display_type_name(argument)}' does "
                    f"not satisfy bound '{display_type_name(param.bound)}' "
                    f"for parameter '{param.name}' in '{function.name}'",
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
                )
            return None
        return concrete_args

    def _resolve_template_call_target(
        self,
        function: SemanticFunction,
        arg_types: list[astx.DataType | None],
        node: astx.AST,
    ) -> SemanticFunction | None:
        """
        title: Resolve one template function call to a concrete specialization.
        parameters:
          function:
            type: SemanticFunction
          arg_types:
            type: list[astx.DataType | None]
          node:
            type: astx.AST
        returns:
          type: SemanticFunction | None
        """
        concrete_args = self._resolve_template_arguments(
            function,
            arg_types,
            node,
        )
        if concrete_args is None:
            return None
        specialization = self._build_template_specialization(
            function,
            concrete_args,
        )
        if specialization is None:
            return None
        definition = specialization.definition
        if definition is not None and not getattr(
            definition,
            _SPECIALIZATION_ANALYZED_ATTR,
            False,
        ):
            self._analyze_prepared_template_specialization(specialization)
        return specialization

    def _resolve_template_method_call_target(
        self,
        function: SemanticFunction,
        visible_function: SemanticFunction,
        arg_types: list[astx.DataType | None],
        node: astx.AST,
    ) -> SemanticFunction | None:
        """
        title: Resolve one template method call to a concrete specialization.
        parameters:
          function:
            type: SemanticFunction
          visible_function:
            type: SemanticFunction
          arg_types:
            type: list[astx.DataType | None]
          node:
            type: astx.AST
        returns:
          type: SemanticFunction | None
        """
        visible_template = replace(
            visible_function,
            template_params=function.template_params,
        )
        concrete_args = self._resolve_template_arguments(
            visible_template,
            arg_types,
            node,
        )
        if concrete_args is None:
            return None
        specialization = self._build_template_specialization(
            function,
            concrete_args,
        )
        if specialization is None:
            return None
        definition = specialization.definition
        if definition is not None and not getattr(
            definition,
            _SPECIALIZATION_ANALYZED_ATTR,
            False,
        ):
            self._analyze_prepared_template_specialization(specialization)
        return specialization

    def _template_specializations_prepared(
        self,
        function: SemanticFunction,
    ) -> bool:
        """
        title: >-
          Return whether one template function already prepared its clones.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: bool
        """
        carrier = function.definition or function.prototype
        return bool(getattr(carrier, _TEMPLATE_PREPARED_ATTR, False))

    def _mark_template_specializations_prepared(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Mark one template function as having prepared its clones.
        parameters:
          function:
            type: SemanticFunction
        """
        carrier = function.definition or function.prototype
        setattr(carrier, _TEMPLATE_PREPARED_ATTR, True)

    def _prepare_function_template_specializations(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Materialize all concrete specializations for one template func.
        parameters:
          function:
            type: SemanticFunction
        """
        if (
            not function.template_params
            or function.template_definition is not None
        ):
            return
        if function.definition is None:
            self.context.diagnostics.add(
                f"Template function '{function.name}' must have a definition",
                node=function.prototype,
                code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
            )
            return
        if self._template_specializations_prepared(function):
            return
        domains = tuple(
            self._template_param_domain(param)
            for param in function.template_params
        )
        for concrete_args in product(*domains):
            self._build_template_specialization(function, concrete_args)
        self._mark_template_specializations_prepared(function)

    def _prepare_template_specialization_skeletons(
        self,
        module: astx.Module,
    ) -> None:
        """
        title: Materialize specialization skeletons for module templates.
        parameters:
          module:
            type: astx.Module
        """
        for node in module.nodes:
            if isinstance(node, astx.FunctionPrototype):
                setattr(node, _OWNER_MODULE_ATTR, module)
                function = self.context.get_function(
                    self._current_module_key(),
                    node.name,
                )
                if (
                    function is not None
                    and function.template_params
                    and function.definition is None
                ):
                    self.context.diagnostics.add(
                        f"Template function '{function.name}' must have a "
                        "definition",
                        node=node,
                        code=DiagnosticCodes.SEMANTIC_DUPLICATE_DECLARATION,
                    )
                continue
            if not isinstance(node, astx.FunctionDef):
                continue
            setattr(node, _OWNER_MODULE_ATTR, module)
            setattr(node.prototype, _OWNER_MODULE_ATTR, module)
            function = self.context.get_function(
                self._current_module_key(),
                node.name,
            )
            if function is None:
                continue
            self._prepare_function_template_specializations(function)

    def _analyze_specialized_function_body(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Analyze one generated concrete specialization body.
        parameters:
          function:
            type: SemanticFunction
        """
        definition = function.definition
        if definition is None:
            raise TypeError("template specialization requires a definition")
        for argument in definition.prototype.args.nodes:
            self._resolve_declared_type(argument.type_, node=argument)
        self._resolve_declared_type(
            definition.prototype.return_type,
            node=definition,
        )
        self._set_function(definition.prototype, function)
        self._set_function(definition, function)
        self._set_type(definition.prototype, None)
        self._set_type(definition, None)
        hidden_parameter_count = len(function.args) - len(
            definition.prototype.args.nodes
        )
        with self.context.in_function(function):
            with self.context.scope("function"):
                for index, arg_symbol in enumerate(function.args):
                    self.context.scopes.declare(arg_symbol)
                    if index < hidden_parameter_count:
                        continue
                    arg_node = definition.prototype.args.nodes[
                        index - hidden_parameter_count
                    ]
                    self._set_symbol(arg_node, arg_symbol)
                    self._set_type(arg_node, arg_symbol.type_)
                self.visit(definition.body)
        if not isinstance(
            function.return_type, astx.NoneType
        ) and not self._guarantees_return(definition.body):
            self.context.diagnostics.add(
                f"Function '{function.name}' with return type "
                f"'{function.return_type}' is missing a return statement",
                node=definition,
            )

    def _format_template_bindings(
        self,
        bindings: Iterable[TemplateArgumentBinding],
    ) -> str:
        """
        title: Render one template-binding set for diagnostics.
        parameters:
          bindings:
            type: Iterable[TemplateArgumentBinding]
        returns:
          type: str
        """
        return ", ".join(
            f"{binding.name} = {display_type_name(binding.type_)}"
            for binding in bindings
        )

    def _analyze_prepared_template_specialization(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Analyze one prepared concrete specialization once.
        parameters:
          function:
            type: SemanticFunction
        """
        definition = function.definition
        if definition is None:
            raise TypeError(
                "prepared template specialization lacks definition"
            )
        if getattr(definition, _SPECIALIZATION_ANALYZED_ATTR, False):
            return
        diagnostic_count_before = len(self.context.diagnostics.diagnostics)
        self._analyze_specialized_function_body(function)
        setattr(definition, _SPECIALIZATION_ANALYZED_ATTR, True)
        if (
            len(self.context.diagnostics.diagnostics)
            == diagnostic_count_before
        ):
            return
        template_definition = function.template_definition
        if template_definition is None:
            return
        bindings_text = self._format_template_bindings(
            function.template_bindings
        )
        self.context.diagnostics.add(
            f"Template function '{template_definition.name}' is invalid for "
            f"{bindings_text}",
            node=(
                template_definition.definition or template_definition.prototype
            ),
            code=DiagnosticCodes.SEMANTIC_TYPE_MISMATCH,
        )

    def _analyze_function_template_specializations(
        self,
        function: SemanticFunction,
    ) -> None:
        """
        title: Analyze all prepared specializations for one template function.
        parameters:
          function:
            type: SemanticFunction
        """
        for specialization in tuple(function.specializations.values()):
            self._analyze_prepared_template_specialization(specialization)

    def _analyze_prepared_template_specializations(
        self,
        module: astx.Module,
    ) -> None:
        """
        title: Analyze the generated specializations attached to one module.
        parameters:
          module:
            type: astx.Module
        """
        for node in astx.generated_template_nodes(module):
            semantic = getattr(node, "semantic", None)
            function = getattr(semantic, "resolved_function", None)
            if not isinstance(function, SemanticFunction):
                continue
            self._analyze_prepared_template_specialization(function)

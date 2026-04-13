"""
title: Semantic entity factories.
summary: >-
  Build semantic sidecar entities and visible binding wrappers in one place so
  analyzer visitors can focus on traversal and rule orchestration.
"""

from __future__ import annotations

from public import public

from irx import astx
from irx.analysis.context import SemanticContext
from irx.analysis.module_interfaces import ModuleKey
from irx.analysis.module_symbols import (
    qualified_class_member_name,
    qualified_class_name,
    qualified_function_name,
    qualified_struct_name,
)
from irx.analysis.resolved_nodes import (
    CallableResolution,
    CallingConvention,
    ClassMemberKind,
    FFICallableInfo,
    FunctionSignature,
    ParameterPassingKind,
    ParameterSpec,
    ResolvedImportBinding,
    SemanticBinding,
    SemanticClass,
    SemanticClassMember,
    SemanticFunction,
    SemanticModule,
    SemanticStruct,
    SemanticSymbol,
)
from irx.analysis.symbols import variable_symbol
from irx.analysis.types import clone_type
from irx.typecheck import typechecked


@public
@typechecked
class SemanticEntityFactory:
    """
    title: Central semantic entity construction.
    summary: >-
      Create semantic symbols, functions, structs, modules, and visible
      bindings with consistent ids and qualified names.
    attributes:
      context:
        type: SemanticContext
    """

    context: SemanticContext

    def __init__(self, context: SemanticContext) -> None:
        """
        title: Initialize SemanticEntityFactory.
        parameters:
          context:
            type: SemanticContext
        """
        self.context = context

    def make_variable_symbol(
        self,
        module_key: ModuleKey,
        name: str,
        type_: astx.DataType,
        *,
        is_mutable: bool,
        declaration: astx.AST | None,
        kind: str = "variable",
    ) -> SemanticSymbol:
        """
        title: Create a variable-like semantic symbol.
        parameters:
          module_key:
            type: ModuleKey
          name:
            type: str
          type_:
            type: astx.DataType
          is_mutable:
            type: bool
          declaration:
            type: astx.AST | None
          kind:
            type: str
        returns:
          type: SemanticSymbol
        """
        return variable_symbol(
            self.context.next_symbol_id(kind),
            module_key,
            name,
            clone_type(type_),
            is_mutable=is_mutable,
            declaration=declaration,
            kind=kind,
        )

    def make_parameter_symbol(
        self,
        module_key: ModuleKey,
        argument: astx.Argument,
    ) -> SemanticSymbol:
        """
        title: Create one function-parameter semantic symbol.
        parameters:
          module_key:
            type: ModuleKey
          argument:
            type: astx.Argument
        returns:
          type: SemanticSymbol
        """
        return self.make_variable_symbol(
            module_key,
            argument.name,
            argument.type_,
            is_mutable=True,
            declaration=argument,
            kind="argument",
        )

    def make_parameter_spec(
        self,
        argument: astx.Argument,
    ) -> ParameterSpec:
        """
        title: Create one canonical semantic parameter specification.
        parameters:
          argument:
            type: astx.Argument
        returns:
          type: ParameterSpec
        """
        return ParameterSpec(
            name=argument.name,
            type_=clone_type(argument.type_),
            passing_kind=ParameterPassingKind.BY_VALUE,
        )

    def make_function_signature(
        self,
        prototype: astx.FunctionPrototype,
        *,
        calling_convention: CallingConvention,
        is_variadic: bool,
        is_extern: bool,
        symbol_name: str,
        required_runtime_features: tuple[str, ...] = (),
        ffi: FFICallableInfo | None = None,
    ) -> FunctionSignature:
        """
        title: Create one canonical semantic function signature.
        parameters:
          prototype:
            type: astx.FunctionPrototype
          calling_convention:
            type: CallingConvention
          is_variadic:
            type: bool
          is_extern:
            type: bool
          symbol_name:
            type: str
          required_runtime_features:
            type: tuple[str, Ellipsis]
          ffi:
            type: FFICallableInfo | None
        returns:
          type: FunctionSignature
        """
        return FunctionSignature(
            name=prototype.name,
            parameters=tuple(
                self.make_parameter_spec(argument)
                for argument in prototype.args.nodes
            ),
            return_type=clone_type(prototype.return_type),
            calling_convention=calling_convention,
            is_variadic=is_variadic,
            is_extern=is_extern,
            symbol_name=symbol_name,
            required_runtime_features=required_runtime_features,
            ffi=ffi,
        )

    def make_function(
        self,
        module_key: ModuleKey,
        prototype: astx.FunctionPrototype,
        *,
        signature: FunctionSignature,
        definition: astx.FunctionDef | None = None,
        args: tuple[SemanticSymbol, ...] | None = None,
    ) -> SemanticFunction:
        """
        title: Create one semantic function entity.
        parameters:
          module_key:
            type: ModuleKey
          prototype:
            type: astx.FunctionPrototype
          signature:
            type: FunctionSignature
          definition:
            type: astx.FunctionDef | None
          args:
            type: tuple[SemanticSymbol, Ellipsis] | None
        returns:
          type: SemanticFunction
        """
        semantic_args = args
        if semantic_args is None:
            semantic_args = tuple(
                self.make_parameter_symbol(module_key, argument)
                for argument in prototype.args.nodes
            )
        return SemanticFunction(
            symbol_id=self.context.next_symbol_id("fn"),
            name=prototype.name,
            return_type=clone_type(signature.return_type),
            args=semantic_args,
            signature=signature,
            prototype=prototype,
            definition=definition,
            module_key=module_key,
            qualified_name=qualified_function_name(module_key, prototype.name),
        )

    def make_callable_resolution(
        self,
        function: SemanticFunction,
    ) -> CallableResolution:
        """
        title: Create one resolved callable wrapper.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: CallableResolution
        """
        return CallableResolution(
            function=function,
            signature=function.signature,
        )

    def make_struct(
        self,
        module_key: ModuleKey,
        node: astx.StructDefStmt,
    ) -> SemanticStruct:
        """
        title: Create one semantic struct entity.
        parameters:
          module_key:
            type: ModuleKey
          node:
            type: astx.StructDefStmt
        returns:
          type: SemanticStruct
        """
        return SemanticStruct(
            symbol_id=self.context.next_symbol_id("struct"),
            name=node.name,
            module_key=module_key,
            qualified_name=qualified_struct_name(module_key, node.name),
            declaration=node,
        )

    def make_class(
        self,
        module_key: ModuleKey,
        node: astx.ClassDefStmt,
    ) -> SemanticClass:
        """
        title: Create one semantic class entity.
        parameters:
          module_key:
            type: ModuleKey
          node:
            type: astx.ClassDefStmt
        returns:
          type: SemanticClass
        """
        return SemanticClass(
            symbol_id=self.context.next_symbol_id("class"),
            name=node.name,
            module_key=module_key,
            qualified_name=qualified_class_name(module_key, node.name),
            declaration=node,
        )

    def make_class_member(
        self,
        class_: SemanticClass,
        *,
        name: str,
        kind: ClassMemberKind,
        declaration: astx.AST,
        visibility: astx.VisibilityKind,
        is_static: bool,
        is_constant: bool,
        is_mutable: bool,
        type_: astx.DataType | None = None,
        signature: FunctionSignature | None = None,
        overrides: str | None = None,
    ) -> SemanticClassMember:
        """
        title: Create one semantic class-member record.
        parameters:
          class_:
            type: SemanticClass
          name:
            type: str
          kind:
            type: ClassMemberKind
          declaration:
            type: astx.AST
          visibility:
            type: astx.VisibilityKind
          is_static:
            type: bool
          is_constant:
            type: bool
          is_mutable:
            type: bool
          type_:
            type: astx.DataType | None
          signature:
            type: FunctionSignature | None
          overrides:
            type: str | None
        returns:
          type: SemanticClassMember
        """
        prefix = (
            "class_attr"
            if kind is ClassMemberKind.ATTRIBUTE
            else "class_method"
        )
        return SemanticClassMember(
            symbol_id=self.context.next_symbol_id(prefix),
            name=name,
            qualified_name=qualified_class_member_name(
                class_.module_key,
                class_.name,
                name,
            ),
            owner_name=class_.name,
            owner_qualified_name=class_.qualified_name,
            kind=kind,
            visibility=visibility,
            is_static=is_static,
            is_constant=is_constant,
            is_mutable=is_mutable,
            declaration=declaration,
            type_=clone_type(type_) if type_ is not None else None,
            signature=signature,
            overrides=overrides,
        )

    def make_module(
        self,
        module_key: ModuleKey,
        *,
        display_name: str | None = None,
    ) -> SemanticModule:
        """
        title: Create one semantic module entity.
        parameters:
          module_key:
            type: ModuleKey
          display_name:
            type: str | None
        returns:
          type: SemanticModule
        """
        return SemanticModule(
            module_key=module_key,
            display_name=display_name,
        )

    def make_function_binding(
        self,
        function: SemanticFunction,
    ) -> SemanticBinding:
        """
        title: Create a visible binding for a function.
        parameters:
          function:
            type: SemanticFunction
        returns:
          type: SemanticBinding
        """
        return SemanticBinding(
            kind="function",
            module_key=function.module_key,
            qualified_name=function.qualified_name,
            function=function,
        )

    def make_struct_binding(
        self,
        struct: SemanticStruct,
    ) -> SemanticBinding:
        """
        title: Create a visible binding for a struct.
        parameters:
          struct:
            type: SemanticStruct
        returns:
          type: SemanticBinding
        """
        return SemanticBinding(
            kind="struct",
            module_key=struct.module_key,
            qualified_name=struct.qualified_name,
            struct=struct,
        )

    def make_class_binding(
        self,
        class_: SemanticClass,
    ) -> SemanticBinding:
        """
        title: Create a visible binding for a class.
        parameters:
          class_:
            type: SemanticClass
        returns:
          type: SemanticBinding
        """
        return SemanticBinding(
            kind="class",
            module_key=class_.module_key,
            qualified_name=class_.qualified_name,
            class_=class_,
        )

    def make_module_binding(
        self,
        module: SemanticModule,
    ) -> SemanticBinding:
        """
        title: Create a visible binding for a module.
        parameters:
          module:
            type: SemanticModule
        returns:
          type: SemanticBinding
        """
        return SemanticBinding(
            kind="module",
            module_key=module.module_key,
            qualified_name=str(module.module_key),
            module=module,
        )

    def make_import_binding(
        self,
        *,
        local_name: str,
        requested_name: str,
        source_module_key: ModuleKey,
        binding: SemanticBinding,
    ) -> ResolvedImportBinding:
        """
        title: Create one resolved import binding record.
        parameters:
          local_name:
            type: str
          requested_name:
            type: str
          source_module_key:
            type: ModuleKey
          binding:
            type: SemanticBinding
        returns:
          type: ResolvedImportBinding
        """
        return ResolvedImportBinding(
            local_name=local_name,
            requested_name=requested_name,
            source_module_key=source_module_key,
            binding=binding,
        )

# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=untyped-decorator

"""
Expression helpers for assignment and mutation targets.
"""

from __future__ import annotations

from typing import cast

from irx import astx
from irx.analysis.handlers.base import SemanticAnalyzerCore
from irx.analysis.handlers.class_helpers import (
    ClassMemberFormattingVisitorMixin,
)
from irx.analysis.resolved_nodes import (
    SemanticClassMember,
    SemanticInfo,
    SemanticSymbol,
)
from irx.analysis.validation import validate_assignment
from irx.diagnostics import DiagnosticCodes
from irx.typecheck import typechecked


@typechecked
class ExpressionMutationVisitorMixin(ClassMemberFormattingVisitorMixin):
    """
    title: Expression helpers for assignment and mutation targets
    """

    def _class_member_target_symbol(
        self,
        target: astx.AST,
        member: SemanticClassMember,
        target_type: astx.DataType | None,
    ) -> SemanticSymbol:
        """
        title: >-
          Build one variable-like symbol for a resolved class field target.
        parameters:
          target:
            type: astx.AST
          member:
            type: SemanticClassMember
          target_type:
            type: astx.DataType | None
        returns:
          type: SemanticSymbol
        """
        declaring_class = self._member_declaring_class(member)
        module_key = (
            declaring_class.module_key
            if declaring_class is not None
            else self._member_owner_module_key(member)
        )
        symbol_type = member.type_ or cast(astx.DataType, target_type)
        symbol_kind = (
            "class_static_field" if member.is_static else "class_field"
        )
        return self.factory.make_variable_symbol(
            module_key,
            self._member_display_name(member),
            symbol_type,
            is_mutable=member.is_mutable,
            declaration=target,
            kind=symbol_kind,
        )

    def _resolve_mutation_target(
        self,
        target: astx.AST,
        *,
        node: astx.AST,
        action: str,
        invalid_message: str,
    ) -> tuple[SemanticSymbol, str, astx.DataType | None] | None:
        """
        title: >-
          Resolve one direct mutable target for assignment-like operations.
        parameters:
          target:
            type: astx.AST
          node:
            type: astx.AST
          action:
            type: str
          invalid_message:
            type: str
        returns:
          type: tuple[SemanticSymbol, str, astx.DataType | None] | None
        """
        target_type = self._expr_type(target)
        if isinstance(target, astx.Identifier):
            symbol = cast(
                SemanticInfo, getattr(target, "semantic", SemanticInfo())
            ).resolved_symbol
            if symbol is None:
                self.context.diagnostics.add(
                    invalid_message,
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
                return None
            target_name = symbol.name
            if not symbol.is_mutable:
                self.context.diagnostics.add(
                    f"Cannot {action} '{target_name}': declared as constant",
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
            return symbol, target_name, target_type

        if isinstance(target, astx.FieldAccess):
            semantic = getattr(target, "semantic", None)
            class_field_access = getattr(
                semantic,
                "resolved_class_field_access",
                None,
            )
            if class_field_access is not None:
                member = class_field_access.member
                target_name = self._member_display_name(member)
                symbol = self._class_member_target_symbol(
                    target,
                    member,
                    target_type,
                )
                if not member.is_mutable:
                    message = (
                        f"Cannot {action} '{target_name}': "
                        "declared as constant"
                    )
                    self.context.diagnostics.add(
                        message,
                        node=node,
                        code=(
                            DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET
                        ),
                    )
                return symbol, target_name, target_type

            if isinstance(
                target.value,
                (astx.BaseFieldAccess, astx.StaticFieldAccess),
            ):
                self.context.diagnostics.add(
                    invalid_message.replace(
                        "a variable or field",
                        "a direct variable or field",
                    ),
                    node=node,
                    code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
                )
                return None

            symbol = self._root_assignment_symbol(target)
            target_name = target.field_name
            if symbol is None:
                self.context.diagnostics.add(
                    invalid_message,
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
                return None
            if not symbol.is_mutable:
                self.context.diagnostics.add(
                    f"Cannot {action} '{symbol.name}': declared as constant",
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
            return symbol, target_name, target_type

        if isinstance(target, astx.BaseFieldAccess):
            resolved_access = getattr(
                getattr(target, "semantic", None),
                "resolved_base_class_field_access",
                None,
            )
            if resolved_access is None:
                self.context.diagnostics.add(
                    invalid_message,
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
                return None
            member = resolved_access.member
            target_name = self._member_display_name(member)
            symbol = self._class_member_target_symbol(
                target,
                member,
                target_type,
            )
            if not member.is_mutable:
                self.context.diagnostics.add(
                    f"Cannot {action} '{target_name}': declared as constant",
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
            return symbol, target_name, target_type

        if isinstance(target, astx.StaticFieldAccess):
            resolved_access = getattr(
                getattr(target, "semantic", None),
                "resolved_static_class_field_access",
                None,
            )
            if resolved_access is None:
                self.context.diagnostics.add(
                    invalid_message,
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
                return None
            member = resolved_access.member
            target_name = self._member_display_name(member)
            symbol = self._class_member_target_symbol(
                target,
                member,
                target_type,
            )
            if not member.is_mutable:
                self.context.diagnostics.add(
                    f"Cannot {action} '{target_name}': declared as constant",
                    node=node,
                    code=(DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET),
                )
            return symbol, target_name, target_type

        self.context.diagnostics.add(
            invalid_message,
            node=node,
            code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
        )
        return None

    @SemanticAnalyzerCore.visit.dispatch
    def visit(self, node: astx.VariableAssignment) -> None:
        """
        title: Visit VariableAssignment nodes.
        parameters:
          node:
            type: astx.VariableAssignment
        """
        self.visit(node.value)
        symbol = self.context.scopes.resolve(node.name)
        if symbol is None:
            self.context.diagnostics.add(
                f"cannot assign to unresolved name '{node.name}'",
                node=node,
                code=DiagnosticCodes.SEMANTIC_UNRESOLVED_NAME,
            )
            return
        if not symbol.is_mutable:
            self.context.diagnostics.add(
                f"Cannot assign to '{node.name}': declared as constant",
                node=node,
                code=DiagnosticCodes.SEMANTIC_INVALID_ASSIGNMENT_TARGET,
            )
        if self._require_value_expression(
            node.value,
            context=f"Assignment to '{node.name}'",
        ):
            validate_assignment(
                self.context.diagnostics,
                target_name=node.name,
                target_type=symbol.type_,
                value_type=self._expr_type(node.value),
                node=node,
            )
        self._set_symbol(node, symbol)
        self._set_assignment(node, symbol)
        self._set_type(node, symbol.type_)

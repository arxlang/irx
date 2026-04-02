# mypy: disable-error-code=no-redef

"""
title: Shared Plum-dispatch visitor base for IRx.
"""

from __future__ import annotations

from plum import dispatch

from irx import astx


class BaseVisitor:
    """
    title: Concrete ASTx visitor scaffold with explicit not-implemented paths.
    """

    def _not_implemented(self, node: astx.AST) -> None:
        """
        title: Raise a consistent error for unsupported ASTx nodes.
        parameters:
          node:
            type: astx.AST
        raises:
          NotImplementedError: When the visitor does not implement the node.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.visit({type(node).__name__}) "
            "is not implemented"
        )

    @dispatch
    def visit(self, node: astx.AST) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ASTNodes) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AliasExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AndOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Argument) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Arguments) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AssignmentExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AsyncForRangeLoopExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AsyncForRangeLoopStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AugAssign) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AwaitExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.BinaryOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Block) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.BoolBinaryOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.BoolUnaryOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Boolean) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.BreakStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.CaseStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.CatchHandlerStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ClassDeclStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ClassDefStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.CollectionType) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.CompareOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Complex) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Complex32) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Complex64) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Comprehension) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ComprehensionClause) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ContinueStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DataType) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DataTypeOps) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Date) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DateTime) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DeleteStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DictType) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DoWhileExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DoWhileStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Ellipsis) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.EnumDeclStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ExceptionHandlerStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Expr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ExprType) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FinallyHandlerStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Float16) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Float32) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Float64) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Floating) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ForCountLoopExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ForCountLoopStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ForRangeLoopExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ForRangeLoopStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FunctionAsyncDef) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FunctionCall) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FunctionDef) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FunctionPrototype) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FunctionReturn) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.GeneratorExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.GotoStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Identifier) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.IfExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.IfStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ImportExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ImportFromExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ImportFromStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ImportStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.InlineVariableDeclaration) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Int16) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Int32) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Int64) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Int8) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Integer) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LambdaExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ListComprehension) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ListType) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Literal) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralBoolean) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralComplex) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralComplex32) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralComplex64) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralDate) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralDateTime) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralDict) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralFloat16) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralFloat32) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralFloat64) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralInt128) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralInt16) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralInt32) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralInt64) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralInt8) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralList) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralNone) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralSet) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralString) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralTime) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralTimestamp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralTuple) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUInt128) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUInt16) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUInt32) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUInt64) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUInt8) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUTF8Char) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUTF8String) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Module) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.NandOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.NoneType) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.NorOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.NotOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Number) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.OperatorType) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.OrOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Package) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ParenthesizedExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Program) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.SetComprehension) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.SetType) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.SignedInteger) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Starred) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.StatementType) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.String) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.StructDeclStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.StructDefStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.SubscriptExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.SwitchStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Target) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ThrowStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Time) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Timestamp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.TupleType) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.TypeCastExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UInt128) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UInt16) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UInt32) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UInt64) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UInt8) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UTF8Char) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UTF8String) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UnaryOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Undefined) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UnsignedInteger) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Variable) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.VariableAssignment) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.VariableDeclaration) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.WalrusOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.WhileExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.WhileStmt) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.XnorOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.XorOp) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.YieldExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.YieldFromExpr) -> None:
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.YieldStmt) -> None:
        self._not_implemented(node)

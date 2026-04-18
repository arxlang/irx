# mypy: disable-error-code=no-redef

"""
title: Shared Plum-dispatch visitor base for IRx.
"""

from __future__ import annotations

from plum import dispatch

from irx import astx
from irx.typecheck import typechecked


@typechecked
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
        """
        title: Visit AST nodes.
        parameters:
          node:
            type: astx.AST
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ASTNodes) -> None:
        """
        title: Visit ASTNodes nodes.
        parameters:
          node:
            type: astx.ASTNodes
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AliasExpr) -> None:
        """
        title: Visit AliasExpr nodes.
        parameters:
          node:
            type: astx.AliasExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AndOp) -> None:
        """
        title: Visit AndOp nodes.
        parameters:
          node:
            type: astx.AndOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Argument) -> None:
        """
        title: Visit Argument nodes.
        parameters:
          node:
            type: astx.Argument
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Arguments) -> None:
        """
        title: Visit Arguments nodes.
        parameters:
          node:
            type: astx.Arguments
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AssertStmt) -> None:
        """
        title: Visit AssertStmt nodes.
        parameters:
          node:
            type: astx.AssertStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AssignmentExpr) -> None:
        """
        title: Visit AssignmentExpr nodes.
        parameters:
          node:
            type: astx.AssignmentExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AsyncForRangeLoopExpr) -> None:
        """
        title: Visit AsyncForRangeLoopExpr nodes.
        parameters:
          node:
            type: astx.AsyncForRangeLoopExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AsyncForRangeLoopStmt) -> None:
        """
        title: Visit AsyncForRangeLoopStmt nodes.
        parameters:
          node:
            type: astx.AsyncForRangeLoopStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AugAssign) -> None:
        """
        title: Visit AugAssign nodes.
        parameters:
          node:
            type: astx.AugAssign
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.AwaitExpr) -> None:
        """
        title: Visit AwaitExpr nodes.
        parameters:
          node:
            type: astx.AwaitExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.BinaryOp) -> None:
        """
        title: Visit BinaryOp nodes.
        parameters:
          node:
            type: astx.BinaryOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Block) -> None:
        """
        title: Visit Block nodes.
        parameters:
          node:
            type: astx.Block
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.BoolBinaryOp) -> None:
        """
        title: Visit BoolBinaryOp nodes.
        parameters:
          node:
            type: astx.BoolBinaryOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.BoolUnaryOp) -> None:
        """
        title: Visit BoolUnaryOp nodes.
        parameters:
          node:
            type: astx.BoolUnaryOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Boolean) -> None:
        """
        title: Visit Boolean nodes.
        parameters:
          node:
            type: astx.Boolean
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.BreakStmt) -> None:
        """
        title: Visit BreakStmt nodes.
        parameters:
          node:
            type: astx.BreakStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.CaseStmt) -> None:
        """
        title: Visit CaseStmt nodes.
        parameters:
          node:
            type: astx.CaseStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.CatchHandlerStmt) -> None:
        """
        title: Visit CatchHandlerStmt nodes.
        parameters:
          node:
            type: astx.CatchHandlerStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ClassDeclStmt) -> None:
        """
        title: Visit ClassDeclStmt nodes.
        parameters:
          node:
            type: astx.ClassDeclStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ClassDefStmt) -> None:
        """
        title: Visit ClassDefStmt nodes.
        parameters:
          node:
            type: astx.ClassDefStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.CollectionType) -> None:
        """
        title: Visit CollectionType nodes.
        parameters:
          node:
            type: astx.CollectionType
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.CompareOp) -> None:
        """
        title: Visit CompareOp nodes.
        parameters:
          node:
            type: astx.CompareOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Complex) -> None:
        """
        title: Visit Complex nodes.
        parameters:
          node:
            type: astx.Complex
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Complex32) -> None:
        """
        title: Visit Complex32 nodes.
        parameters:
          node:
            type: astx.Complex32
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Complex64) -> None:
        """
        title: Visit Complex64 nodes.
        parameters:
          node:
            type: astx.Complex64
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Comprehension) -> None:
        """
        title: Visit Comprehension nodes.
        parameters:
          node:
            type: astx.Comprehension
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ComprehensionClause) -> None:
        """
        title: Visit ComprehensionClause nodes.
        parameters:
          node:
            type: astx.ComprehensionClause
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ContinueStmt) -> None:
        """
        title: Visit ContinueStmt nodes.
        parameters:
          node:
            type: astx.ContinueStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DataType) -> None:
        """
        title: Visit DataType nodes.
        parameters:
          node:
            type: astx.DataType
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DataTypeOps) -> None:
        """
        title: Visit DataTypeOps nodes.
        parameters:
          node:
            type: astx.DataTypeOps
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Date) -> None:
        """
        title: Visit Date nodes.
        parameters:
          node:
            type: astx.Date
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DateTime) -> None:
        """
        title: Visit DateTime nodes.
        parameters:
          node:
            type: astx.DateTime
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DeleteStmt) -> None:
        """
        title: Visit DeleteStmt nodes.
        parameters:
          node:
            type: astx.DeleteStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DictType) -> None:
        """
        title: Visit DictType nodes.
        parameters:
          node:
            type: astx.DictType
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DoWhileExpr) -> None:
        """
        title: Visit DoWhileExpr nodes.
        parameters:
          node:
            type: astx.DoWhileExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.DoWhileStmt) -> None:
        """
        title: Visit DoWhileStmt nodes.
        parameters:
          node:
            type: astx.DoWhileStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Ellipsis) -> None:
        """
        title: Visit Ellipsis nodes.
        parameters:
          node:
            type: astx.Ellipsis
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.EnumDeclStmt) -> None:
        """
        title: Visit EnumDeclStmt nodes.
        parameters:
          node:
            type: astx.EnumDeclStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ExceptionHandlerStmt) -> None:
        """
        title: Visit ExceptionHandlerStmt nodes.
        parameters:
          node:
            type: astx.ExceptionHandlerStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Expr) -> None:
        """
        title: Visit Expr nodes.
        parameters:
          node:
            type: astx.Expr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ExprType) -> None:
        """
        title: Visit ExprType nodes.
        parameters:
          node:
            type: astx.ExprType
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FinallyHandlerStmt) -> None:
        """
        title: Visit FinallyHandlerStmt nodes.
        parameters:
          node:
            type: astx.FinallyHandlerStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Float16) -> None:
        """
        title: Visit Float16 nodes.
        parameters:
          node:
            type: astx.Float16
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Float32) -> None:
        """
        title: Visit Float32 nodes.
        parameters:
          node:
            type: astx.Float32
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Float64) -> None:
        """
        title: Visit Float64 nodes.
        parameters:
          node:
            type: astx.Float64
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Floating) -> None:
        """
        title: Visit Floating nodes.
        parameters:
          node:
            type: astx.Floating
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ForCountLoopExpr) -> None:
        """
        title: Visit ForCountLoopExpr nodes.
        parameters:
          node:
            type: astx.ForCountLoopExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ForCountLoopStmt) -> None:
        """
        title: Visit ForCountLoopStmt nodes.
        parameters:
          node:
            type: astx.ForCountLoopStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ForRangeLoopExpr) -> None:
        """
        title: Visit ForRangeLoopExpr nodes.
        parameters:
          node:
            type: astx.ForRangeLoopExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ForRangeLoopStmt) -> None:
        """
        title: Visit ForRangeLoopStmt nodes.
        parameters:
          node:
            type: astx.ForRangeLoopStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FunctionAsyncDef) -> None:
        """
        title: Visit FunctionAsyncDef nodes.
        parameters:
          node:
            type: astx.FunctionAsyncDef
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FunctionCall) -> None:
        """
        title: Visit FunctionCall nodes.
        parameters:
          node:
            type: astx.FunctionCall
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FunctionDef) -> None:
        """
        title: Visit FunctionDef nodes.
        parameters:
          node:
            type: astx.FunctionDef
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FunctionPrototype) -> None:
        """
        title: Visit FunctionPrototype nodes.
        parameters:
          node:
            type: astx.FunctionPrototype
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.FunctionReturn) -> None:
        """
        title: Visit FunctionReturn nodes.
        parameters:
          node:
            type: astx.FunctionReturn
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.GeneratorExpr) -> None:
        """
        title: Visit GeneratorExpr nodes.
        parameters:
          node:
            type: astx.GeneratorExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.GotoStmt) -> None:
        """
        title: Visit GotoStmt nodes.
        parameters:
          node:
            type: astx.GotoStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Identifier) -> None:
        """
        title: Visit Identifier nodes.
        parameters:
          node:
            type: astx.Identifier
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.IfExpr) -> None:
        """
        title: Visit IfExpr nodes.
        parameters:
          node:
            type: astx.IfExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.IfStmt) -> None:
        """
        title: Visit IfStmt nodes.
        parameters:
          node:
            type: astx.IfStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ImportExpr) -> None:
        """
        title: Visit ImportExpr nodes.
        parameters:
          node:
            type: astx.ImportExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ImportFromExpr) -> None:
        """
        title: Visit ImportFromExpr nodes.
        parameters:
          node:
            type: astx.ImportFromExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ImportFromStmt) -> None:
        """
        title: Visit ImportFromStmt nodes.
        parameters:
          node:
            type: astx.ImportFromStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ImportStmt) -> None:
        """
        title: Visit ImportStmt nodes.
        parameters:
          node:
            type: astx.ImportStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.InlineVariableDeclaration) -> None:
        """
        title: Visit InlineVariableDeclaration nodes.
        parameters:
          node:
            type: astx.InlineVariableDeclaration
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Int16) -> None:
        """
        title: Visit Int16 nodes.
        parameters:
          node:
            type: astx.Int16
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Int32) -> None:
        """
        title: Visit Int32 nodes.
        parameters:
          node:
            type: astx.Int32
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Int64) -> None:
        """
        title: Visit Int64 nodes.
        parameters:
          node:
            type: astx.Int64
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Int8) -> None:
        """
        title: Visit Int8 nodes.
        parameters:
          node:
            type: astx.Int8
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Integer) -> None:
        """
        title: Visit Integer nodes.
        parameters:
          node:
            type: astx.Integer
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LambdaExpr) -> None:
        """
        title: Visit LambdaExpr nodes.
        parameters:
          node:
            type: astx.LambdaExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ListComprehension) -> None:
        """
        title: Visit ListComprehension nodes.
        parameters:
          node:
            type: astx.ListComprehension
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ListType) -> None:
        """
        title: Visit ListType nodes.
        parameters:
          node:
            type: astx.ListType
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Literal) -> None:
        """
        title: Visit Literal nodes.
        parameters:
          node:
            type: astx.Literal
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralBoolean) -> None:
        """
        title: Visit LiteralBoolean nodes.
        parameters:
          node:
            type: astx.LiteralBoolean
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralComplex) -> None:
        """
        title: Visit LiteralComplex nodes.
        parameters:
          node:
            type: astx.LiteralComplex
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralComplex32) -> None:
        """
        title: Visit LiteralComplex32 nodes.
        parameters:
          node:
            type: astx.LiteralComplex32
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralComplex64) -> None:
        """
        title: Visit LiteralComplex64 nodes.
        parameters:
          node:
            type: astx.LiteralComplex64
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralDate) -> None:
        """
        title: Visit LiteralDate nodes.
        parameters:
          node:
            type: astx.LiteralDate
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralDateTime) -> None:
        """
        title: Visit LiteralDateTime nodes.
        parameters:
          node:
            type: astx.LiteralDateTime
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralDict) -> None:
        """
        title: Visit LiteralDict nodes.
        parameters:
          node:
            type: astx.LiteralDict
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralFloat16) -> None:
        """
        title: Visit LiteralFloat16 nodes.
        parameters:
          node:
            type: astx.LiteralFloat16
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralFloat32) -> None:
        """
        title: Visit LiteralFloat32 nodes.
        parameters:
          node:
            type: astx.LiteralFloat32
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralFloat64) -> None:
        """
        title: Visit LiteralFloat64 nodes.
        parameters:
          node:
            type: astx.LiteralFloat64
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralInt128) -> None:
        """
        title: Visit LiteralInt128 nodes.
        parameters:
          node:
            type: astx.LiteralInt128
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralInt16) -> None:
        """
        title: Visit LiteralInt16 nodes.
        parameters:
          node:
            type: astx.LiteralInt16
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralInt32) -> None:
        """
        title: Visit LiteralInt32 nodes.
        parameters:
          node:
            type: astx.LiteralInt32
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralInt64) -> None:
        """
        title: Visit LiteralInt64 nodes.
        parameters:
          node:
            type: astx.LiteralInt64
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralInt8) -> None:
        """
        title: Visit LiteralInt8 nodes.
        parameters:
          node:
            type: astx.LiteralInt8
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralList) -> None:
        """
        title: Visit LiteralList nodes.
        parameters:
          node:
            type: astx.LiteralList
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralNone) -> None:
        """
        title: Visit LiteralNone nodes.
        parameters:
          node:
            type: astx.LiteralNone
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralSet) -> None:
        """
        title: Visit LiteralSet nodes.
        parameters:
          node:
            type: astx.LiteralSet
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralString) -> None:
        """
        title: Visit LiteralString nodes.
        parameters:
          node:
            type: astx.LiteralString
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralTime) -> None:
        """
        title: Visit LiteralTime nodes.
        parameters:
          node:
            type: astx.LiteralTime
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralTimestamp) -> None:
        """
        title: Visit LiteralTimestamp nodes.
        parameters:
          node:
            type: astx.LiteralTimestamp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralTuple) -> None:
        """
        title: Visit LiteralTuple nodes.
        parameters:
          node:
            type: astx.LiteralTuple
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUInt128) -> None:
        """
        title: Visit LiteralUInt128 nodes.
        parameters:
          node:
            type: astx.LiteralUInt128
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUInt16) -> None:
        """
        title: Visit LiteralUInt16 nodes.
        parameters:
          node:
            type: astx.LiteralUInt16
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUInt32) -> None:
        """
        title: Visit LiteralUInt32 nodes.
        parameters:
          node:
            type: astx.LiteralUInt32
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUInt64) -> None:
        """
        title: Visit LiteralUInt64 nodes.
        parameters:
          node:
            type: astx.LiteralUInt64
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUInt8) -> None:
        """
        title: Visit LiteralUInt8 nodes.
        parameters:
          node:
            type: astx.LiteralUInt8
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUTF8Char) -> None:
        """
        title: Visit LiteralUTF8Char nodes.
        parameters:
          node:
            type: astx.LiteralUTF8Char
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.LiteralUTF8String) -> None:
        """
        title: Visit LiteralUTF8String nodes.
        parameters:
          node:
            type: astx.LiteralUTF8String
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Module) -> None:
        """
        title: Visit Module nodes.
        parameters:
          node:
            type: astx.Module
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.NandOp) -> None:
        """
        title: Visit NandOp nodes.
        parameters:
          node:
            type: astx.NandOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.NoneType) -> None:
        """
        title: Visit NoneType nodes.
        parameters:
          node:
            type: astx.NoneType
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.NorOp) -> None:
        """
        title: Visit NorOp nodes.
        parameters:
          node:
            type: astx.NorOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.NotOp) -> None:
        """
        title: Visit NotOp nodes.
        parameters:
          node:
            type: astx.NotOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Number) -> None:
        """
        title: Visit Number nodes.
        parameters:
          node:
            type: astx.Number
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.OperatorType) -> None:
        """
        title: Visit OperatorType nodes.
        parameters:
          node:
            type: astx.OperatorType
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.OrOp) -> None:
        """
        title: Visit OrOp nodes.
        parameters:
          node:
            type: astx.OrOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Package) -> None:
        """
        title: Visit Package nodes.
        parameters:
          node:
            type: astx.Package
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ParenthesizedExpr) -> None:
        """
        title: Visit ParenthesizedExpr nodes.
        parameters:
          node:
            type: astx.ParenthesizedExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Program) -> None:
        """
        title: Visit Program nodes.
        parameters:
          node:
            type: astx.Program
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.SetComprehension) -> None:
        """
        title: Visit SetComprehension nodes.
        parameters:
          node:
            type: astx.SetComprehension
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.SetType) -> None:
        """
        title: Visit SetType nodes.
        parameters:
          node:
            type: astx.SetType
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.SignedInteger) -> None:
        """
        title: Visit SignedInteger nodes.
        parameters:
          node:
            type: astx.SignedInteger
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Starred) -> None:
        """
        title: Visit Starred nodes.
        parameters:
          node:
            type: astx.Starred
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.StatementType) -> None:
        """
        title: Visit StatementType nodes.
        parameters:
          node:
            type: astx.StatementType
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.String) -> None:
        """
        title: Visit String nodes.
        parameters:
          node:
            type: astx.String
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.StructDeclStmt) -> None:
        """
        title: Visit StructDeclStmt nodes.
        parameters:
          node:
            type: astx.StructDeclStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.StructDefStmt) -> None:
        """
        title: Visit StructDefStmt nodes.
        parameters:
          node:
            type: astx.StructDefStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.SubscriptExpr) -> None:
        """
        title: Visit SubscriptExpr nodes.
        parameters:
          node:
            type: astx.SubscriptExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.SwitchStmt) -> None:
        """
        title: Visit SwitchStmt nodes.
        parameters:
          node:
            type: astx.SwitchStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Target) -> None:
        """
        title: Visit Target nodes.
        parameters:
          node:
            type: astx.Target
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.ThrowStmt) -> None:
        """
        title: Visit ThrowStmt nodes.
        parameters:
          node:
            type: astx.ThrowStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Time) -> None:
        """
        title: Visit Time nodes.
        parameters:
          node:
            type: astx.Time
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Timestamp) -> None:
        """
        title: Visit Timestamp nodes.
        parameters:
          node:
            type: astx.Timestamp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.TupleType) -> None:
        """
        title: Visit TupleType nodes.
        parameters:
          node:
            type: astx.TupleType
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.TypeCastExpr) -> None:
        """
        title: Visit TypeCastExpr nodes.
        parameters:
          node:
            type: astx.TypeCastExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UInt128) -> None:
        """
        title: Visit UInt128 nodes.
        parameters:
          node:
            type: astx.UInt128
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UInt16) -> None:
        """
        title: Visit UInt16 nodes.
        parameters:
          node:
            type: astx.UInt16
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UInt32) -> None:
        """
        title: Visit UInt32 nodes.
        parameters:
          node:
            type: astx.UInt32
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UInt64) -> None:
        """
        title: Visit UInt64 nodes.
        parameters:
          node:
            type: astx.UInt64
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UInt8) -> None:
        """
        title: Visit UInt8 nodes.
        parameters:
          node:
            type: astx.UInt8
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UTF8Char) -> None:
        """
        title: Visit UTF8Char nodes.
        parameters:
          node:
            type: astx.UTF8Char
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UTF8String) -> None:
        """
        title: Visit UTF8String nodes.
        parameters:
          node:
            type: astx.UTF8String
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UnaryOp) -> None:
        """
        title: Visit UnaryOp nodes.
        parameters:
          node:
            type: astx.UnaryOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Undefined) -> None:
        """
        title: Visit Undefined nodes.
        parameters:
          node:
            type: astx.Undefined
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.UnsignedInteger) -> None:
        """
        title: Visit UnsignedInteger nodes.
        parameters:
          node:
            type: astx.UnsignedInteger
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.Variable) -> None:
        """
        title: Visit Variable nodes.
        parameters:
          node:
            type: astx.Variable
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.VariableAssignment) -> None:
        """
        title: Visit VariableAssignment nodes.
        parameters:
          node:
            type: astx.VariableAssignment
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.VariableDeclaration) -> None:
        """
        title: Visit VariableDeclaration nodes.
        parameters:
          node:
            type: astx.VariableDeclaration
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.WalrusOp) -> None:
        """
        title: Visit WalrusOp nodes.
        parameters:
          node:
            type: astx.WalrusOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.WhileExpr) -> None:
        """
        title: Visit WhileExpr nodes.
        parameters:
          node:
            type: astx.WhileExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.WhileStmt) -> None:
        """
        title: Visit WhileStmt nodes.
        parameters:
          node:
            type: astx.WhileStmt
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.XnorOp) -> None:
        """
        title: Visit XnorOp nodes.
        parameters:
          node:
            type: astx.XnorOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.XorOp) -> None:
        """
        title: Visit XorOp nodes.
        parameters:
          node:
            type: astx.XorOp
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.YieldExpr) -> None:
        """
        title: Visit YieldExpr nodes.
        parameters:
          node:
            type: astx.YieldExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.YieldFromExpr) -> None:
        """
        title: Visit YieldFromExpr nodes.
        parameters:
          node:
            type: astx.YieldFromExpr
        """
        self._not_implemented(node)

    @dispatch
    def visit(self, node: astx.YieldStmt) -> None:
        """
        title: Visit YieldStmt nodes.
        parameters:
          node:
            type: astx.YieldStmt
        """
        self._not_implemented(node)

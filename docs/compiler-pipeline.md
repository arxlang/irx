# Compiler Pipeline

IRx now treats semantic analysis as a first-class phase between raw ASTx nodes
and LLVM lowering:

`AST -> semantic analysis -> resolved AST sidecars -> LLVM IR codegen`

Semantic analysis is responsible for meaning and validity. It resolves symbols,
tracks scopes, validates mutability and control-flow legality, normalizes
operator flags such as unsigned and fast-math intent, and attaches the results
to `node.semantic`. Codegen is then free to focus on emitting LLVM IR from
already-analyzed nodes instead of re-inventing semantics during lowering.

`visit(self, node: ...)` remains the only public multidispatch boundary for LLVM
lowering. We intentionally did not replace it with a free-function registry or a
second lowering API, because the method-based Plum dispatch is still the
clearest public surface for AST-family-specific codegen.

The `llvmliteir` package keeps foundational modules such as `core.py`,
`protocols.py`, `types.py`, and `vector.py` at the package root because they are
architectural building blocks, not incidental helpers. Backend packages should
expose short generic names such as `Builder`, `Visitor`, `VisitorProtocol`, and
an optional `_VisitorCore`; the package path itself identifies which backend is
in use. The protocol gives mixins and runtime feature declarations a stable
typing contract, while the concrete core class owns the shared mutable lowering
state and lifecycle.

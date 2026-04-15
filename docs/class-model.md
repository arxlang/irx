# Class Model

IRx now has a first-class low-level class system designed for Arx to target
without turning IRx into a high-level object runtime.

## What Classes Add Beyond Structs

- `StructDefStmt` remains the ABI-oriented passive composite type
- `ClassDefStmt` adds inheritance, methods, access control, static members,
  deterministic dispatch metadata, and deterministic construction/lowering
- class values lower as pointers to identified object structs with reserved
  header slots; struct values remain plain by-value composites unless another
  rule says otherwise

## Supported Modifiers

IRx currently supports these class-member modifiers:

- `public`
- `protected`
- `private`
- `static`
- `constant`
- `mutable`

Current semantics:

- `public` is accessible wherever the containing class value is visible
- `protected` is accessible only inside the declaring class and subclasses
- `private` is accessible only inside the declaring class
- `static` members use class-qualified access and class-scoped global storage
- `constant` members are immutable after initialization
- `mutable` members may be reassigned after initialization

## Inheritance And MRO

IRx supports multiple inheritance with deterministic C3 linearization.

- method lookup follows C3 MRO order
- ambiguous inherited attributes are rejected
- conflicting inherited methods with the same effective signature are rejected
  unless the subclass supplies a legal override
- shared diamond ancestors are stored once in the canonical flattened layout
- implicit ancestor field views are still deferred; explicit base-qualified
  access is the supported form in this phase

## Layout And Dispatch

- every class object reserves a type-descriptor slot and a dispatch-table slot
- instance fields are flattened in canonical ancestor-first order
- static attributes lower to internal module globals
- instance methods lower to ordinary functions with a hidden receiver parameter
- non-private overridable instance methods use stable dispatch slots
- static methods lower to direct calls with no hidden receiver

## Static Members, Constants, And Mutability

- `StaticFieldAccess("Name", "field")` is the class-qualified read/write form
  for static attributes
- static mutable fields lower to `internal global` storage
- static constant fields lower to `internal constant` storage when possible
- constant instance, base-qualified, and static members reject assignment and
  unary mutation during semantic analysis
- static initialization remains literal/default-only in the current phase

## Access Control

- instance access uses `FieldAccess`
- class-qualified static access uses `StaticFieldAccess`
- explicit legal ancestor access uses `BaseFieldAccess` and `BaseMethodCall`
- access control is enforced during semantic analysis, never deferred to
  lowering or runtime

## ABI And Interop

- IRx-defined functions pass and return class values by pointer
- class methods and dispatch globals use deterministic internal symbol names
- class static storage uses deterministic internal global names
- general classes are **not** part of IRx's stable public foreign ABI yet
- foreign boundaries should still use structs, typed pointers, and opaque
  handles rather than `ClassType`

## Current Limitations

- no high-level constructor bodies yet; `ClassConstruct("Name")` is the current
  low-level construction form
- no stable foreign object ABI for general classes
- no implicit ancestor field views beyond explicit base-qualified access
- no callback/function-pointer class interop in the public FFI layer
- overload selection requires exact matches; conversion-ranked method overloads
  remain deferred

## Examples

### Instance Field Plus Method

```python
read_body = astx.Block()
read_body.append(
    astx.FunctionReturn(
        astx.FieldAccess(astx.Identifier("self"), "value")
    )
)
counter = astx.ClassDefStmt(
    name="Counter",
    attributes=[
        astx.VariableDeclaration(name="value", type_=astx.Int32()),
    ],
    methods=[
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                name="read",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=read_body,
        )
    ],
)
```

### Base/Derived Override

```python
base_area_body = astx.Block()
base_area_body.append(astx.FunctionReturn(astx.LiteralInt32(1)))
child_area_body = astx.Block()
child_area_body.append(astx.FunctionReturn(astx.LiteralInt32(2)))
measure_body = astx.Block()
measure_body.append(
    astx.FunctionReturn(
        astx.MethodCall(astx.Identifier("shape"), "area", [])
    )
)
base = astx.ClassDefStmt(
    name="Base",
    methods=[
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                name="area",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=base_area_body,
        )
    ],
)
child = astx.ClassDefStmt(
    name="Child",
    bases=[astx.ClassType("Base")],
    methods=[
        astx.FunctionDef(
            prototype=astx.FunctionPrototype(
                name="area",
                args=astx.Arguments(),
                return_type=astx.Int32(),
            ),
            body=child_area_body,
        )
    ],
)
measure = astx.FunctionDef(
    prototype=astx.FunctionPrototype(
        name="measure",
        args=astx.Arguments(astx.Argument("shape", astx.ClassType("Base"))),
        return_type=astx.Int32(),
    ),
    body=measure_body,
)
```

### Multiple Inheritance

```python
root = astx.ClassDefStmt(
    name="Root",
    attributes=[astx.VariableDeclaration(name="root", type_=astx.Int32())],
)
left = astx.ClassDefStmt(
    name="Left",
    bases=[astx.ClassType("Root")],
    attributes=[astx.VariableDeclaration(name="left", type_=astx.Boolean())],
)
right = astx.ClassDefStmt(
    name="Right",
    bases=[astx.ClassType("Root")],
    attributes=[astx.VariableDeclaration(name="right", type_=astx.Float64())],
)
child = astx.ClassDefStmt(
    name="Child",
    bases=[astx.ClassType("Left"), astx.ClassType("Right")],
)
```

The effective MRO is `Child -> Left -> Right -> Root`, and the shared `Root`
subobject appears once in the flattened layout.

### Static Constant Example

```python
limit = astx.VariableDeclaration(
    name="limit",
    type_=astx.Int32(),
    mutability=astx.MutabilityKind.constant,
    scope=astx.ScopeKind.global_,
    value=astx.LiteralInt32(99),
)
limit.is_static = True
counter = astx.ClassDefStmt(
    name="Counter",
    attributes=[limit],
)
read_limit = astx.StaticFieldAccess("Counter", "limit")
```

For the normative lowering and semantic details, see
[`docs/semantic-contract.md`](./semantic-contract.md).

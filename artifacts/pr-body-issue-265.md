Description

Implemented the fix for `BinaryOp` handling `"="` overlapping with `VariableAssignment`, so assignment is now handled only through `VariableAssignment`.

Removed assignment handling from `BinaryOp` in `src/irx/builders/llvmliteir.py`.
Added a safety guard so `BinaryOp("=")` now raises:
`Assignment '=' should not be handled in BinaryOp. Use VariableAssignment instead.`
Kept assignment handling in `VariableAssignment`, which remains the only valid assignment path.
Added a regression test in `tests/test_binary_op.py` to verify `BinaryOp("=")` is rejected.
Added a regression test in `tests/test_variable_assignment.py` to verify `VariableAssignment` still updates the variable correctly and `print(x)` works.

This is a codegen-only change and does not alter parser/frontend behavior.

Fixes: #265

Comments

Please verify that assignment is only handled through `VariableAssignment`.
Ensure other binary operators like `+`, `-`, `*`, `/`, and comparisons still behave normally.
The `BinaryOp("=")` guard is intentional to prevent duplicated codegen paths and preserve the intended language design.

Checklist:

- [x] The code follows the project style.
- [x] Regression tests were added.
- [ ] Tests were fully executed locally.

Testing note:
Local automated test execution is still blocked on this machine by environment/dependency setup issues, not by a known code failure.

Before

![Before](https://raw.githubusercontent.com/dhruvv16-hash/irx/codex/fix-binaryop-assignment-overlap/artifacts/before-binaryop-assignment.png)

After

![After](https://raw.githubusercontent.com/dhruvv16-hash/irx/codex/fix-binaryop-assignment-overlap/artifacts/after-binaryop-assignment.png)

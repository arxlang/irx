# Release Notes
---

# [1.14.0](https://github.com/arxlang/irx/compare/1.13.0...1.14.0) (2026-04-21)


### Features

* Set arrow as builtin backend for arrays ([#304](https://github.com/arxlang/irx/issues/304)) ([6bf5e3f](https://github.com/arxlang/irx/commit/6bf5e3f42c2377126575a15906d6b78a4311db32))

# [1.13.0](https://github.com/arxlang/irx/compare/1.12.0...1.13.0) (2026-04-20)


### Features

* Implement function/method template ([#302](https://github.com/arxlang/irx/issues/302)) ([8b005fc](https://github.com/arxlang/irx/commit/8b005fc9592e8397d1c3beb4387f0242bbe47db1))
* Improve import module from package ([#301](https://github.com/arxlang/irx/issues/301)) ([6218f69](https://github.com/arxlang/irx/commit/6218f69d26ad6c3232b45a44c41434cfa5745d2d))

# [1.12.0](https://github.com/arxlang/irx/compare/1.11.0...1.12.0) (2026-04-19)


### Features

* Improve module handling ([#300](https://github.com/arxlang/irx/issues/300)) ([c777251](https://github.com/arxlang/irx/commit/c777251f814a4f8509cd65dc9668b2bc65f45776))

# [1.11.0](https://github.com/arxlang/irx/compare/1.10.0...1.11.0) (2026-04-18)


### Features

* **irx:** add fatal AssertStmt lowering and assertion runtime ([#299](https://github.com/arxlang/irx/issues/299)) ([186c2fe](https://github.com/arxlang/irx/commit/186c2fede124c044126c7f232a866feed7a8cbd9))

# [1.10.0](https://github.com/arxlang/irx/compare/1.9.0...1.10.0) (2026-04-15)


### Features

* Add FFI as a first-class public layer ([#285](https://github.com/arxlang/irx/issues/285)) ([bd48dd5](https://github.com/arxlang/irx/commit/bd48dd5dad5c2e6d0e4782e35e97af9d55072569))
* add first-class buffer view indexing lowering ([#281](https://github.com/arxlang/irx/issues/281)) ([341e36e](https://github.com/arxlang/irx/commit/341e36e09d34c283a0679b15aec6934d22adc7f3))
* Add initial support for classes ([#290](https://github.com/arxlang/irx/issues/290)) ([558b05c](https://github.com/arxlang/irx/commit/558b05cd4a41bf51634fd6db57ed4aff9924e4da))
* Add support for import and import from features ([#273](https://github.com/arxlang/irx/issues/273)) ([dcbe85c](https://github.com/arxlang/irx/commit/dcbe85cbe7345c735781bc0be490fb1fb1bda76f))
* Complete the scalar numeric foundation ([#277](https://github.com/arxlang/irx/issues/277)) ([b2a8b55](https://github.com/arxlang/irx/commit/b2a8b55a9e7d5df86f562d78cb7262e4348d27f4))
* Define the buffer/view model ([#280](https://github.com/arxlang/irx/issues/280)) ([f3a8fb4](https://github.com/arxlang/irx/commit/f3a8fb4815d385020941af47b0d881b5b5147e79))
* Expand Arrow from MVP to usable scientific interop substrate ([#286](https://github.com/arxlang/irx/issues/286)) ([0a415b0](https://github.com/arxlang/irx/commit/0a415b0dfdfc09803465049b78f4cb6e8972e98f))
* Formalize the IRx semantic contract ([#276](https://github.com/arxlang/irx/issues/276)) ([d1157ad](https://github.com/arxlang/irx/commit/d1157ad5d14145881a26dafb925259df9283faaa))
* Implement class improvement (Stages 2, 3, 4) ([#291](https://github.com/arxlang/irx/issues/291)) ([531d81b](https://github.com/arxlang/irx/commit/531d81be7b64444e1e6e72ab8e7254b5dd00f8b4))
* Improve classes - Access control enforcement ([#292](https://github.com/arxlang/irx/issues/292)) ([7767553](https://github.com/arxlang/irx/commit/7767553c8fa0524cc757d32132e148d43164b569))
* Improve Classes - Member access and lowering ([#294](https://github.com/arxlang/irx/issues/294)) ([80fdb51](https://github.com/arxlang/irx/commit/80fdb516f4a40983087b379380933f34d8dd0e12))
* Improve Classes - Static storage, constants, and mutability ([#295](https://github.com/arxlang/irx/issues/295)) ([71bff44](https://github.com/arxlang/irx/commit/71bff4425272a0851794ec027f339923a7f09968))
* Improve Classes: Define Construction and initialization ([#293](https://github.com/arxlang/irx/issues/293)) ([5d1e708](https://github.com/arxlang/irx/commit/5d1e70862612d3b9b3f0bdbb91fed3ba17629da6))
* Improve diagnostics and source mapping ([#287](https://github.com/arxlang/irx/issues/287)) ([28b04a1](https://github.com/arxlang/irx/commit/28b04a1b247ae13eea60d3e4d2fda7d42620b311))
* Loop and control-flow hardening for numeric kernels ([#288](https://github.com/arxlang/irx/issues/288)) ([9e6a975](https://github.com/arxlang/irx/commit/9e6a975154d33121243f7ad2bc766902ebe40ff3))
* Make booleans first-class across semantics and lowering ([#278](https://github.com/arxlang/irx/issues/278)) ([7dafa90](https://github.com/arxlang/irx/commit/7dafa90a1f27167cd0b73356efdcbd4561486964))
* **semantics:** harden function signatures and calling conventions ([#284](https://github.com/arxlang/irx/issues/284)) ([59b6758](https://github.com/arxlang/irx/commit/59b67581489de11ed78e8394e1d98a19823fb376))
* Stabilize structs/records as the base composite type ([#279](https://github.com/arxlang/irx/issues/279)) ([dfa3eeb](https://github.com/arxlang/irx/commit/dfa3eeba2f221039464d05d1f47ed28ae179a620))

# [1.9.0](https://github.com/arxlang/irx/compare/1.8.1...1.9.0) (2026-04-02)


### Bug Fixes

* **codegen:** use safe_pop for loop body lowering; use fadd/fsub for float unary inc and dec ([#264](https://github.com/arxlang/irx/issues/264)) ([326a0db](https://github.com/arxlang/irx/commit/326a0dbe4404e4d2251c43cc912183c9c628689f))
* **compiler:** use fadd instead of add for float loop variables ([#219](https://github.com/arxlang/irx/issues/219)) ([2922af3](https://github.com/arxlang/irx/commit/2922af349b54d535171805cc202cb036c6cdfdbb)), closes [#217](https://github.com/arxlang/irx/issues/217)
* default vector int division to signed to match scalar behavior ([#205](https://github.com/arxlang/irx/issues/205)) ([a9bbc72](https://github.com/arxlang/irx/commit/a9bbc72252c90a86b774eb7a5d300c646898e821))
* FunctionDef does not emit a return for void functions ([#252](https://github.com/arxlang/irx/issues/252)) ([d890f24](https://github.com/arxlang/irx/commit/d890f246bcfc5b6086dec5beb0b4fb0f4a31ce8d))


### Features

* **#247:** implement set operations (union, intersection, difference, symmetric difference) ([#254](https://github.com/arxlang/irx/issues/254)) ([2cbe8f7](https://github.com/arxlang/irx/commit/2cbe8f760e95a3f0972f6146db4fd1047b9219a0)), closes [#247](https://github.com/arxlang/irx/issues/247)
* **#249:** implement dict key lookup via SubscriptExpr visitor ([#253](https://github.com/arxlang/irx/issues/253)) ([e871553](https://github.com/arxlang/irx/commit/e871553dc90182506eab6511702b43de3e7b001b)), closes [#249](https://github.com/arxlang/irx/issues/249)
* add BreakStmt and ContinueStmt support in IRx ([#259](https://github.com/arxlang/irx/issues/259)) ([da55ff2](https://github.com/arxlang/irx/commit/da55ff261d720f8fdc9c1ceb5797bf9f873cbe8c))
* Add initial support for Arrow ([#232](https://github.com/arxlang/irx/issues/232)) ([ca1a302](https://github.com/arxlang/irx/commit/ca1a30203151c79a713b3e1928d65ea495df5bd3))
* add LLVM IR codegen for StructDefStmt and tests ([#201](https://github.com/arxlang/irx/issues/201)) ([9d2498c](https://github.com/arxlang/irx/commit/9d2498cb116c261a5101a2f7f6009d305311c5c5))
* add minimal LiteralTuple lowering support ([#225](https://github.com/arxlang/irx/issues/225)) ([9b1f2c6](https://github.com/arxlang/irx/commit/9b1f2c65be2d31d5057b6fd27ed3f68695ed62bb))
* Add structured CommandResult and error handling for run_command ([#215](https://github.com/arxlang/irx/issues/215)) ([b40b478](https://github.com/arxlang/irx/commit/b40b47889cbe4de9ff498a2c6239c1d4cb5339d8))
* extend LiteralSet lowering to support mixed-width integer constants ([#231](https://github.com/arxlang/irx/issues/231)) ([56454ff](https://github.com/arxlang/irx/commit/56454ff3b07dc03624cf95a0e9b6c808e2a3a833))
* minimal LiteralDict lowering support ([#174](https://github.com/arxlang/irx/issues/174)) ([82675e4](https://github.com/arxlang/irx/commit/82675e4490bb4e7139755e14f1dfb28831d68744))

## [1.8.1](https://github.com/arxlang/irx/compare/1.8.0...1.8.1) (2026-03-06)


### Bug Fixes

* Update astx; mermaid-ascii is optional now ([#206](https://github.com/arxlang/irx/issues/206)) ([511efba](https://github.com/arxlang/irx/commit/511efbac43b16af56d6a6179ffdc62f3dd6439b5))

# [1.8.0](https://github.com/arxlang/irx/compare/1.7.0...1.8.0) (2026-03-06)


### Bug Fixes

* Add option for -no-pie compilation ([#203](https://github.com/arxlang/irx/issues/203)) ([168edd9](https://github.com/arxlang/irx/commit/168edd9e9f84079a24ecb375aa40c17945fdb6fb))


### Features

* add LiteralTime support ([5c88e1e](https://github.com/arxlang/irx/commit/5c88e1e1abaa9286044c2973b51c8fb75a339498))

# [1.7.0](https://github.com/arxlang/irx/compare/1.6.0...1.7.0) (2026-03-06)


### Bug Fixes

* Fix If statement, add tests, add support for python 3.14 ([#195](https://github.com/arxlang/irx/issues/195)) ([a5c4b5d](https://github.com/arxlang/irx/commit/a5c4b5d7151ad6d8113df66c2738711fa053fb4b))
* Fix print for numeric ([#196](https://github.com/arxlang/irx/issues/196)) ([620afbe](https://github.com/arxlang/irx/commit/620afbe1422d5f8f3938e36e700f034fc0d9aeaf))


### Features

* enforce mutability checking for variable declarations in LLVM-IR builder ([#192](https://github.com/arxlang/irx/issues/192)) ([f1f4036](https://github.com/arxlang/irx/commit/f1f4036904d1c70c4c4d6f3689f882f0be5027b7))

# [1.6.0](https://github.com/arxlang/irx/compare/1.5.0...1.6.0) (2026-03-05)


### Bug Fixes

* ensure ForRangeLoopStmt condition is checked before loop body execution ([#178](https://github.com/arxlang/irx/issues/178)) ([6bb3b7f](https://github.com/arxlang/irx/commit/6bb3b7fd20f2dde7f37dc862f4f8a7fb7dd949b0))
* make fast-math toggles work ([#140](https://github.com/arxlang/irx/issues/140)) ([67f3668](https://github.com/arxlang/irx/commit/67f3668ea5a59b1b5d7bcea4cd12001f53a93983))
* set module triple/data layout; drop forced 64-bit size_t override ([#134](https://github.com/arxlang/irx/issues/134)) ([4361689](https://github.com/arxlang/irx/commit/4361689ca8f3b9fdd681a06fcfffda03a1a47902))


### Features

* add minimal LiteralList lowering and tests ([#128](https://github.com/arxlang/irx/issues/128)) ([3683488](https://github.com/arxlang/irx/commit/3683488bf912a191a2710fc0d2c9faefca27bcc3))
* Fix incomplete float type detection in binary operations ([#146](https://github.com/arxlang/irx/issues/146)) ([290d1f1](https://github.com/arxlang/irx/commit/290d1f1b9c0244c915f58acb4a8fafd0c44952bc)), closes [#137](https://github.com/arxlang/irx/issues/137)

# [1.5.0](https://github.com/arxlang/irx/compare/1.4.0...1.5.0) (2025-11-30)


### Bug Fixes

* replace alloca with malloc in strcat to avoid use-after-return ([#131](https://github.com/arxlang/irx/issues/131)) ([eb7886f](https://github.com/arxlang/irx/commit/eb7886f24919a67bf96d6e332cbe59e729aedb83))
* use hybrid version approach with importlib and fallback ([#142](https://github.com/arxlang/irx/issues/142)) ([754ef42](https://github.com/arxlang/irx/commit/754ef429b25326c8b59f6136296640e65e2cbea6))
* zero-extend i1 boolean for correct string printing ([#123](https://github.com/arxlang/irx/issues/123)) ([9177426](https://github.com/arxlang/irx/commit/9177426ab17e2520a5b7df92d11a4f4879bb23ab))


### Features

* Add casting numeric to string ([#93](https://github.com/arxlang/irx/issues/93)) ([c4e8953](https://github.com/arxlang/irx/commit/c4e89537d5135db836348dfcf53d3f727c04be97))
* add LiteralDateTime  ([#124](https://github.com/arxlang/irx/issues/124)) ([7bd8f3e](https://github.com/arxlang/irx/commit/7bd8f3e85c5c5a374460cad0a3ee9124ba84a712))
* Feature/literal timestamp v2 ([#120](https://github.com/arxlang/irx/issues/120)) ([cdbcee9](https://github.com/arxlang/irx/commit/cdbcee966b0cdac5b659121299e544100801ae0e))
* guard against negative snprintf return in snprintf_heap ([#122](https://github.com/arxlang/irx/issues/122)) ([9635e59](https://github.com/arxlang/irx/commit/9635e5950d41fc31e2530b5f177156de84f27fe4))

# [1.4.0](https://github.com/arxlang/irx/compare/1.3.1...1.4.0) (2025-08-20)


### Bug Fixes

* Update ASTx to version 0.23.0 ([#82](https://github.com/arxlang/irx/issues/82)) ([f20bd26](https://github.com/arxlang/irx/commit/f20bd26d32b9e4340dbb94de811be861a8d34045))


### Features

* Add support for String ([#73](https://github.com/arxlang/irx/issues/73)) ([f2135ac](https://github.com/arxlang/irx/commit/f2135acb5ae1b12694df3ed79c9aef381f5d2ad6))
* Replace Variable by Identifier ([#83](https://github.com/arxlang/irx/issues/83)) ([0f4ee4b](https://github.com/arxlang/irx/commit/0f4ee4bfd5d545925e55d3fa61cacca788393e59))

## [1.3.1](https://github.com/arxlang/irx/compare/1.3.0...1.3.1) (2025-08-07)


### Bug Fixes

* Update usage of FunctionCall ([#80](https://github.com/arxlang/irx/issues/80)) ([3de9cd6](https://github.com/arxlang/irx/commit/3de9cd66d9365d3dcf780efe13ef080383d47012))

# [1.3.0](https://github.com/arxlang/irx/compare/1.2.1...1.3.0) (2025-08-06)


### Bug Fixes

*  Reset the names_values in LLVMLiteIRVisitor ([#66](https://github.com/arxlang/irx/issues/66)) ([9bcc3e6](https://github.com/arxlang/irx/commit/9bcc3e65a705095e26a24c7eafd78cc5bd7a6c45))
* Fix ForCount Implementation and its tests ([#41](https://github.com/arxlang/irx/issues/41)) ([3d36157](https://github.com/arxlang/irx/commit/3d36157af9ea5f5c394b0b259cfac09975321fb7))
* Fix incorrect result handling in Block visitor ([#62](https://github.com/arxlang/irx/issues/62)) ([538e031](https://github.com/arxlang/irx/commit/538e031f7cad0b00b954f78d9a31555304868206))
* Fix issues with symbol table and add test cases for it ([#43](https://github.com/arxlang/irx/issues/43)) ([a769d79](https://github.com/arxlang/irx/commit/a769d7931007cb9cead6bc08437da7877cb5c50e))
* Fix typing with clang ([#70](https://github.com/arxlang/irx/issues/70)) ([1936259](https://github.com/arxlang/irx/commit/19362599130babdd18427cfebd4289fcf540f141))
* ForRangeLoop now checks loop var < end, not end != 0 ([#61](https://github.com/arxlang/irx/issues/61)) ([aad8c2d](https://github.com/arxlang/irx/commit/aad8c2de86b4212b9ffd49cfafbac2c65fd37988))
* Rename Function to FunctionDef ([#75](https://github.com/arxlang/irx/issues/75)) ([da95f82](https://github.com/arxlang/irx/commit/da95f824e42c0ef775f603117d3b105e0188dc22))
* Update astx dependency; improve tests for void ([#79](https://github.com/arxlang/irx/issues/79)) ([3e147f3](https://github.com/arxlang/irx/commit/3e147f3b4e1fb5031e10d75832c39cc28d895a10))


### Features

* Add casting to use printexpr with integers ([#67](https://github.com/arxlang/irx/issues/67)) ([d3ab378](https://github.com/arxlang/irx/commit/d3ab378890a2837198ed2a210d05b61e4cf75580))
* Add support for boolean ([#74](https://github.com/arxlang/irx/issues/74)) ([34d3f4b](https://github.com/arxlang/irx/commit/34d3f4bdb61020955683670c4894fee93a5beea3))
* add support for float16 ([#77](https://github.com/arxlang/irx/issues/77)) ([6fe9ece](https://github.com/arxlang/irx/commit/6fe9ece456f3d2adc9b88cfff78232cf2134f88d))
* Add support for int64 ([#72](https://github.com/arxlang/irx/issues/72)) ([d6d0702](https://github.com/arxlang/irx/commit/d6d07026734f5bbd92c84aadf40f4096e99b9e4a))
* Add support for int8 datatype ([#71](https://github.com/arxlang/irx/issues/71)) ([6748c57](https://github.com/arxlang/irx/commit/6748c57f0afc7c59ba085bcf84cce75a02b6b1a6))
* Add support for none type ([#78](https://github.com/arxlang/irx/issues/78)) ([cf2978d](https://github.com/arxlang/irx/commit/cf2978d7c4d8b6b172b8cf53ee8f756fadb100ed))
* Add support for runtime type check ([#52](https://github.com/arxlang/irx/issues/52)) ([4c95199](https://github.com/arxlang/irx/commit/4c95199e681636d4e3bac3009b1b182f58be1835))
* Add support for system print function ([#57](https://github.com/arxlang/irx/issues/57)) ([88b9e10](https://github.com/arxlang/irx/commit/88b9e10b1a22207fc1206d686313808ecb3b1316))
* Add translation for `WhileStmt` ([#64](https://github.com/arxlang/irx/issues/64)) ([2933b51](https://github.com/arxlang/irx/commit/2933b519719ff286c6b5efd293dfd1402ee5c8d2))

## [1.2.1](https://github.com/arxlang/irx/compare/1.2.0...1.2.1) (2024-05-02)


### Bug Fixes

* Update ASTx to v0.12.2 ([#15](https://github.com/arxlang/irx/issues/15)) ([3ae5797](https://github.com/arxlang/irx/commit/3ae57979b1f9f7a7741a14001d153b0981d80d5a))

# [1.2.0](https://github.com/arxlang/irx/compare/1.1.0...1.2.0) (2024-04-21)


### Features

* Replace local customized AST implementation for Target and Module by astx classes ([#13](https://github.com/arxlang/irx/issues/13)) ([972eba6](https://github.com/arxlang/irx/commit/972eba6cbd0ba94f8425fa4d69788a56685aa1aa))

# [1.1.0](https://github.com/arxlang/irx/compare/1.0.2...1.1.0) (2024-03-24)


### Features

* Improve a several things: documentation, tests, CI, and the usage of ASTx ([#8](https://github.com/arxlang/irx/issues/8)) ([63393ff](https://github.com/arxlang/irx/commit/63393ff02fee059174474876a334623a4e23204f))

## [1.0.2](https://github.com/arxlang/irx/compare/1.0.1...1.0.2) (2024-01-19)


### Bug Fixes

* **docs:** fix installation step ([#11](https://github.com/arxlang/irx/issues/11)) ([9c07b1e](https://github.com/arxlang/irx/commit/9c07b1e98be3f3495dc442dbef1762244bfe523c))

## [1.0.1](https://github.com/arxlang/irx/compare/1.0.0...1.0.1) (2024-01-19)


### Bug Fixes

* Fix documentation publishing ([#10](https://github.com/arxlang/irx/issues/10)) ([3ce6c3d](https://github.com/arxlang/irx/commit/3ce6c3d2a188764a944cc1d8af955028ec3b61f4))

# 1.0.0 (2024-01-19)


### Bug Fixes

* Fix typing issues ([#6](https://github.com/arxlang/irx/issues/6)) ([0cd87c6](https://github.com/arxlang/irx/commit/0cd87c62c18966b584e966f523e7c9ddf34d0b7c))


### Features

* Add initial structure and initial translator for LLVM-IR ([#1](https://github.com/arxlang/irx/issues/1)) ([1c54eb4](https://github.com/arxlang/irx/commit/1c54eb474e9a53c97f7540435126972117e3d7a9))
* Rename project to IRx ([#9](https://github.com/arxlang/irx/issues/9)) ([3a85652](https://github.com/arxlang/irx/commit/3a85652650f723e787243be0e629cd7e21830a7f))
* Replace home-made llvm-ir compiler from llvmlite ([#7](https://github.com/arxlang/irx/issues/7)) ([daafff3](https://github.com/arxlang/irx/commit/daafff34201af60f3f08167a936f8edfa331d38a))

# Release Notes
---

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

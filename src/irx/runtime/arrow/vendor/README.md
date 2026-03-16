Pinned vendored dependencies for the IRx Arrow runtime feature.

- `nanoarrow` version: `apache-arrow-nanoarrow-0.6.0`
- Upstream: `https://github.com/apache/arrow-nanoarrow`
- Python package note: IRx also installs the Python `nanoarrow` package by
  default, but the wheel does not include the raw C headers/sources needed by
  the native Arrow runtime build
- Included files: C library headers and the minimal `array/schema/utils` sources
  needed by the IRx MVP runtime
- Build note: IRx compiles nanoarrow with `-DNANOARROW_NAMESPACE=IrxNanoarrow`
  so these helper symbols stay internal to the runtime feature

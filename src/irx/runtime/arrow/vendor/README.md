Pinned vendored dependencies for the IRx Arrow runtime feature.

- `nanoarrow` version: `apache-arrow-nanoarrow-0.6.0`
- Upstream: `https://github.com/apache/arrow-nanoarrow`
- Included files: C library headers and the minimal `array/schema/utils` sources
  needed by the IRx MVP runtime
- Build note: IRx compiles nanoarrow with `-DNANOARROW_NAMESPACE=IrxNanoarrow`
  so these helper symbols stay internal to the runtime feature

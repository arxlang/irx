// Copyright IRx contributors.

#ifndef IRX_ARROW_RUNTIME_H_INCLUDED
#define IRX_ARROW_RUNTIME_H_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "irx_arrow_c_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct irx_arrow_array_builder_handle irx_arrow_array_builder_handle;
typedef struct irx_arrow_array_handle irx_arrow_array_handle;

enum irx_arrow_type_id {
  IRX_ARROW_TYPE_UNKNOWN = 0,
  IRX_ARROW_TYPE_INT32 = 1,
};

int irx_arrow_array_builder_int32_new(
    irx_arrow_array_builder_handle** out_builder);
int irx_arrow_array_builder_append_int32(
    irx_arrow_array_builder_handle* builder, int32_t value);
int irx_arrow_array_builder_finish(
    irx_arrow_array_builder_handle* builder,
    irx_arrow_array_handle** out_array);
void irx_arrow_array_builder_release(irx_arrow_array_builder_handle* builder);

int64_t irx_arrow_array_length(const irx_arrow_array_handle* array);
int64_t irx_arrow_array_null_count(const irx_arrow_array_handle* array);
int32_t irx_arrow_array_type_id(const irx_arrow_array_handle* array);

int irx_arrow_array_export(
    const irx_arrow_array_handle* array,
    struct ArrowArray* out_array,
    struct ArrowSchema* out_schema);
int irx_arrow_array_import(
    const struct ArrowArray* array,
    const struct ArrowSchema* schema,
    irx_arrow_array_handle** out_array);

void irx_arrow_array_release(irx_arrow_array_handle* array);
const char* irx_arrow_last_error(void);

#ifdef __cplusplus
}
#endif

#endif

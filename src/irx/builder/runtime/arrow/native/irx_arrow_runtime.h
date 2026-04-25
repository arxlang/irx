// Copyright IRx contributors.

#ifndef IRX_ARROW_RUNTIME_H_INCLUDED
#define IRX_ARROW_RUNTIME_H_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "irx_arrow_c_abi.h"
#include "irx_buffer_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct irx_arrow_schema_handle irx_arrow_schema_handle;
typedef struct irx_arrow_array_builder_handle irx_arrow_array_builder_handle;
typedef struct irx_arrow_array_handle irx_arrow_array_handle;
typedef struct irx_arrow_tensor_builder_handle irx_arrow_tensor_builder_handle;
typedef struct irx_arrow_tensor_handle irx_arrow_tensor_handle;

enum irx_arrow_type_id {
  IRX_ARROW_TYPE_UNKNOWN = 0,
  IRX_ARROW_TYPE_INT32 = 1,
  IRX_ARROW_TYPE_INT8 = 2,
  IRX_ARROW_TYPE_INT16 = 3,
  IRX_ARROW_TYPE_INT64 = 4,
  IRX_ARROW_TYPE_UINT8 = 5,
  IRX_ARROW_TYPE_UINT16 = 6,
  IRX_ARROW_TYPE_UINT32 = 7,
  IRX_ARROW_TYPE_UINT64 = 8,
  IRX_ARROW_TYPE_FLOAT32 = 9,
  IRX_ARROW_TYPE_FLOAT64 = 10,
  IRX_ARROW_TYPE_BOOL = 11,
};

int irx_arrow_schema_import_copy(
    const struct ArrowSchema* schema,
    irx_arrow_schema_handle** out_schema);
int irx_arrow_schema_export(
    const irx_arrow_schema_handle* schema,
    struct ArrowSchema* out_schema);
int32_t irx_arrow_schema_type_id(const irx_arrow_schema_handle* schema);
int32_t irx_arrow_schema_is_nullable(const irx_arrow_schema_handle* schema);
int irx_arrow_schema_retain(irx_arrow_schema_handle* schema);
void irx_arrow_schema_release(irx_arrow_schema_handle* schema);

int irx_arrow_array_builder_new(
    int32_t type_id,
    irx_arrow_array_builder_handle** out_builder);
int irx_arrow_array_builder_append_null(
    irx_arrow_array_builder_handle* builder,
    int64_t count);
int irx_arrow_array_builder_append_int(
    irx_arrow_array_builder_handle* builder,
    int64_t value);
int irx_arrow_array_builder_append_uint(
    irx_arrow_array_builder_handle* builder,
    uint64_t value);
int irx_arrow_array_builder_append_double(
    irx_arrow_array_builder_handle* builder,
    double value);

int irx_arrow_array_builder_int32_new(
    irx_arrow_array_builder_handle** out_builder);
int irx_arrow_array_builder_append_int32(
    irx_arrow_array_builder_handle* builder, int32_t value);
int irx_arrow_array_builder_finish(
    irx_arrow_array_builder_handle* builder,
    irx_arrow_array_handle** out_array);
void irx_arrow_array_builder_release(irx_arrow_array_builder_handle* builder);

int64_t irx_arrow_array_length(const irx_arrow_array_handle* array);
int64_t irx_arrow_array_offset(const irx_arrow_array_handle* array);
int64_t irx_arrow_array_null_count(const irx_arrow_array_handle* array);
int32_t irx_arrow_array_type_id(const irx_arrow_array_handle* array);
int32_t irx_arrow_array_is_nullable(const irx_arrow_array_handle* array);
int32_t irx_arrow_array_has_validity_bitmap(
    const irx_arrow_array_handle* array);
int32_t irx_arrow_array_can_borrow_buffer_view(
    const irx_arrow_array_handle* array);

int irx_arrow_array_schema_copy(
    const irx_arrow_array_handle* array,
    irx_arrow_schema_handle** out_schema);

int irx_arrow_array_export(
    const irx_arrow_array_handle* array,
    struct ArrowArray* out_array,
    struct ArrowSchema* out_schema);
int irx_arrow_array_import(
    const struct ArrowArray* array,
    const struct ArrowSchema* schema,
    irx_arrow_array_handle** out_array);
int irx_arrow_array_import_copy(
    const struct ArrowArray* array,
    const struct ArrowSchema* schema,
    irx_arrow_array_handle** out_array);
int irx_arrow_array_import_move(
    struct ArrowArray* array,
    struct ArrowSchema* schema,
    irx_arrow_array_handle** out_array);

int irx_arrow_array_validity_bitmap(
    const irx_arrow_array_handle* array,
    const void** out_data,
    int64_t* out_offset_bits,
    int64_t* out_length_bits);
int irx_arrow_array_borrow_buffer_view(
    const irx_arrow_array_handle* array,
    irx_buffer_view* out_view);

int irx_arrow_array_retain(irx_arrow_array_handle* array);
void irx_arrow_array_release(irx_arrow_array_handle* array);

int irx_arrow_tensor_builder_new(
    int32_t type_id,
    int32_t ndim,
    const int64_t* shape,
    const int64_t* strides,
    irx_arrow_tensor_builder_handle** out_builder);
int irx_arrow_tensor_builder_append_int(
    irx_arrow_tensor_builder_handle* builder,
    int64_t value);
int irx_arrow_tensor_builder_append_uint(
    irx_arrow_tensor_builder_handle* builder,
    uint64_t value);
int irx_arrow_tensor_builder_append_double(
    irx_arrow_tensor_builder_handle* builder,
    double value);
int irx_arrow_tensor_builder_finish(
    irx_arrow_tensor_builder_handle* builder,
    irx_arrow_tensor_handle** out_tensor);
void irx_arrow_tensor_builder_release(
    irx_arrow_tensor_builder_handle* builder);

int32_t irx_arrow_tensor_type_id(const irx_arrow_tensor_handle* tensor);
int32_t irx_arrow_tensor_ndim(const irx_arrow_tensor_handle* tensor);
int64_t irx_arrow_tensor_size(const irx_arrow_tensor_handle* tensor);
const int64_t* irx_arrow_tensor_shape(const irx_arrow_tensor_handle* tensor);
const int64_t* irx_arrow_tensor_strides(const irx_arrow_tensor_handle* tensor);
int irx_arrow_tensor_borrow_buffer_view(
    const irx_arrow_tensor_handle* tensor,
    irx_buffer_view* out_view);
int irx_arrow_tensor_retain(irx_arrow_tensor_handle* tensor);
void irx_arrow_tensor_release(irx_arrow_tensor_handle* tensor);
const char* irx_arrow_last_error(void);

#ifdef __cplusplus
}
#endif

#endif

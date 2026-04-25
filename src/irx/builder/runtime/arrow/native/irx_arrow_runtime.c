// Copyright IRx contributors.

#include "irx_arrow_runtime.h"

#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"

typedef enum irx_arrow_append_kind {
  IRX_ARROW_APPEND_SIGNED = 1,
  IRX_ARROW_APPEND_UNSIGNED = 2,
  IRX_ARROW_APPEND_DOUBLE = 3,
} irx_arrow_append_kind;

typedef struct irx_arrow_type_spec {
  int32_t type_id;
  enum ArrowType storage_type;
  uintptr_t dtype_token;
  int64_t element_size_bytes;
  int32_t buffer_view_compatible;
  irx_arrow_append_kind append_kind;
  const char* name;
} irx_arrow_type_spec;

typedef struct irx_arrow_resolved_schema {
  const irx_arrow_type_spec* spec;
  int32_t nullable;
} irx_arrow_resolved_schema;

struct irx_arrow_schema_handle {
  int64_t refcount;
  struct ArrowSchema schema;
  int32_t type_id;
  int32_t nullable;
};

struct irx_arrow_array_builder_handle {
  struct ArrowSchema schema;
  struct ArrowArray array;
  int32_t type_id;
};

struct irx_arrow_array_handle {
  int64_t refcount;
  struct ArrowSchema schema;
  struct ArrowArray array;
  int32_t type_id;
  int32_t nullable;
  uintptr_t dtype_token;
  int64_t element_size_bytes;
  int32_t buffer_view_compatible;
  int64_t shape[1];
  int64_t strides[1];
};

struct irx_arrow_tensor_builder_handle {
  int32_t type_id;
  int32_t ndim;
  int64_t element_count;
  int64_t values_appended;
  int64_t data_nbytes;
  uintptr_t dtype_token;
  int64_t element_size_bytes;
  void* data;
  int64_t* shape;
  int64_t* strides;
};

struct irx_arrow_tensor_handle {
  int64_t refcount;
  int32_t type_id;
  int32_t ndim;
  int64_t element_count;
  int64_t data_nbytes;
  uintptr_t dtype_token;
  int64_t element_size_bytes;
  void* data;
  int64_t* shape;
  int64_t* strides;
};

static const irx_arrow_type_spec irx_arrow_type_specs[] = {
    {
        IRX_ARROW_TYPE_INT32,
        NANOARROW_TYPE_INT32,
        IRX_BUFFER_DTYPE_INT32,
        4,
        1,
        IRX_ARROW_APPEND_SIGNED,
        "int32",
    },
    {
        IRX_ARROW_TYPE_INT8,
        NANOARROW_TYPE_INT8,
        IRX_BUFFER_DTYPE_INT8,
        1,
        1,
        IRX_ARROW_APPEND_SIGNED,
        "int8",
    },
    {
        IRX_ARROW_TYPE_INT16,
        NANOARROW_TYPE_INT16,
        IRX_BUFFER_DTYPE_INT16,
        2,
        1,
        IRX_ARROW_APPEND_SIGNED,
        "int16",
    },
    {
        IRX_ARROW_TYPE_INT64,
        NANOARROW_TYPE_INT64,
        IRX_BUFFER_DTYPE_INT64,
        8,
        1,
        IRX_ARROW_APPEND_SIGNED,
        "int64",
    },
    {
        IRX_ARROW_TYPE_UINT8,
        NANOARROW_TYPE_UINT8,
        IRX_BUFFER_DTYPE_UINT8,
        1,
        1,
        IRX_ARROW_APPEND_UNSIGNED,
        "uint8",
    },
    {
        IRX_ARROW_TYPE_UINT16,
        NANOARROW_TYPE_UINT16,
        IRX_BUFFER_DTYPE_UINT16,
        2,
        1,
        IRX_ARROW_APPEND_UNSIGNED,
        "uint16",
    },
    {
        IRX_ARROW_TYPE_UINT32,
        NANOARROW_TYPE_UINT32,
        IRX_BUFFER_DTYPE_UINT32,
        4,
        1,
        IRX_ARROW_APPEND_UNSIGNED,
        "uint32",
    },
    {
        IRX_ARROW_TYPE_UINT64,
        NANOARROW_TYPE_UINT64,
        IRX_BUFFER_DTYPE_UINT64,
        8,
        1,
        IRX_ARROW_APPEND_UNSIGNED,
        "uint64",
    },
    {
        IRX_ARROW_TYPE_FLOAT32,
        NANOARROW_TYPE_FLOAT,
        IRX_BUFFER_DTYPE_FLOAT32,
        4,
        1,
        IRX_ARROW_APPEND_DOUBLE,
        "float32",
    },
    {
        IRX_ARROW_TYPE_FLOAT64,
        NANOARROW_TYPE_DOUBLE,
        IRX_BUFFER_DTYPE_FLOAT64,
        8,
        1,
        IRX_ARROW_APPEND_DOUBLE,
        "float64",
    },
    {
        IRX_ARROW_TYPE_BOOL,
        NANOARROW_TYPE_BOOL,
        IRX_BUFFER_DTYPE_BOOL,
        0,
        0,
        IRX_ARROW_APPEND_SIGNED,
        "bool",
    },
};

static char irx_arrow_last_error_buffer[512];

static void irx_arrow_clear_error(void) {
  irx_arrow_last_error_buffer[0] = '\0';
}

static int irx_arrow_set_error_code(int code, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(
      irx_arrow_last_error_buffer,
      sizeof(irx_arrow_last_error_buffer),
      fmt,
      args);
  va_end(args);
  return code;
}

static int irx_arrow_capture_nanoarrow_error(
    int code,
    const struct ArrowError* error,
    const char* context) {
  if (code == NANOARROW_OK) {
    return NANOARROW_OK;
  }

  if (error != NULL && error->message[0] != '\0') {
    return irx_arrow_set_error_code(code, "%s: %s", context, error->message);
  }

  return irx_arrow_set_error_code(
      code,
      "%s failed with error code %d",
      context,
      code);
}

static void irx_arrow_release_schema(struct ArrowSchema* schema) {
  if (schema->release != NULL) {
    ArrowSchemaRelease(schema);
  }
}

static void irx_arrow_release_array(struct ArrowArray* array) {
  if (array->release != NULL) {
    ArrowArrayRelease(array);
  }
}

static const irx_arrow_type_spec* irx_arrow_type_spec_from_type_id(
    int32_t type_id) {
  size_t i = 0;
  for (i = 0; i < sizeof(irx_arrow_type_specs) / sizeof(irx_arrow_type_specs[0]);
       ++i) {
    if (irx_arrow_type_specs[i].type_id == type_id) {
      return &irx_arrow_type_specs[i];
    }
  }

  return NULL;
}

static const irx_arrow_type_spec* irx_arrow_type_spec_from_storage_type(
    enum ArrowType storage_type) {
  size_t i = 0;
  for (i = 0; i < sizeof(irx_arrow_type_specs) / sizeof(irx_arrow_type_specs[0]);
       ++i) {
    if (irx_arrow_type_specs[i].storage_type == storage_type) {
      return &irx_arrow_type_specs[i];
    }
  }

  return NULL;
}

static int irx_arrow_init_schema_for_type(
    const irx_arrow_type_spec* spec,
    struct ArrowSchema* schema) {
  ArrowSchemaInit(schema);

  const int code = ArrowSchemaSetType(schema, spec->storage_type);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowSchemaSetType(%s) failed with error code %d",
        spec->name,
        code);
  }

  return NANOARROW_OK;
}

static int irx_arrow_init_array_from_schema(
    struct ArrowSchema* schema,
    struct ArrowArray* array) {
  struct ArrowError error;
  memset(&error, 0, sizeof(error));

  int code = ArrowArrayInitFromSchema(array, schema, &error);
  if (code != NANOARROW_OK) {
    return irx_arrow_capture_nanoarrow_error(
        code,
        &error,
        "ArrowArrayInitFromSchema");
  }

  code = ArrowArrayStartAppending(array);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowArrayStartAppending failed with error code %d",
        code);
  }

  return NANOARROW_OK;
}

static int irx_arrow_validate_supported_schema(
    const struct ArrowSchema* schema,
    irx_arrow_resolved_schema* out_resolved) {
  struct ArrowError error;
  struct ArrowSchemaView schema_view;
  memset(&error, 0, sizeof(error));
  memset(&schema_view, 0, sizeof(schema_view));

  int code = ArrowSchemaViewInit(&schema_view, schema, &error);
  if (code != NANOARROW_OK) {
    return irx_arrow_capture_nanoarrow_error(
        code,
        &error,
        "ArrowSchemaViewInit");
  }

  if (schema_view.type != schema_view.storage_type) {
    return irx_arrow_set_error_code(
        EINVAL,
        "Only plain primitive Arrow arrays are supported in this phase");
  }

  out_resolved->spec =
      irx_arrow_type_spec_from_storage_type(schema_view.storage_type);
  if (out_resolved->spec == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "Unsupported Arrow storage type; supported types are bool, "
        "int8, int16, int32, int64, uint8, uint16, uint32, uint64, "
        "float32, and float64");
  }

  out_resolved->nullable = (schema->flags & ARROW_FLAG_NULLABLE) != 0;
  return NANOARROW_OK;
}

static int irx_arrow_validate_supported_array(
    const struct ArrowArray* array,
    const struct ArrowSchema* schema,
    struct ArrowArrayView* out_view,
    irx_arrow_resolved_schema* out_resolved) {
  struct ArrowError error;
  memset(&error, 0, sizeof(error));

  int code = irx_arrow_validate_supported_schema(schema, out_resolved);
  if (code != NANOARROW_OK) {
    return code;
  }

  ArrowArrayViewInitFromType(out_view, NANOARROW_TYPE_UNINITIALIZED);

  code = ArrowArrayViewInitFromSchema(out_view, schema, &error);
  if (code != NANOARROW_OK) {
    ArrowArrayViewReset(out_view);
    return irx_arrow_capture_nanoarrow_error(
        code,
        &error,
        "ArrowArrayViewInitFromSchema");
  }

  if (out_view->storage_type != out_resolved->spec->storage_type) {
    ArrowArrayViewReset(out_view);
    return irx_arrow_set_error_code(
        EINVAL,
        "Arrow array storage type did not match the supported schema");
  }

  code = ArrowArrayViewSetArray(out_view, array, &error);
  if (code != NANOARROW_OK) {
    ArrowArrayViewReset(out_view);
    return irx_arrow_capture_nanoarrow_error(
        code,
        &error,
        "ArrowArrayViewSetArray");
  }

  code = ArrowArrayViewValidate(
      out_view,
      NANOARROW_VALIDATION_LEVEL_DEFAULT,
      &error);
  if (code != NANOARROW_OK) {
    ArrowArrayViewReset(out_view);
    return irx_arrow_capture_nanoarrow_error(
        code,
        &error,
        "ArrowArrayViewValidate");
  }

  return NANOARROW_OK;
}

static int irx_arrow_append_view_value(
    struct ArrowArray* out_array,
    const struct ArrowArrayView* view,
    const irx_arrow_resolved_schema* resolved,
    int64_t index) {
  switch (resolved->spec->append_kind) {
    case IRX_ARROW_APPEND_SIGNED:
      return ArrowArrayAppendInt(out_array, ArrowArrayViewGetIntUnsafe(view, index));
    case IRX_ARROW_APPEND_UNSIGNED:
      return ArrowArrayAppendUInt(
          out_array,
          ArrowArrayViewGetUIntUnsafe(view, index));
    case IRX_ARROW_APPEND_DOUBLE:
      return ArrowArrayAppendDouble(
          out_array,
          ArrowArrayViewGetDoubleUnsafe(view, index));
    default:
      return EINVAL;
  }
}

static int irx_arrow_copy_supported_view(
    const struct ArrowArrayView* view,
    const struct ArrowSchema* schema,
    const irx_arrow_resolved_schema* resolved,
    struct ArrowSchema* out_schema,
    struct ArrowArray* out_array) {
  struct ArrowError error;
  memset(&error, 0, sizeof(error));
  memset(out_schema, 0, sizeof(*out_schema));
  memset(out_array, 0, sizeof(*out_array));

  int code = ArrowSchemaDeepCopy(schema, out_schema);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowSchemaDeepCopy failed with error code %d",
        code);
  }

  code = irx_arrow_init_array_from_schema(out_schema, out_array);
  if (code != NANOARROW_OK) {
    irx_arrow_release_schema(out_schema);
    return code;
  }

  code = ArrowArrayReserve(out_array, view->length);
  if (code != NANOARROW_OK) {
    irx_arrow_release_array(out_array);
    irx_arrow_release_schema(out_schema);
    return irx_arrow_set_error_code(
        code,
        "ArrowArrayReserve failed with error code %d",
        code);
  }

  for (int64_t i = 0; i < view->length; ++i) {
    if (ArrowArrayViewIsNull(view, i)) {
      code = ArrowArrayAppendNull(out_array, 1);
    } else {
      code = irx_arrow_append_view_value(out_array, view, resolved, i);
    }

    if (code != NANOARROW_OK) {
      irx_arrow_release_array(out_array);
      irx_arrow_release_schema(out_schema);
      return irx_arrow_set_error_code(
          code,
          "Arrow array copy append failed with error code %d",
          code);
    }
  }

  code = ArrowArrayFinishBuildingDefault(out_array, &error);
  if (code != NANOARROW_OK) {
    irx_arrow_release_array(out_array);
    irx_arrow_release_schema(out_schema);
    return irx_arrow_capture_nanoarrow_error(
        code,
        &error,
        "ArrowArrayFinishBuildingDefault");
  }

  return NANOARROW_OK;
}

static void irx_arrow_populate_array_metadata(
    irx_arrow_array_handle* handle,
    const irx_arrow_resolved_schema* resolved) {
  handle->type_id = resolved->spec->type_id;
  handle->nullable = resolved->nullable;
  handle->dtype_token = resolved->spec->dtype_token;
  handle->element_size_bytes = resolved->spec->element_size_bytes;
  handle->buffer_view_compatible = resolved->spec->buffer_view_compatible;
  handle->shape[0] = handle->array.length;
  handle->strides[0] = resolved->spec->element_size_bytes;
}

static int irx_arrow_populate_schema_metadata(
    irx_arrow_schema_handle* handle) {
  irx_arrow_resolved_schema resolved;
  const int code =
      irx_arrow_validate_supported_schema(&handle->schema, &resolved);
  if (code != NANOARROW_OK) {
    return code;
  }

  handle->type_id = resolved.spec->type_id;
  handle->nullable = resolved.nullable;
  return NANOARROW_OK;
}

static int irx_arrow_array_has_validity_buffer(
    const irx_arrow_array_handle* array) {
  return array->array.n_buffers > 0 &&
         array->array.buffers != NULL &&
         array->array.buffers[0] != NULL;
}

static int irx_arrow_checked_offset_bytes(
    int64_t offset,
    int64_t element_size_bytes,
    int64_t* out_offset_bytes) {
  if (offset < 0 || element_size_bytes < 0) {
    return irx_arrow_set_error_code(
        EINVAL,
        "buffer view offset computation requires non-negative values");
  }

  if (offset > 0 && element_size_bytes > INT64_MAX / offset) {
    return irx_arrow_set_error_code(
        EOVERFLOW,
        "Arrow array offset overflowed buffer view byte offset");
  }

  *out_offset_bytes = offset * element_size_bytes;
  return NANOARROW_OK;
}

static void irx_arrow_tensor_builder_free(
    irx_arrow_tensor_builder_handle* builder) {
  if (builder == NULL) {
    return;
  }

  free(builder->data);
  free(builder->shape);
  free(builder->strides);
  free(builder);
}

static void irx_arrow_tensor_handle_free(irx_arrow_tensor_handle* tensor) {
  if (tensor == NULL) {
    return;
  }

  free(tensor->data);
  free(tensor->shape);
  free(tensor->strides);
  free(tensor);
}

static int irx_arrow_tensor_shape_extent(
    int32_t ndim,
    const int64_t* shape,
    int64_t* out_element_count) {
  int64_t element_count = 1;

  if (ndim < 0) {
    return irx_arrow_set_error_code(
        EINVAL,
        "tensor ndim must be non-negative");
  }
  if (ndim > 0 && shape == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "tensor shape must not be NULL when ndim is positive");
  }

  for (int32_t axis = 0; axis < ndim; ++axis) {
    const int64_t dim = shape[axis];
    if (dim < 0) {
      return irx_arrow_set_error_code(
          EINVAL,
          "tensor shape dimensions must be non-negative");
    }
    if (dim != 0 && element_count > INT64_MAX / dim) {
      return irx_arrow_set_error_code(
          EOVERFLOW,
          "tensor shape extent overflowed int64");
    }
    element_count *= dim;
  }

  *out_element_count = element_count;
  return NANOARROW_OK;
}

static int irx_arrow_tensor_copy_layout(
    int32_t ndim,
    const int64_t* shape,
    const int64_t* strides,
    int64_t element_size_bytes,
    int64_t** out_shape,
    int64_t** out_strides) {
  int64_t* copied_shape = NULL;
  int64_t* copied_strides = NULL;
  int64_t stride = element_size_bytes;

  *out_shape = NULL;
  *out_strides = NULL;

  if (ndim == 0) {
    return NANOARROW_OK;
  }

  copied_shape = (int64_t*)calloc((size_t)ndim, sizeof(*copied_shape));
  copied_strides = (int64_t*)calloc((size_t)ndim, sizeof(*copied_strides));
  if (copied_shape == NULL || copied_strides == NULL) {
    free(copied_shape);
    free(copied_strides);
    return irx_arrow_set_error_code(
        ENOMEM,
        "failed to allocate tensor layout metadata");
  }

  for (int32_t axis = 0; axis < ndim; ++axis) {
    copied_shape[axis] = shape[axis];
  }

  if (strides != NULL) {
    for (int32_t axis = 0; axis < ndim; ++axis) {
      if (strides[axis] < 0) {
        free(copied_shape);
        free(copied_strides);
        return irx_arrow_set_error_code(
            EINVAL,
            "tensor strides must be non-negative");
      }
      copied_strides[axis] = strides[axis];
    }
  } else {
    for (int32_t axis = ndim - 1; axis >= 0; --axis) {
      copied_strides[axis] = stride;
      const int64_t dim = copied_shape[axis] > 1 ? copied_shape[axis] : 1;
      if (stride > 0 && dim > INT64_MAX / stride) {
        free(copied_shape);
        free(copied_strides);
        return irx_arrow_set_error_code(
            EOVERFLOW,
            "tensor default stride computation overflowed int64");
      }
      stride *= dim;
    }
  }

  *out_shape = copied_shape;
  *out_strides = copied_strides;
  return NANOARROW_OK;
}

static int irx_arrow_tensor_data_nbytes(
    int64_t element_count,
    int64_t element_size_bytes,
    int64_t* out_data_nbytes) {
  if (element_size_bytes <= 0) {
    return irx_arrow_set_error_code(
        EINVAL,
        "tensor element type must have a positive byte width");
  }
  if (element_count > 0 && element_count > INT64_MAX / element_size_bytes) {
    return irx_arrow_set_error_code(
        EOVERFLOW,
        "tensor data size overflowed int64");
  }

  *out_data_nbytes = element_count * element_size_bytes;
  return NANOARROW_OK;
}

static int irx_arrow_tensor_builder_require_slot(
    irx_arrow_tensor_builder_handle* builder,
    void** out_slot) {
  if (builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "tensor builder must not be NULL");
  }
  if (builder->values_appended >= builder->element_count) {
    return irx_arrow_set_error_code(
        EINVAL,
        "too many values appended to tensor builder");
  }

  *out_slot = (void*)((uint8_t*)builder->data +
                      builder->values_appended * builder->element_size_bytes);
  builder->values_appended += 1;
  return NANOARROW_OK;
}

static int irx_arrow_tensor_is_c_contiguous(
    const irx_arrow_tensor_handle* tensor) {
  int64_t stride = tensor->element_size_bytes;

  for (int32_t axis = tensor->ndim - 1; axis >= 0; --axis) {
    if (tensor->strides[axis] != stride) {
      return 0;
    }
    const int64_t dim = tensor->shape[axis] > 1 ? tensor->shape[axis] : 1;
    if (stride > 0 && dim > INT64_MAX / stride) {
      return 0;
    }
    stride *= dim;
  }

  return 1;
}

static int irx_arrow_tensor_is_f_contiguous(
    const irx_arrow_tensor_handle* tensor) {
  int64_t stride = tensor->element_size_bytes;

  for (int32_t axis = 0; axis < tensor->ndim; ++axis) {
    if (tensor->strides[axis] != stride) {
      return 0;
    }
    const int64_t dim = tensor->shape[axis] > 1 ? tensor->shape[axis] : 1;
    if (stride > 0 && dim > INT64_MAX / stride) {
      return 0;
    }
    stride *= dim;
  }

  return 1;
}

int irx_arrow_schema_import_copy(
    const struct ArrowSchema* schema,
    irx_arrow_schema_handle** out_schema) {
  irx_arrow_clear_error();
  if (schema == NULL) {
    return irx_arrow_set_error_code(EINVAL, "schema must not be NULL");
  }
  if (out_schema == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_schema must not be NULL");
  }

  *out_schema = NULL;

  irx_arrow_schema_handle* handle =
      (irx_arrow_schema_handle*)malloc(sizeof(*handle));
  if (handle == NULL) {
    return irx_arrow_set_error_code(ENOMEM, "failed to allocate Arrow schema");
  }

  memset(handle, 0, sizeof(*handle));
  handle->refcount = 1;

  int code = ArrowSchemaDeepCopy(schema, &handle->schema);
  if (code != NANOARROW_OK) {
    free(handle);
    return irx_arrow_set_error_code(
        code,
        "ArrowSchemaDeepCopy failed with error code %d",
        code);
  }

  code = irx_arrow_populate_schema_metadata(handle);
  if (code != NANOARROW_OK) {
    irx_arrow_release_schema(&handle->schema);
    free(handle);
    return code;
  }

  *out_schema = handle;
  return NANOARROW_OK;
}

int irx_arrow_schema_export(
    const irx_arrow_schema_handle* schema,
    struct ArrowSchema* out_schema) {
  irx_arrow_clear_error();
  if (schema == NULL) {
    return irx_arrow_set_error_code(EINVAL, "schema must not be NULL");
  }
  if (out_schema == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_schema must not be NULL");
  }

  memset(out_schema, 0, sizeof(*out_schema));
  const int code = ArrowSchemaDeepCopy(&schema->schema, out_schema);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowSchemaDeepCopy failed with error code %d",
        code);
  }

  return NANOARROW_OK;
}

int32_t irx_arrow_schema_type_id(const irx_arrow_schema_handle* schema) {
  irx_arrow_clear_error();
  if (schema == NULL) {
    irx_arrow_set_error_code(EINVAL, "schema must not be NULL");
    return IRX_ARROW_TYPE_UNKNOWN;
  }

  return schema->type_id;
}

int32_t irx_arrow_schema_is_nullable(const irx_arrow_schema_handle* schema) {
  irx_arrow_clear_error();
  if (schema == NULL) {
    irx_arrow_set_error_code(EINVAL, "schema must not be NULL");
    return 0;
  }

  return schema->nullable;
}

int irx_arrow_schema_retain(irx_arrow_schema_handle* schema) {
  irx_arrow_clear_error();
  if (schema == NULL) {
    return NANOARROW_OK;
  }
  if (schema->refcount <= 0) {
    return irx_arrow_set_error_code(EINVAL, "schema handle is released");
  }

  schema->refcount += 1;
  return NANOARROW_OK;
}

void irx_arrow_schema_release(irx_arrow_schema_handle* schema) {
  if (schema == NULL) {
    return;
  }
  if (schema->refcount <= 0) {
    return;
  }

  schema->refcount -= 1;
  if (schema->refcount == 0) {
    irx_arrow_release_schema(&schema->schema);
    free(schema);
  }
}

int irx_arrow_array_builder_new(
    int32_t type_id,
    irx_arrow_array_builder_handle** out_builder) {
  irx_arrow_clear_error();
  if (out_builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_builder must not be NULL");
  }

  const irx_arrow_type_spec* spec = irx_arrow_type_spec_from_type_id(type_id);
  if (spec == NULL) {
    return irx_arrow_set_error_code(EINVAL, "unsupported Arrow type id %d", type_id);
  }

  irx_arrow_array_builder_handle* builder =
      (irx_arrow_array_builder_handle*)malloc(sizeof(*builder));
  if (builder == NULL) {
    return irx_arrow_set_error_code(ENOMEM, "failed to allocate Arrow builder");
  }

  memset(builder, 0, sizeof(*builder));
  builder->type_id = type_id;

  int code = irx_arrow_init_schema_for_type(spec, &builder->schema);
  if (code != NANOARROW_OK) {
    free(builder);
    return code;
  }

  code = irx_arrow_init_array_from_schema(&builder->schema, &builder->array);
  if (code != NANOARROW_OK) {
    irx_arrow_release_schema(&builder->schema);
    free(builder);
    return code;
  }

  *out_builder = builder;
  return NANOARROW_OK;
}

int irx_arrow_array_builder_append_null(
    irx_arrow_array_builder_handle* builder,
    int64_t count) {
  irx_arrow_clear_error();
  if (builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "builder must not be NULL");
  }
  if (count < 0) {
    return irx_arrow_set_error_code(EINVAL, "null append count must be non-negative");
  }

  const int code = ArrowArrayAppendNull(&builder->array, count);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowArrayAppendNull failed with error code %d",
        code);
  }

  return NANOARROW_OK;
}

int irx_arrow_array_builder_append_int(
    irx_arrow_array_builder_handle* builder,
    int64_t value) {
  irx_arrow_clear_error();
  if (builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "builder must not be NULL");
  }

  const int code = ArrowArrayAppendInt(&builder->array, value);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowArrayAppendInt failed with error code %d",
        code);
  }

  return NANOARROW_OK;
}

int irx_arrow_array_builder_append_uint(
    irx_arrow_array_builder_handle* builder,
    uint64_t value) {
  irx_arrow_clear_error();
  if (builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "builder must not be NULL");
  }

  const int code = ArrowArrayAppendUInt(&builder->array, value);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowArrayAppendUInt failed with error code %d",
        code);
  }

  return NANOARROW_OK;
}

int irx_arrow_array_builder_append_double(
    irx_arrow_array_builder_handle* builder,
    double value) {
  irx_arrow_clear_error();
  if (builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "builder must not be NULL");
  }

  const int code = ArrowArrayAppendDouble(&builder->array, value);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowArrayAppendDouble failed with error code %d",
        code);
  }

  return NANOARROW_OK;
}

int irx_arrow_array_builder_int32_new(
    irx_arrow_array_builder_handle** out_builder) {
  return irx_arrow_array_builder_new(IRX_ARROW_TYPE_INT32, out_builder);
}

int irx_arrow_array_builder_append_int32(
    irx_arrow_array_builder_handle* builder,
    int32_t value) {
  return irx_arrow_array_builder_append_int(builder, value);
}

int irx_arrow_array_builder_finish(
    irx_arrow_array_builder_handle* builder,
    irx_arrow_array_handle** out_array) {
  struct ArrowError error;
  const irx_arrow_type_spec* spec;
  irx_arrow_resolved_schema resolved;
  memset(&error, 0, sizeof(error));
  irx_arrow_clear_error();

  if (builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "builder must not be NULL");
  }
  if (out_array == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_array must not be NULL");
  }

  *out_array = NULL;

  spec = irx_arrow_type_spec_from_type_id(builder->type_id);
  if (spec == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "builder used unsupported Arrow type id %d",
        builder->type_id);
  }

  int code = ArrowArrayFinishBuildingDefault(&builder->array, &error);
  if (code != NANOARROW_OK) {
    return irx_arrow_capture_nanoarrow_error(
        code,
        &error,
        "ArrowArrayFinishBuildingDefault");
  }

  irx_arrow_array_handle* array =
      (irx_arrow_array_handle*)malloc(sizeof(*array));
  if (array == NULL) {
    return irx_arrow_set_error_code(ENOMEM, "failed to allocate Arrow array");
  }

  memset(array, 0, sizeof(*array));
  array->refcount = 1;
  ArrowSchemaMove(&builder->schema, &array->schema);
  ArrowArrayMove(&builder->array, &array->array);

  resolved.spec = spec;
  resolved.nullable = (array->schema.flags & ARROW_FLAG_NULLABLE) != 0;
  irx_arrow_populate_array_metadata(array, &resolved);

  free(builder);
  *out_array = array;
  return NANOARROW_OK;
}

void irx_arrow_array_builder_release(irx_arrow_array_builder_handle* builder) {
  if (builder == NULL) {
    return;
  }

  irx_arrow_release_array(&builder->array);
  irx_arrow_release_schema(&builder->schema);
  free(builder);
}

int64_t irx_arrow_array_length(const irx_arrow_array_handle* array) {
  irx_arrow_clear_error();
  if (array == NULL) {
    irx_arrow_set_error_code(EINVAL, "array must not be NULL");
    return -1;
  }

  return array->array.length;
}

int64_t irx_arrow_array_offset(const irx_arrow_array_handle* array) {
  irx_arrow_clear_error();
  if (array == NULL) {
    irx_arrow_set_error_code(EINVAL, "array must not be NULL");
    return -1;
  }

  return array->array.offset;
}

int64_t irx_arrow_array_null_count(const irx_arrow_array_handle* array) {
  irx_arrow_clear_error();
  if (array == NULL) {
    irx_arrow_set_error_code(EINVAL, "array must not be NULL");
    return -1;
  }

  return array->array.null_count;
}

int32_t irx_arrow_array_type_id(const irx_arrow_array_handle* array) {
  irx_arrow_clear_error();
  if (array == NULL) {
    irx_arrow_set_error_code(EINVAL, "array must not be NULL");
    return IRX_ARROW_TYPE_UNKNOWN;
  }

  return array->type_id;
}

int32_t irx_arrow_array_is_nullable(const irx_arrow_array_handle* array) {
  irx_arrow_clear_error();
  if (array == NULL) {
    irx_arrow_set_error_code(EINVAL, "array must not be NULL");
    return 0;
  }

  return array->nullable;
}

int32_t irx_arrow_array_has_validity_bitmap(
    const irx_arrow_array_handle* array) {
  irx_arrow_clear_error();
  if (array == NULL) {
    irx_arrow_set_error_code(EINVAL, "array must not be NULL");
    return 0;
  }

  return irx_arrow_array_has_validity_buffer(array);
}

int32_t irx_arrow_array_can_borrow_buffer_view(
    const irx_arrow_array_handle* array) {
  irx_arrow_clear_error();
  if (array == NULL) {
    irx_arrow_set_error_code(EINVAL, "array must not be NULL");
    return 0;
  }

  return array->buffer_view_compatible;
}

int irx_arrow_array_schema_copy(
    const irx_arrow_array_handle* array,
    irx_arrow_schema_handle** out_schema) {
  int code = NANOARROW_OK;
  irx_arrow_clear_error();
  if (array == NULL) {
    return irx_arrow_set_error_code(EINVAL, "array must not be NULL");
  }
  if (out_schema == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_schema must not be NULL");
  }

  *out_schema = NULL;

  irx_arrow_schema_handle* handle =
      (irx_arrow_schema_handle*)malloc(sizeof(*handle));
  if (handle == NULL) {
    return irx_arrow_set_error_code(ENOMEM, "failed to allocate Arrow schema");
  }

  memset(handle, 0, sizeof(*handle));
  handle->refcount = 1;

  code = ArrowSchemaDeepCopy(&array->schema, &handle->schema);
  if (code != NANOARROW_OK) {
    free(handle);
    return irx_arrow_set_error_code(
        code,
        "ArrowSchemaDeepCopy failed with error code %d",
        code);
  }

  code = irx_arrow_populate_schema_metadata(handle);
  if (code != NANOARROW_OK) {
    irx_arrow_release_schema(&handle->schema);
    free(handle);
    return code;
  }

  *out_schema = handle;
  return NANOARROW_OK;
}

int irx_arrow_array_export(
    const irx_arrow_array_handle* array,
    struct ArrowArray* out_array,
    struct ArrowSchema* out_schema) {
  struct ArrowArrayView view;
  irx_arrow_resolved_schema resolved;
  irx_arrow_clear_error();

  if (array == NULL) {
    return irx_arrow_set_error_code(EINVAL, "array must not be NULL");
  }
  if (out_array == NULL || out_schema == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "out_array and out_schema must not be NULL");
  }

  const int code = irx_arrow_validate_supported_array(
      &array->array,
      &array->schema,
      &view,
      &resolved);
  if (code != NANOARROW_OK) {
    return code;
  }

  const int copy_code = irx_arrow_copy_supported_view(
      &view,
      &array->schema,
      &resolved,
      out_schema,
      out_array);
  ArrowArrayViewReset(&view);
  return copy_code;
}

int irx_arrow_array_import(
    const struct ArrowArray* array,
    const struct ArrowSchema* schema,
    irx_arrow_array_handle** out_array) {
  return irx_arrow_array_import_copy(array, schema, out_array);
}

int irx_arrow_array_import_copy(
    const struct ArrowArray* array,
    const struct ArrowSchema* schema,
    irx_arrow_array_handle** out_array) {
  struct ArrowArrayView view;
  irx_arrow_resolved_schema resolved;
  irx_arrow_clear_error();

  if (array == NULL || schema == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "array and schema must not be NULL");
  }
  if (out_array == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_array must not be NULL");
  }

  *out_array = NULL;

  int code = irx_arrow_validate_supported_array(array, schema, &view, &resolved);
  if (code != NANOARROW_OK) {
    return code;
  }

  irx_arrow_array_handle* handle =
      (irx_arrow_array_handle*)malloc(sizeof(*handle));
  if (handle == NULL) {
    ArrowArrayViewReset(&view);
    return irx_arrow_set_error_code(ENOMEM, "failed to allocate Arrow array");
  }

  memset(handle, 0, sizeof(*handle));
  handle->refcount = 1;

  code = irx_arrow_copy_supported_view(
      &view,
      schema,
      &resolved,
      &handle->schema,
      &handle->array);
  ArrowArrayViewReset(&view);
  if (code != NANOARROW_OK) {
    free(handle);
    return code;
  }

  irx_arrow_populate_array_metadata(handle, &resolved);
  *out_array = handle;
  return NANOARROW_OK;
}

int irx_arrow_array_import_move(
    struct ArrowArray* array,
    struct ArrowSchema* schema,
    irx_arrow_array_handle** out_array) {
  struct ArrowArrayView view;
  irx_arrow_resolved_schema resolved;
  irx_arrow_clear_error();

  if (array == NULL || schema == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "array and schema must not be NULL");
  }
  if (out_array == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_array must not be NULL");
  }

  *out_array = NULL;

  int code = irx_arrow_validate_supported_array(array, schema, &view, &resolved);
  if (code != NANOARROW_OK) {
    return code;
  }

  if (array->null_count < 0) {
    array->null_count = ArrowArrayViewComputeNullCount(&view);
  }
  ArrowArrayViewReset(&view);

  irx_arrow_array_handle* handle =
      (irx_arrow_array_handle*)malloc(sizeof(*handle));
  if (handle == NULL) {
    return irx_arrow_set_error_code(ENOMEM, "failed to allocate Arrow array");
  }

  memset(handle, 0, sizeof(*handle));
  handle->refcount = 1;
  ArrowSchemaMove(schema, &handle->schema);
  ArrowArrayMove(array, &handle->array);
  irx_arrow_populate_array_metadata(handle, &resolved);

  *out_array = handle;
  return NANOARROW_OK;
}

int irx_arrow_array_validity_bitmap(
    const irx_arrow_array_handle* array,
    const void** out_data,
    int64_t* out_offset_bits,
    int64_t* out_length_bits) {
  irx_arrow_clear_error();
  if (array == NULL) {
    return irx_arrow_set_error_code(EINVAL, "array must not be NULL");
  }
  if (out_data == NULL || out_offset_bits == NULL || out_length_bits == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "out_data, out_offset_bits, and out_length_bits must not be NULL");
  }

  *out_data = NULL;
  *out_offset_bits = 0;
  *out_length_bits = array->array.length;

  if (irx_arrow_array_has_validity_buffer(array)) {
    *out_data = array->array.buffers[0];
    *out_offset_bits = array->array.offset;
  }

  return NANOARROW_OK;
}

int irx_arrow_array_borrow_buffer_view(
    const irx_arrow_array_handle* array,
    irx_buffer_view* out_view) {
  int64_t offset_bytes = 0;
  irx_arrow_clear_error();

  if (array == NULL) {
    return irx_arrow_set_error_code(EINVAL, "array must not be NULL");
  }
  if (out_view == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_view must not be NULL");
  }
  if (!array->buffer_view_compatible) {
    return irx_arrow_set_error_code(
        EINVAL,
        "Arrow bool arrays use bit-packed values and cannot be exposed "
        "as plain buffer views");
  }

  const int code = irx_arrow_checked_offset_bytes(
      array->array.offset,
      array->element_size_bytes,
      &offset_bytes);
  if (code != NANOARROW_OK) {
    return code;
  }

  memset(out_view, 0, sizeof(*out_view));
  out_view->data =
      (void*)((array->array.n_buffers > 1 && array->array.buffers != NULL)
                  ? array->array.buffers[1]
                  : NULL);
  out_view->owner = NULL;
  out_view->dtype = (void*)array->dtype_token;
  out_view->ndim = 1;
  out_view->shape = (int64_t*)array->shape;
  out_view->strides = (int64_t*)array->strides;
  out_view->offset_bytes = offset_bytes;
  out_view->flags = IRX_BUFFER_FLAG_BORROWED |
                    IRX_BUFFER_FLAG_READONLY |
                    IRX_BUFFER_FLAG_C_CONTIGUOUS;

  if (irx_arrow_array_has_validity_buffer(array)) {
    out_view->flags |= IRX_BUFFER_FLAG_VALIDITY_BITMAP;
  }

  return NANOARROW_OK;
}

int irx_arrow_array_retain(irx_arrow_array_handle* array) {
  irx_arrow_clear_error();
  if (array == NULL) {
    return NANOARROW_OK;
  }
  if (array->refcount <= 0) {
    return irx_arrow_set_error_code(EINVAL, "array handle is released");
  }

  array->refcount += 1;
  return NANOARROW_OK;
}

void irx_arrow_array_release(irx_arrow_array_handle* array) {
  if (array == NULL) {
    return;
  }
  if (array->refcount <= 0) {
    return;
  }

  array->refcount -= 1;
  if (array->refcount == 0) {
    irx_arrow_release_array(&array->array);
    irx_arrow_release_schema(&array->schema);
    free(array);
  }
}

int irx_arrow_tensor_builder_new(
    int32_t type_id,
    int32_t ndim,
    const int64_t* shape,
    const int64_t* strides,
    irx_arrow_tensor_builder_handle** out_builder) {
  int64_t element_count = 0;
  int64_t data_nbytes = 0;
  int64_t* copied_shape = NULL;
  int64_t* copied_strides = NULL;
  void* data = NULL;

  irx_arrow_clear_error();
  if (out_builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_builder must not be NULL");
  }
  *out_builder = NULL;

  const irx_arrow_type_spec* spec = irx_arrow_type_spec_from_type_id(type_id);
  if (spec == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "unsupported Arrow tensor type id %d",
        type_id);
  }
  if (!spec->buffer_view_compatible || spec->element_size_bytes <= 0) {
    return irx_arrow_set_error_code(
        EINVAL,
        "Arrow tensor builder requires a fixed-width primitive value type");
  }

  int code = irx_arrow_tensor_shape_extent(ndim, shape, &element_count);
  if (code != NANOARROW_OK) {
    return code;
  }
  code = irx_arrow_tensor_data_nbytes(
      element_count,
      spec->element_size_bytes,
      &data_nbytes);
  if (code != NANOARROW_OK) {
    return code;
  }
  code = irx_arrow_tensor_copy_layout(
      ndim,
      shape,
      strides,
      spec->element_size_bytes,
      &copied_shape,
      &copied_strides);
  if (code != NANOARROW_OK) {
    return code;
  }

  if (data_nbytes > 0) {
    data = calloc(1, (size_t)data_nbytes);
    if (data == NULL) {
      free(copied_shape);
      free(copied_strides);
      return irx_arrow_set_error_code(
          ENOMEM,
          "failed to allocate Arrow tensor data buffer");
    }
  }

  irx_arrow_tensor_builder_handle* builder =
      (irx_arrow_tensor_builder_handle*)calloc(1, sizeof(*builder));
  if (builder == NULL) {
    free(data);
    free(copied_shape);
    free(copied_strides);
    return irx_arrow_set_error_code(
        ENOMEM,
        "failed to allocate Arrow tensor builder");
  }

  builder->type_id = type_id;
  builder->ndim = ndim;
  builder->element_count = element_count;
  builder->data_nbytes = data_nbytes;
  builder->dtype_token = spec->dtype_token;
  builder->element_size_bytes = spec->element_size_bytes;
  builder->data = data;
  builder->shape = copied_shape;
  builder->strides = copied_strides;

  *out_builder = builder;
  return NANOARROW_OK;
}

int irx_arrow_tensor_builder_append_int(
    irx_arrow_tensor_builder_handle* builder,
    int64_t value) {
  void* slot = NULL;
  irx_arrow_clear_error();

  const int code = irx_arrow_tensor_builder_require_slot(builder, &slot);
  if (code != NANOARROW_OK) {
    return code;
  }

  switch (builder->type_id) {
    case IRX_ARROW_TYPE_INT8: {
      const int8_t typed_value = (int8_t)value;
      memcpy(slot, &typed_value, sizeof(typed_value));
      return NANOARROW_OK;
    }
    case IRX_ARROW_TYPE_INT16: {
      const int16_t typed_value = (int16_t)value;
      memcpy(slot, &typed_value, sizeof(typed_value));
      return NANOARROW_OK;
    }
    case IRX_ARROW_TYPE_INT32: {
      const int32_t typed_value = (int32_t)value;
      memcpy(slot, &typed_value, sizeof(typed_value));
      return NANOARROW_OK;
    }
    case IRX_ARROW_TYPE_INT64: {
      const int64_t typed_value = value;
      memcpy(slot, &typed_value, sizeof(typed_value));
      return NANOARROW_OK;
    }
    default:
      builder->values_appended -= 1;
      return irx_arrow_set_error_code(
          EINVAL,
          "tensor builder expected a signed integer element type");
  }
}

int irx_arrow_tensor_builder_append_uint(
    irx_arrow_tensor_builder_handle* builder,
    uint64_t value) {
  void* slot = NULL;
  irx_arrow_clear_error();

  const int code = irx_arrow_tensor_builder_require_slot(builder, &slot);
  if (code != NANOARROW_OK) {
    return code;
  }

  switch (builder->type_id) {
    case IRX_ARROW_TYPE_UINT8: {
      const uint8_t typed_value = (uint8_t)value;
      memcpy(slot, &typed_value, sizeof(typed_value));
      return NANOARROW_OK;
    }
    case IRX_ARROW_TYPE_UINT16: {
      const uint16_t typed_value = (uint16_t)value;
      memcpy(slot, &typed_value, sizeof(typed_value));
      return NANOARROW_OK;
    }
    case IRX_ARROW_TYPE_UINT32: {
      const uint32_t typed_value = (uint32_t)value;
      memcpy(slot, &typed_value, sizeof(typed_value));
      return NANOARROW_OK;
    }
    case IRX_ARROW_TYPE_UINT64: {
      const uint64_t typed_value = value;
      memcpy(slot, &typed_value, sizeof(typed_value));
      return NANOARROW_OK;
    }
    default:
      builder->values_appended -= 1;
      return irx_arrow_set_error_code(
          EINVAL,
          "tensor builder expected an unsigned integer element type");
  }
}

int irx_arrow_tensor_builder_append_double(
    irx_arrow_tensor_builder_handle* builder,
    double value) {
  void* slot = NULL;
  irx_arrow_clear_error();

  const int code = irx_arrow_tensor_builder_require_slot(builder, &slot);
  if (code != NANOARROW_OK) {
    return code;
  }

  switch (builder->type_id) {
    case IRX_ARROW_TYPE_FLOAT32: {
      const float typed_value = (float)value;
      memcpy(slot, &typed_value, sizeof(typed_value));
      return NANOARROW_OK;
    }
    case IRX_ARROW_TYPE_FLOAT64: {
      const double typed_value = value;
      memcpy(slot, &typed_value, sizeof(typed_value));
      return NANOARROW_OK;
    }
    default:
      builder->values_appended -= 1;
      return irx_arrow_set_error_code(
          EINVAL,
          "tensor builder expected a floating element type");
  }
}

int irx_arrow_tensor_builder_finish(
    irx_arrow_tensor_builder_handle* builder,
    irx_arrow_tensor_handle** out_tensor) {
  irx_arrow_clear_error();
  if (builder == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "tensor builder must not be NULL");
  }
  if (out_tensor == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_tensor must not be NULL");
  }
  *out_tensor = NULL;

  if (builder->values_appended != builder->element_count) {
    return irx_arrow_set_error_code(
        EINVAL,
        "tensor builder value count does not match tensor shape extent");
  }

  irx_arrow_tensor_handle* tensor =
      (irx_arrow_tensor_handle*)calloc(1, sizeof(*tensor));
  if (tensor == NULL) {
    return irx_arrow_set_error_code(
        ENOMEM,
        "failed to allocate Arrow tensor handle");
  }

  tensor->refcount = 1;
  tensor->type_id = builder->type_id;
  tensor->ndim = builder->ndim;
  tensor->element_count = builder->element_count;
  tensor->data_nbytes = builder->data_nbytes;
  tensor->dtype_token = builder->dtype_token;
  tensor->element_size_bytes = builder->element_size_bytes;
  tensor->data = builder->data;
  tensor->shape = builder->shape;
  tensor->strides = builder->strides;

  builder->data = NULL;
  builder->shape = NULL;
  builder->strides = NULL;
  irx_arrow_tensor_builder_free(builder);

  *out_tensor = tensor;
  return NANOARROW_OK;
}

void irx_arrow_tensor_builder_release(
    irx_arrow_tensor_builder_handle* builder) {
  irx_arrow_tensor_builder_free(builder);
}

int32_t irx_arrow_tensor_type_id(const irx_arrow_tensor_handle* tensor) {
  irx_arrow_clear_error();
  if (tensor == NULL) {
    irx_arrow_set_error_code(EINVAL, "tensor must not be NULL");
    return IRX_ARROW_TYPE_UNKNOWN;
  }

  return tensor->type_id;
}

int32_t irx_arrow_tensor_ndim(const irx_arrow_tensor_handle* tensor) {
  irx_arrow_clear_error();
  if (tensor == NULL) {
    irx_arrow_set_error_code(EINVAL, "tensor must not be NULL");
    return -1;
  }

  return tensor->ndim;
}

int64_t irx_arrow_tensor_size(const irx_arrow_tensor_handle* tensor) {
  irx_arrow_clear_error();
  if (tensor == NULL) {
    irx_arrow_set_error_code(EINVAL, "tensor must not be NULL");
    return -1;
  }

  return tensor->element_count;
}

const int64_t* irx_arrow_tensor_shape(const irx_arrow_tensor_handle* tensor) {
  irx_arrow_clear_error();
  if (tensor == NULL) {
    irx_arrow_set_error_code(EINVAL, "tensor must not be NULL");
    return NULL;
  }

  return tensor->shape;
}

const int64_t* irx_arrow_tensor_strides(const irx_arrow_tensor_handle* tensor) {
  irx_arrow_clear_error();
  if (tensor == NULL) {
    irx_arrow_set_error_code(EINVAL, "tensor must not be NULL");
    return NULL;
  }

  return tensor->strides;
}

int irx_arrow_tensor_borrow_buffer_view(
    const irx_arrow_tensor_handle* tensor,
    irx_buffer_view* out_view) {
  irx_arrow_clear_error();
  if (tensor == NULL) {
    return irx_arrow_set_error_code(EINVAL, "tensor must not be NULL");
  }
  if (out_view == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_view must not be NULL");
  }

  memset(out_view, 0, sizeof(*out_view));
  out_view->data = tensor->data;
  out_view->owner = NULL;
  out_view->dtype = (void*)tensor->dtype_token;
  out_view->ndim = tensor->ndim;
  out_view->shape = tensor->shape;
  out_view->strides = tensor->strides;
  out_view->offset_bytes = 0;
  out_view->flags = IRX_BUFFER_FLAG_BORROWED | IRX_BUFFER_FLAG_READONLY;

  if (irx_arrow_tensor_is_c_contiguous(tensor)) {
    out_view->flags |= IRX_BUFFER_FLAG_C_CONTIGUOUS;
  }
  if (irx_arrow_tensor_is_f_contiguous(tensor)) {
    out_view->flags |= IRX_BUFFER_FLAG_F_CONTIGUOUS;
  }

  return NANOARROW_OK;
}

int irx_arrow_tensor_retain(irx_arrow_tensor_handle* tensor) {
  irx_arrow_clear_error();
  if (tensor == NULL) {
    return NANOARROW_OK;
  }
  if (tensor->refcount <= 0) {
    return irx_arrow_set_error_code(EINVAL, "tensor handle is released");
  }

  tensor->refcount += 1;
  return NANOARROW_OK;
}

void irx_arrow_tensor_release(irx_arrow_tensor_handle* tensor) {
  if (tensor == NULL) {
    return;
  }
  if (tensor->refcount <= 0) {
    return;
  }

  tensor->refcount -= 1;
  if (tensor->refcount == 0) {
    irx_arrow_tensor_handle_free(tensor);
  }
}

const char* irx_arrow_last_error(void) {
  return irx_arrow_last_error_buffer;
}

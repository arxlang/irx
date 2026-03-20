// Copyright IRx contributors.

#include "irx_arrow_runtime.h"

#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"

struct irx_arrow_array_builder_handle {
  struct ArrowSchema schema;
  struct ArrowArray array;
  int32_t type_id;
};

struct irx_arrow_array_handle {
  struct ArrowSchema schema;
  struct ArrowArray array;
  int32_t type_id;
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

static int irx_arrow_init_int32_schema(struct ArrowSchema* schema) {
  ArrowSchemaInit(schema);

  const int code = ArrowSchemaSetType(schema, NANOARROW_TYPE_INT32);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowSchemaSetType(int32) failed with error code %d",
        code);
  }

  return NANOARROW_OK;
}

static int irx_arrow_init_int32_array(
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

static int irx_arrow_validate_int32_view(
    const struct ArrowArray* array,
    const struct ArrowSchema* schema,
    struct ArrowArrayView* out_view) {
  struct ArrowError error;
  memset(&error, 0, sizeof(error));
  ArrowArrayViewInitFromType(out_view, NANOARROW_TYPE_UNINITIALIZED);

  int code = ArrowArrayViewInitFromSchema(out_view, schema, &error);
  if (code != NANOARROW_OK) {
    ArrowArrayViewReset(out_view);
    return irx_arrow_capture_nanoarrow_error(
        code,
        &error,
        "ArrowArrayViewInitFromSchema");
  }

  if (out_view->storage_type != NANOARROW_TYPE_INT32) {
    ArrowArrayViewReset(out_view);
    return irx_arrow_set_error_code(
        EINVAL,
        "Only Arrow int32 arrays are supported in this phase");
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

  if (ArrowArrayViewComputeNullCount(out_view) != 0) {
    ArrowArrayViewReset(out_view);
    return irx_arrow_set_error_code(
        EINVAL,
        "Nullable Arrow arrays are not supported in this phase");
  }

  return NANOARROW_OK;
}

static int irx_arrow_copy_int32_view(
    const struct ArrowArrayView* view,
    struct ArrowSchema* out_schema,
    struct ArrowArray* out_array) {
  struct ArrowError error;
  memset(&error, 0, sizeof(error));
  memset(out_schema, 0, sizeof(*out_schema));
  memset(out_array, 0, sizeof(*out_array));

  int code = irx_arrow_init_int32_schema(out_schema);
  if (code != NANOARROW_OK) {
    return code;
  }

  code = irx_arrow_init_int32_array(out_schema, out_array);
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
    code = ArrowArrayAppendInt(out_array, ArrowArrayViewGetIntUnsafe(view, i));
    if (code != NANOARROW_OK) {
      irx_arrow_release_array(out_array);
      irx_arrow_release_schema(out_schema);
      return irx_arrow_set_error_code(
          code,
          "ArrowArrayAppendInt failed with error code %d",
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

int irx_arrow_array_builder_int32_new(
    irx_arrow_array_builder_handle** out_builder) {
  irx_arrow_clear_error();
  if (out_builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_builder must not be NULL");
  }

  irx_arrow_array_builder_handle* builder =
      (irx_arrow_array_builder_handle*)malloc(sizeof(*builder));
  if (builder == NULL) {
    return irx_arrow_set_error_code(ENOMEM, "failed to allocate Arrow builder");
  }

  memset(builder, 0, sizeof(*builder));

  int code = irx_arrow_init_int32_schema(&builder->schema);
  if (code != NANOARROW_OK) {
    free(builder);
    return code;
  }

  code = irx_arrow_init_int32_array(&builder->schema, &builder->array);
  if (code != NANOARROW_OK) {
    irx_arrow_release_schema(&builder->schema);
    free(builder);
    return code;
  }

  builder->type_id = IRX_ARROW_TYPE_INT32;
  *out_builder = builder;
  return NANOARROW_OK;
}

static int irx_arrow_init_float32_schema(struct ArrowSchema* schema) {
  ArrowSchemaInit(schema);

  const int code = ArrowSchemaSetType(schema, NANOARROW_TYPE_FLOAT);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowSchemaSetType(float) failed with error code %d",
        code);
  }

  return NANOARROW_OK;
}

static int irx_arrow_init_float32_array(
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

int irx_arrow_array_builder_float32_new(
    irx_arrow_array_builder_handle** out_builder) {
  irx_arrow_clear_error();
  if (out_builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_builder must not be NULL");
  }

  irx_arrow_array_builder_handle* builder =
      (irx_arrow_array_builder_handle*)malloc(sizeof(*builder));
  if (builder == NULL) {
    return irx_arrow_set_error_code(ENOMEM, "failed to allocate Arrow builder");
  }

  memset(builder, 0, sizeof(*builder));

  int code = irx_arrow_init_float32_schema(&builder->schema);
  if (code != NANOARROW_OK) {
    free(builder);
    return code;
  }

  code = irx_arrow_init_float32_array(&builder->schema, &builder->array);
  if (code != NANOARROW_OK) {
    irx_arrow_release_schema(&builder->schema);
    free(builder);
    return code;
  }

  builder->type_id = IRX_ARROW_TYPE_FLOAT32;
  *out_builder = builder;
  return NANOARROW_OK;
}

int irx_arrow_array_builder_append_float32(
    irx_arrow_array_builder_handle* builder,
    float value) {
  irx_arrow_clear_error();
  if (builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "builder must not be NULL");
  }

  const int code = ArrowArrayAppendDouble(&builder->array, (double)value);
  if (code != NANOARROW_OK) {
    return irx_arrow_set_error_code(
        code,
        "ArrowArrayAppendDouble(float32) failed with error code %d",
        code);
  }

  return NANOARROW_OK;
}

int irx_arrow_array_builder_append_int32(
    irx_arrow_array_builder_handle* builder,
    int32_t value) {
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

int irx_arrow_array_builder_finish(
    irx_arrow_array_builder_handle* builder,
    irx_arrow_array_handle** out_array) {
  struct ArrowError error;
  memset(&error, 0, sizeof(error));
  irx_arrow_clear_error();

  if (builder == NULL) {
    return irx_arrow_set_error_code(EINVAL, "builder must not be NULL");
  }

  if (out_array == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_array must not be NULL");
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
  ArrowSchemaMove(&builder->schema, &array->schema);
  ArrowArrayMove(&builder->array, &array->array);
  array->type_id = builder->type_id;

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

int irx_arrow_array_export(
    const irx_arrow_array_handle* array,
    struct ArrowArray* out_array,
    struct ArrowSchema* out_schema) {
  struct ArrowArrayView view;
  irx_arrow_clear_error();

  if (array == NULL) {
    return irx_arrow_set_error_code(EINVAL, "array must not be NULL");
  }

  if (out_array == NULL || out_schema == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "out_array and out_schema must not be NULL");
  }

  const int code = irx_arrow_validate_int32_view(
      &array->array,
      &array->schema,
      &view);
  if (code != NANOARROW_OK) {
    return code;
  }

  const int copy_code = irx_arrow_copy_int32_view(&view, out_schema, out_array);
  ArrowArrayViewReset(&view);
  return copy_code;
}

int irx_arrow_array_import(
    const struct ArrowArray* array,
    const struct ArrowSchema* schema,
    irx_arrow_array_handle** out_array) {
  struct ArrowArrayView view;
  irx_arrow_clear_error();

  if (array == NULL || schema == NULL) {
    return irx_arrow_set_error_code(
        EINVAL,
        "array and schema must not be NULL");
  }

  if (out_array == NULL) {
    return irx_arrow_set_error_code(EINVAL, "out_array must not be NULL");
  }

  int code = irx_arrow_validate_int32_view(array, schema, &view);
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
  code = irx_arrow_copy_int32_view(&view, &handle->schema, &handle->array);
  ArrowArrayViewReset(&view);
  if (code != NANOARROW_OK) {
    free(handle);
    return code;
  }

  handle->type_id = IRX_ARROW_TYPE_INT32;
  *out_array = handle;
  return NANOARROW_OK;
}

void irx_arrow_array_release(irx_arrow_array_handle* array) {
  if (array == NULL) {
    return;
  }

  irx_arrow_release_array(&array->array);
  irx_arrow_release_schema(&array->schema);
  free(array);
}

const char* irx_arrow_last_error(void) {
  return irx_arrow_last_error_buffer;
}

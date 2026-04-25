// Copyright IRx contributors.

#include "irx_arrow_runtime.h"

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/tensor.h>

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kArrowOk = 0;
constexpr int64_t kInitialRefcount = 1;
constexpr int64_t kPrimitiveArrayBufferCount = 2;

thread_local char last_error[512] = {0};

enum class AppendKind {
  kSigned,
  kUnsigned,
  kDouble,
  kBool,
};

struct TypeSpec {
  int32_t type_id;
  arrow::Type::type arrow_type_id;
  uintptr_t dtype_token;
  int64_t element_size_bytes;
  bool buffer_view_compatible;
  AppendKind append_kind;
  const char* name;
  const char* c_data_format;
  std::shared_ptr<arrow::DataType> (*make_type)();
};

std::shared_ptr<arrow::DataType> make_int8_type() { return arrow::int8(); }
std::shared_ptr<arrow::DataType> make_int16_type() { return arrow::int16(); }
std::shared_ptr<arrow::DataType> make_int32_type() { return arrow::int32(); }
std::shared_ptr<arrow::DataType> make_int64_type() { return arrow::int64(); }
std::shared_ptr<arrow::DataType> make_uint8_type() { return arrow::uint8(); }
std::shared_ptr<arrow::DataType> make_uint16_type() { return arrow::uint16(); }
std::shared_ptr<arrow::DataType> make_uint32_type() { return arrow::uint32(); }
std::shared_ptr<arrow::DataType> make_uint64_type() { return arrow::uint64(); }
std::shared_ptr<arrow::DataType> make_float32_type() { return arrow::float32(); }
std::shared_ptr<arrow::DataType> make_float64_type() { return arrow::float64(); }
std::shared_ptr<arrow::DataType> make_bool_type() { return arrow::boolean(); }

const TypeSpec kTypeSpecs[] = {
    {
        IRX_ARROW_TYPE_INT32,
        arrow::Type::INT32,
        IRX_BUFFER_DTYPE_INT32,
        4,
        true,
        AppendKind::kSigned,
        "int32",
        "i",
        make_int32_type,
    },
    {
        IRX_ARROW_TYPE_INT8,
        arrow::Type::INT8,
        IRX_BUFFER_DTYPE_INT8,
        1,
        true,
        AppendKind::kSigned,
        "int8",
        "c",
        make_int8_type,
    },
    {
        IRX_ARROW_TYPE_INT16,
        arrow::Type::INT16,
        IRX_BUFFER_DTYPE_INT16,
        2,
        true,
        AppendKind::kSigned,
        "int16",
        "s",
        make_int16_type,
    },
    {
        IRX_ARROW_TYPE_INT64,
        arrow::Type::INT64,
        IRX_BUFFER_DTYPE_INT64,
        8,
        true,
        AppendKind::kSigned,
        "int64",
        "l",
        make_int64_type,
    },
    {
        IRX_ARROW_TYPE_UINT8,
        arrow::Type::UINT8,
        IRX_BUFFER_DTYPE_UINT8,
        1,
        true,
        AppendKind::kUnsigned,
        "uint8",
        "C",
        make_uint8_type,
    },
    {
        IRX_ARROW_TYPE_UINT16,
        arrow::Type::UINT16,
        IRX_BUFFER_DTYPE_UINT16,
        2,
        true,
        AppendKind::kUnsigned,
        "uint16",
        "S",
        make_uint16_type,
    },
    {
        IRX_ARROW_TYPE_UINT32,
        arrow::Type::UINT32,
        IRX_BUFFER_DTYPE_UINT32,
        4,
        true,
        AppendKind::kUnsigned,
        "uint32",
        "I",
        make_uint32_type,
    },
    {
        IRX_ARROW_TYPE_UINT64,
        arrow::Type::UINT64,
        IRX_BUFFER_DTYPE_UINT64,
        8,
        true,
        AppendKind::kUnsigned,
        "uint64",
        "L",
        make_uint64_type,
    },
    {
        IRX_ARROW_TYPE_FLOAT32,
        arrow::Type::FLOAT,
        IRX_BUFFER_DTYPE_FLOAT32,
        4,
        true,
        AppendKind::kDouble,
        "float32",
        "f",
        make_float32_type,
    },
    {
        IRX_ARROW_TYPE_FLOAT64,
        arrow::Type::DOUBLE,
        IRX_BUFFER_DTYPE_FLOAT64,
        8,
        true,
        AppendKind::kDouble,
        "float64",
        "g",
        make_float64_type,
    },
    {
        IRX_ARROW_TYPE_BOOL,
        arrow::Type::BOOL,
        IRX_BUFFER_DTYPE_BOOL,
        0,
        false,
        AppendKind::kBool,
        "bool",
        "b",
        make_bool_type,
    },
};

struct ResolvedSchema {
  const TypeSpec* spec = nullptr;
  bool nullable = false;
};

}  // namespace

struct irx_arrow_schema_handle {
  int64_t refcount = 0;
  std::shared_ptr<arrow::Field> field;
  int32_t type_id = IRX_ARROW_TYPE_UNKNOWN;
  int32_t nullable = 0;
};

struct irx_arrow_array_builder_handle {
  std::unique_ptr<arrow::ArrayBuilder> builder;
  int32_t type_id = IRX_ARROW_TYPE_UNKNOWN;
};

struct irx_arrow_array_handle {
  int64_t refcount = 0;
  std::shared_ptr<arrow::Array> array;
  int32_t type_id = IRX_ARROW_TYPE_UNKNOWN;
  int32_t nullable = 0;
  uintptr_t dtype_token = 0;
  int64_t element_size_bytes = 0;
  int32_t buffer_view_compatible = 0;
  int64_t shape[1] = {0};
  int64_t strides[1] = {0};
};

struct irx_arrow_tensor_builder_handle {
  int32_t type_id = IRX_ARROW_TYPE_UNKNOWN;
  int32_t ndim = 0;
  int64_t element_count = 0;
  int64_t values_appended = 0;
  uintptr_t dtype_token = 0;
  int64_t element_size_bytes = 0;
  std::shared_ptr<arrow::DataType> type;
  std::vector<uint8_t> data;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

struct irx_arrow_tensor_handle {
  int64_t refcount = 0;
  std::shared_ptr<arrow::Tensor> tensor;
  std::vector<int64_t> shape_cache;
  std::vector<int64_t> strides_cache;
  int32_t type_id = IRX_ARROW_TYPE_UNKNOWN;
  uintptr_t dtype_token = 0;
  int64_t element_size_bytes = 0;
};

namespace {

void clear_error() { last_error[0] = '\0'; }

int set_error(int code, const char* format, ...) {
  va_list args;
  va_start(args, format);
  std::vsnprintf(last_error, sizeof(last_error), format, args);
  va_end(args);
  return code;
}

int set_arrow_error(int code, const char* context, const arrow::Status& status) {
  return set_error(code, "%s: %s", context, status.ToString().c_str());
}

int set_exception_error(const char* context, const std::exception& exc) {
  return set_error(EINVAL, "%s: %s", context, exc.what());
}

const TypeSpec* type_spec_from_type_id(int32_t type_id) {
  for (const TypeSpec& spec : kTypeSpecs) {
    if (spec.type_id == type_id) {
      return &spec;
    }
  }
  return nullptr;
}

const TypeSpec* type_spec_from_arrow_type_id(arrow::Type::type type_id) {
  for (const TypeSpec& spec : kTypeSpecs) {
    if (spec.arrow_type_id == type_id) {
      return &spec;
    }
  }
  return nullptr;
}

const TypeSpec* type_spec_from_c_data_format(const char* format) {
  if (format == nullptr) {
    return nullptr;
  }
  for (const TypeSpec& spec : kTypeSpecs) {
    if (std::strcmp(format, spec.c_data_format) == 0) {
      return &spec;
    }
  }
  return nullptr;
}

int validate_supported_c_schema(
    const ArrowSchema* schema,
    ResolvedSchema* out_resolved) {
  if (schema == nullptr) {
    return set_error(EINVAL, "schema must not be NULL");
  }
  if (schema->n_children != 0 || schema->dictionary != nullptr) {
    return set_error(
        EINVAL,
        "Only plain primitive Arrow arrays are supported in this phase");
  }

  const TypeSpec* spec = type_spec_from_c_data_format(schema->format);
  if (spec == nullptr) {
    return set_error(
        EINVAL,
        "Unsupported Arrow storage type; supported types are bool, "
        "int8, int16, int32, int64, uint8, uint16, uint32, uint64, "
        "float32, and float64");
  }

  out_resolved->spec = spec;
  out_resolved->nullable = (schema->flags & ARROW_FLAG_NULLABLE) != 0;
  return kArrowOk;
}

int populate_array_metadata(
    irx_arrow_array_handle* handle,
    const ResolvedSchema& resolved) {
  if (!handle->array) {
    return set_error(EINVAL, "array handle has no Arrow array");
  }

  handle->type_id = resolved.spec->type_id;
  handle->nullable = resolved.nullable ? 1 : 0;
  handle->dtype_token = resolved.spec->dtype_token;
  handle->element_size_bytes = resolved.spec->element_size_bytes;
  handle->buffer_view_compatible = resolved.spec->buffer_view_compatible ? 1 : 0;
  handle->shape[0] = handle->array->length();
  handle->strides[0] = resolved.spec->element_size_bytes;
  return kArrowOk;
}

ResolvedSchema resolved_from_arrow_type(
    const std::shared_ptr<arrow::DataType>& type,
    bool nullable) {
  ResolvedSchema resolved;
  resolved.spec = type_spec_from_arrow_type_id(type->id());
  resolved.nullable = nullable;
  return resolved;
}

bool c_data_bit_is_set(const void* data, int64_t bit_index) {
  if (data == nullptr) {
    return false;
  }
  const auto* bytes = static_cast<const uint8_t*>(data);
  const uint8_t mask = static_cast<uint8_t>(1U << (bit_index & 7));
  return (bytes[bit_index >> 3] & mask) != 0;
}

bool c_data_value_is_valid(const ArrowArray* array, int64_t logical_index) {
  if (array->null_count == 0 || array->buffers == nullptr ||
      array->buffers[0] == nullptr) {
    return true;
  }
  return c_data_bit_is_set(array->buffers[0], logical_index);
}

void noop_arrow_array_release(ArrowArray* array) {
  if (array == nullptr) {
    return;
  }
  array->release = nullptr;
}

int validate_c_data_array_layout(
    const ArrowArray* array,
    const ResolvedSchema& resolved) {
  if (array == nullptr) {
    return set_error(EINVAL, "array must not be NULL");
  }
  if (array->release == nullptr) {
    return set_error(
        EINVAL,
        "Arrow array release callback must not be NULL");
  }
  if (array->length < 0 || array->offset < 0) {
    return set_error(
        EINVAL,
        "Arrow array length and offset must be non-negative");
  }
  if (array->null_count < -1) {
    return set_error(EINVAL, "Arrow array null_count must be -1 or greater");
  }
  if (array->null_count > array->length) {
    return set_error(
        EINVAL,
        "Arrow array null_count must not exceed array length");
  }
  if (!resolved.nullable && array->null_count != 0) {
    return set_error(
        EINVAL,
        "non-nullable Arrow schema cannot import nullable array data");
  }
  if (array->n_children != 0 || array->dictionary != nullptr) {
    return set_error(
        EINVAL,
        "Only plain primitive Arrow arrays are supported in this phase");
  }
  if (array->n_buffers < kPrimitiveArrayBufferCount) {
    return set_error(
        EINVAL,
        "Arrow array n_buffers is smaller than the primitive layout requires");
  }
  if (array->buffers == nullptr) {
    return set_error(EINVAL, "Arrow array buffers must not be NULL");
  }
  if (array->null_count > 0 && array->buffers[0] == nullptr) {
    return set_error(
        EINVAL,
        "Arrow array validity bitmap must not be NULL when null_count is "
        "positive");
  }
  if (array->length > 0 && array->buffers[1] == nullptr) {
    return set_error(
        EINVAL,
        "Arrow array value buffer must not be NULL for non-empty arrays");
  }
  if (array->offset > std::numeric_limits<int64_t>::max() - array->length) {
    return set_error(
        EOVERFLOW,
        "Arrow array logical range overflowed int64");
  }
  if (resolved.spec->element_size_bytes > 0 && array->length > 0) {
    const int64_t logical_end = array->offset + array->length - 1;
    if (logical_end >
        std::numeric_limits<int64_t>::max() /
            resolved.spec->element_size_bytes) {
      return set_error(
          EOVERFLOW,
          "Arrow array value-buffer byte offset overflowed int64");
    }
  }
  return kArrowOk;
}

int validate_c_data_array_with_arrow_cpp(
    const ArrowArray* array,
    const ResolvedSchema& resolved) {
  int code = validate_c_data_array_layout(array, resolved);
  if (code != kArrowOk) {
    return code;
  }

  ArrowArray temporary = *array;
  temporary.release = noop_arrow_array_release;
  arrow::Result<std::shared_ptr<arrow::Array>> import_result =
      arrow::ImportArray(&temporary, resolved.spec->make_type());
  if (!import_result.ok()) {
    return set_arrow_error(
        EINVAL,
        "Arrow array validation import failed",
        import_result.status());
  }

  std::shared_ptr<arrow::Array> imported =
      std::move(import_result).ValueUnsafe();
  const arrow::Status status = imported->ValidateFull();
  if (!status.ok()) {
    return set_arrow_error(
        EINVAL,
        "Arrow array validation failed",
        status);
  }
  return kArrowOk;
}

template <typename Builder, typename Value>
int append_typed_value(arrow::ArrayBuilder* builder, Value value) {
  auto* typed_builder = dynamic_cast<Builder*>(builder);
  if (typed_builder == nullptr) {
    return set_error(EINVAL, "array builder element type mismatch");
  }
  const arrow::Status status = typed_builder->Append(value);
  if (!status.ok()) {
    return set_arrow_error(EINVAL, "Arrow array append failed", status);
  }
  return kArrowOk;
}

int append_int_value(arrow::ArrayBuilder* builder, int32_t type_id, int64_t value) {
  switch (type_id) {
    case IRX_ARROW_TYPE_INT8:
      return append_typed_value<arrow::Int8Builder>(builder, static_cast<int8_t>(value));
    case IRX_ARROW_TYPE_INT16:
      return append_typed_value<arrow::Int16Builder>(builder, static_cast<int16_t>(value));
    case IRX_ARROW_TYPE_INT32:
      return append_typed_value<arrow::Int32Builder>(builder, static_cast<int32_t>(value));
    case IRX_ARROW_TYPE_INT64:
      return append_typed_value<arrow::Int64Builder>(builder, static_cast<int64_t>(value));
    case IRX_ARROW_TYPE_BOOL:
      return append_typed_value<arrow::BooleanBuilder>(builder, value != 0);
    default:
      return set_error(EINVAL, "array builder expected a signed integer element type");
  }
}

int append_uint_value(
    arrow::ArrayBuilder* builder,
    int32_t type_id,
    uint64_t value) {
  switch (type_id) {
    case IRX_ARROW_TYPE_UINT8:
      return append_typed_value<arrow::UInt8Builder>(builder, static_cast<uint8_t>(value));
    case IRX_ARROW_TYPE_UINT16:
      return append_typed_value<arrow::UInt16Builder>(builder, static_cast<uint16_t>(value));
    case IRX_ARROW_TYPE_UINT32:
      return append_typed_value<arrow::UInt32Builder>(builder, static_cast<uint32_t>(value));
    case IRX_ARROW_TYPE_UINT64:
      return append_typed_value<arrow::UInt64Builder>(builder, static_cast<uint64_t>(value));
    default:
      return set_error(EINVAL, "array builder expected an unsigned integer element type");
  }
}

int append_double_value(
    arrow::ArrayBuilder* builder,
    int32_t type_id,
    double value) {
  switch (type_id) {
    case IRX_ARROW_TYPE_FLOAT32:
      return append_typed_value<arrow::FloatBuilder>(builder, static_cast<float>(value));
    case IRX_ARROW_TYPE_FLOAT64:
      return append_typed_value<arrow::DoubleBuilder>(builder, value);
    default:
      return set_error(EINVAL, "array builder expected a floating element type");
  }
}

int append_c_data_value(
    arrow::ArrayBuilder* builder,
    const TypeSpec* spec,
    const ArrowArray* array,
    int64_t logical_index) {
  if (!c_data_value_is_valid(array, logical_index)) {
    const arrow::Status status = builder->AppendNull();
    if (!status.ok()) {
      return set_arrow_error(EINVAL, "Arrow array null append failed", status);
    }
    return kArrowOk;
  }

  if (array->buffers == nullptr || array->buffers[1] == nullptr) {
    return set_error(EINVAL, "Arrow array value buffer must not be NULL");
  }

  const auto* data = static_cast<const uint8_t*>(array->buffers[1]);
  const uint8_t* slot = data + logical_index * spec->element_size_bytes;

  switch (spec->type_id) {
    case IRX_ARROW_TYPE_INT8:
      return append_int_value(builder, spec->type_id, *reinterpret_cast<const int8_t*>(slot));
    case IRX_ARROW_TYPE_INT16:
      return append_int_value(builder, spec->type_id, *reinterpret_cast<const int16_t*>(slot));
    case IRX_ARROW_TYPE_INT32:
      return append_int_value(builder, spec->type_id, *reinterpret_cast<const int32_t*>(slot));
    case IRX_ARROW_TYPE_INT64:
      return append_int_value(builder, spec->type_id, *reinterpret_cast<const int64_t*>(slot));
    case IRX_ARROW_TYPE_UINT8:
      return append_uint_value(builder, spec->type_id, *reinterpret_cast<const uint8_t*>(slot));
    case IRX_ARROW_TYPE_UINT16:
      return append_uint_value(builder, spec->type_id, *reinterpret_cast<const uint16_t*>(slot));
    case IRX_ARROW_TYPE_UINT32:
      return append_uint_value(builder, spec->type_id, *reinterpret_cast<const uint32_t*>(slot));
    case IRX_ARROW_TYPE_UINT64:
      return append_uint_value(builder, spec->type_id, *reinterpret_cast<const uint64_t*>(slot));
    case IRX_ARROW_TYPE_FLOAT32:
      return append_double_value(builder, spec->type_id, *reinterpret_cast<const float*>(slot));
    case IRX_ARROW_TYPE_FLOAT64:
      return append_double_value(builder, spec->type_id, *reinterpret_cast<const double*>(slot));
    case IRX_ARROW_TYPE_BOOL:
      return append_typed_value<arrow::BooleanBuilder>(
          builder,
          c_data_bit_is_set(array->buffers[1], logical_index));
    default:
      return set_error(EINVAL, "unsupported Arrow array storage type");
  }
}

int build_array_copy_from_c_data(
    const ArrowArray* array,
    const ResolvedSchema& resolved,
    std::shared_ptr<arrow::Array>* out_array) {
  int code = validate_c_data_array_with_arrow_cpp(array, resolved);
  if (code != kArrowOk) {
    return code;
  }

  arrow::Result<std::unique_ptr<arrow::ArrayBuilder>> builder_result =
      arrow::MakeBuilder(resolved.spec->make_type());
  if (!builder_result.ok()) {
    return set_arrow_error(EINVAL, "Arrow builder allocation failed", builder_result.status());
  }

  std::unique_ptr<arrow::ArrayBuilder> builder = std::move(builder_result).ValueUnsafe();
  arrow::Status status = builder->Reserve(array->length);
  if (!status.ok()) {
    return set_arrow_error(EINVAL, "Arrow builder reserve failed", status);
  }

  for (int64_t index = 0; index < array->length; ++index) {
    code = append_c_data_value(
        builder.get(),
        resolved.spec,
        array,
        array->offset + index);
    if (code != kArrowOk) {
      return code;
    }
  }

  status = builder->Finish(out_array);
  if (!status.ok()) {
    return set_arrow_error(EINVAL, "Arrow builder finish failed", status);
  }
  return kArrowOk;
}

int checked_offset_bytes(
    int64_t offset,
    int64_t element_size_bytes,
    int64_t* out_offset_bytes) {
  if (offset < 0 || element_size_bytes < 0) {
    return set_error(
        EINVAL,
        "buffer view offset computation requires non-negative values");
  }
  if (offset > 0 && element_size_bytes > std::numeric_limits<int64_t>::max() / offset) {
    return set_error(
        EOVERFLOW,
        "Arrow array offset overflowed buffer view byte offset");
  }
  *out_offset_bytes = offset * element_size_bytes;
  return kArrowOk;
}

bool array_has_validity_buffer(const irx_arrow_array_handle* array) {
  if (array == nullptr || !array->array) {
    return false;
  }
  const std::shared_ptr<arrow::ArrayData>& data = array->array->data();
  return data && !data->buffers.empty() && data->buffers[0] != nullptr;
}

int64_t tensor_shape_extent(
    int32_t ndim,
    const int64_t* shape,
    int64_t* out_element_count) {
  int64_t element_count = 1;

  if (ndim < 0) {
    return set_error(EINVAL, "tensor ndim must be non-negative");
  }
  if (ndim > 0 && shape == nullptr) {
    return set_error(
        EINVAL,
        "tensor shape must not be NULL when ndim is positive");
  }

  for (int32_t axis = 0; axis < ndim; ++axis) {
    const int64_t dim = shape[axis];
    if (dim < 0) {
      return set_error(EINVAL, "tensor shape dimensions must be non-negative");
    }
    if (dim != 0 && element_count > std::numeric_limits<int64_t>::max() / dim) {
      return set_error(EOVERFLOW, "tensor shape extent overflowed int64");
    }
    element_count *= dim;
  }

  *out_element_count = element_count;
  return kArrowOk;
}

int copy_tensor_layout(
    int32_t ndim,
    const int64_t* shape,
    const int64_t* strides,
    int64_t element_size_bytes,
    std::vector<int64_t>* out_shape,
    std::vector<int64_t>* out_strides) {
  out_shape->clear();
  out_strides->clear();

  if (ndim == 0) {
    return kArrowOk;
  }

  out_shape->assign(shape, shape + ndim);
  out_strides->resize(static_cast<size_t>(ndim));

  if (strides != nullptr) {
    for (int32_t axis = 0; axis < ndim; ++axis) {
      if (strides[axis] < 0) {
        return set_error(EINVAL, "tensor strides must be non-negative");
      }
      (*out_strides)[static_cast<size_t>(axis)] = strides[axis];
    }
    return kArrowOk;
  }

  int64_t stride = element_size_bytes;
  for (int32_t axis = ndim - 1; axis >= 0; --axis) {
    (*out_strides)[static_cast<size_t>(axis)] = stride;
    const int64_t dim = (*out_shape)[static_cast<size_t>(axis)] > 1
                            ? (*out_shape)[static_cast<size_t>(axis)]
                            : 1;
    if (stride > 0 && dim > std::numeric_limits<int64_t>::max() / stride) {
      return set_error(EOVERFLOW, "tensor default stride computation overflowed int64");
    }
    stride *= dim;
  }

  return kArrowOk;
}

int tensor_data_nbytes(
    int64_t element_count,
    int64_t element_size_bytes,
    int64_t* out_data_nbytes) {
  if (element_size_bytes <= 0) {
    return set_error(
        EINVAL,
        "tensor element type must have a positive byte width");
  }
  if (element_count > 0 &&
      element_count > std::numeric_limits<int64_t>::max() / element_size_bytes) {
    return set_error(EOVERFLOW, "tensor data size overflowed int64");
  }
  *out_data_nbytes = element_count * element_size_bytes;
  return kArrowOk;
}

int tensor_builder_require_slot(
    irx_arrow_tensor_builder_handle* builder,
    uint8_t** out_slot) {
  if (builder == nullptr) {
    return set_error(EINVAL, "tensor builder must not be NULL");
  }
  if (builder->values_appended >= builder->element_count) {
    return set_error(EINVAL, "too many values appended to tensor builder");
  }
  *out_slot = builder->data.data() + builder->values_appended * builder->element_size_bytes;
  builder->values_appended += 1;
  return kArrowOk;
}

template <typename T>
void write_tensor_slot(uint8_t* slot, T value) {
  std::memcpy(slot, &value, sizeof(value));
}

bool tensor_is_c_contiguous(const irx_arrow_tensor_handle* tensor) {
  int64_t stride = tensor->element_size_bytes;
  for (int32_t axis = static_cast<int32_t>(tensor->shape_cache.size()) - 1; axis >= 0; --axis) {
    const size_t index = static_cast<size_t>(axis);
    if (tensor->strides_cache[index] != stride) {
      return false;
    }
    const int64_t dim = tensor->shape_cache[index] > 1 ? tensor->shape_cache[index] : 1;
    if (stride > 0 && dim > std::numeric_limits<int64_t>::max() / stride) {
      return false;
    }
    stride *= dim;
  }
  return true;
}

bool tensor_is_f_contiguous(const irx_arrow_tensor_handle* tensor) {
  int64_t stride = tensor->element_size_bytes;
  for (size_t axis = 0; axis < tensor->shape_cache.size(); ++axis) {
    if (tensor->strides_cache[axis] != stride) {
      return false;
    }
    const int64_t dim = tensor->shape_cache[axis] > 1 ? tensor->shape_cache[axis] : 1;
    if (stride > 0 && dim > std::numeric_limits<int64_t>::max() / stride) {
      return false;
    }
    stride *= dim;
  }
  return true;
}

}  // namespace

extern "C" {

int irx_arrow_schema_import_copy(
    const ArrowSchema* schema,
    irx_arrow_schema_handle** out_schema) {
  clear_error();
  try {
    if (out_schema == nullptr) {
      return set_error(EINVAL, "out_schema must not be NULL");
    }
    *out_schema = nullptr;

    ResolvedSchema resolved;
    int code = validate_supported_c_schema(schema, &resolved);
    if (code != kArrowOk) {
      return code;
    }

    auto handle = std::make_unique<irx_arrow_schema_handle>();
    handle->refcount = kInitialRefcount;
    handle->field = arrow::field("", resolved.spec->make_type(), resolved.nullable);
    handle->type_id = resolved.spec->type_id;
    handle->nullable = resolved.nullable ? 1 : 0;

    *out_schema = handle.release();
    return kArrowOk;
  } catch (const std::bad_alloc&) {
    return set_error(ENOMEM, "failed to allocate Arrow schema");
  } catch (const std::exception& exc) {
    return set_exception_error("irx_arrow_schema_import_copy", exc);
  }
}

int irx_arrow_schema_export(
    const irx_arrow_schema_handle* schema,
    ArrowSchema* out_schema) {
  clear_error();
  try {
    if (schema == nullptr) {
      return set_error(EINVAL, "schema must not be NULL");
    }
    if (out_schema == nullptr) {
      return set_error(EINVAL, "out_schema must not be NULL");
    }
    std::memset(out_schema, 0, sizeof(*out_schema));

    const arrow::Status status = arrow::ExportField(*schema->field, out_schema);
    if (!status.ok()) {
      return set_arrow_error(EINVAL, "Arrow schema export failed", status);
    }
    return kArrowOk;
  } catch (const std::exception& exc) {
    return set_exception_error("irx_arrow_schema_export", exc);
  }
}

int32_t irx_arrow_schema_type_id(const irx_arrow_schema_handle* schema) {
  clear_error();
  if (schema == nullptr) {
    set_error(EINVAL, "schema must not be NULL");
    return IRX_ARROW_TYPE_UNKNOWN;
  }
  return schema->type_id;
}

int32_t irx_arrow_schema_is_nullable(const irx_arrow_schema_handle* schema) {
  clear_error();
  if (schema == nullptr) {
    set_error(EINVAL, "schema must not be NULL");
    return 0;
  }
  return schema->nullable;
}

int irx_arrow_schema_retain(irx_arrow_schema_handle* schema) {
  clear_error();
  if (schema == nullptr) {
    return kArrowOk;
  }
  if (schema->refcount <= 0) {
    return set_error(EINVAL, "schema handle is released");
  }
  schema->refcount += 1;
  return kArrowOk;
}

void irx_arrow_schema_release(irx_arrow_schema_handle* schema) {
  if (schema == nullptr || schema->refcount <= 0) {
    return;
  }
  schema->refcount -= 1;
  if (schema->refcount == 0) {
    delete schema;
  }
}

int irx_arrow_array_builder_new(
    int32_t type_id,
    irx_arrow_array_builder_handle** out_builder) {
  clear_error();
  try {
    if (out_builder == nullptr) {
      return set_error(EINVAL, "out_builder must not be NULL");
    }
    *out_builder = nullptr;

    const TypeSpec* spec = type_spec_from_type_id(type_id);
    if (spec == nullptr) {
      return set_error(EINVAL, "unsupported Arrow type id %d", type_id);
    }

    arrow::Result<std::unique_ptr<arrow::ArrayBuilder>> builder_result =
        arrow::MakeBuilder(spec->make_type());
    if (!builder_result.ok()) {
      return set_arrow_error(EINVAL, "Arrow builder allocation failed", builder_result.status());
    }

    auto handle = std::make_unique<irx_arrow_array_builder_handle>();
    handle->builder = std::move(builder_result).ValueUnsafe();
    handle->type_id = type_id;
    *out_builder = handle.release();
    return kArrowOk;
  } catch (const std::bad_alloc&) {
    return set_error(ENOMEM, "failed to allocate Arrow builder");
  } catch (const std::exception& exc) {
    return set_exception_error("irx_arrow_array_builder_new", exc);
  }
}

int irx_arrow_array_builder_append_null(
    irx_arrow_array_builder_handle* builder,
    int64_t count) {
  clear_error();
  if (builder == nullptr) {
    return set_error(EINVAL, "builder must not be NULL");
  }
  if (count < 0) {
    return set_error(EINVAL, "null append count must be non-negative");
  }
  const arrow::Status status = builder->builder->AppendNulls(count);
  if (!status.ok()) {
    return set_arrow_error(EINVAL, "Arrow null append failed", status);
  }
  return kArrowOk;
}

int irx_arrow_array_builder_append_int(
    irx_arrow_array_builder_handle* builder,
    int64_t value) {
  clear_error();
  if (builder == nullptr) {
    return set_error(EINVAL, "builder must not be NULL");
  }
  return append_int_value(builder->builder.get(), builder->type_id, value);
}

int irx_arrow_array_builder_append_uint(
    irx_arrow_array_builder_handle* builder,
    uint64_t value) {
  clear_error();
  if (builder == nullptr) {
    return set_error(EINVAL, "builder must not be NULL");
  }
  return append_uint_value(builder->builder.get(), builder->type_id, value);
}

int irx_arrow_array_builder_append_double(
    irx_arrow_array_builder_handle* builder,
    double value) {
  clear_error();
  if (builder == nullptr) {
    return set_error(EINVAL, "builder must not be NULL");
  }
  return append_double_value(builder->builder.get(), builder->type_id, value);
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
  clear_error();
  try {
    if (builder == nullptr) {
      return set_error(EINVAL, "builder must not be NULL");
    }
    if (out_array == nullptr) {
      return set_error(EINVAL, "out_array must not be NULL");
    }
    *out_array = nullptr;

    std::shared_ptr<arrow::Array> array;
    const arrow::Status status = builder->builder->Finish(&array);
    if (!status.ok()) {
      return set_arrow_error(EINVAL, "Arrow builder finish failed", status);
    }

    const TypeSpec* spec = type_spec_from_type_id(builder->type_id);
    if (spec == nullptr) {
      return set_error(EINVAL, "builder used unsupported Arrow type id %d", builder->type_id);
    }

    auto handle = std::make_unique<irx_arrow_array_handle>();
    handle->refcount = kInitialRefcount;
    handle->array = std::move(array);
    ResolvedSchema resolved{spec, true};
    int code = populate_array_metadata(handle.get(), resolved);
    if (code != kArrowOk) {
      return code;
    }

    delete builder;
    *out_array = handle.release();
    return kArrowOk;
  } catch (const std::bad_alloc&) {
    return set_error(ENOMEM, "failed to allocate Arrow array");
  } catch (const std::exception& exc) {
    return set_exception_error("irx_arrow_array_builder_finish", exc);
  }
}

void irx_arrow_array_builder_release(irx_arrow_array_builder_handle* builder) {
  delete builder;
}

int64_t irx_arrow_array_length(const irx_arrow_array_handle* array) {
  clear_error();
  if (array == nullptr || !array->array) {
    set_error(EINVAL, "array must not be NULL");
    return -1;
  }
  return array->array->length();
}

int64_t irx_arrow_array_offset(const irx_arrow_array_handle* array) {
  clear_error();
  if (array == nullptr || !array->array) {
    set_error(EINVAL, "array must not be NULL");
    return -1;
  }
  return array->array->offset();
}

int64_t irx_arrow_array_null_count(const irx_arrow_array_handle* array) {
  clear_error();
  if (array == nullptr || !array->array) {
    set_error(EINVAL, "array must not be NULL");
    return -1;
  }
  return array->array->null_count();
}

int32_t irx_arrow_array_type_id(const irx_arrow_array_handle* array) {
  clear_error();
  if (array == nullptr) {
    set_error(EINVAL, "array must not be NULL");
    return IRX_ARROW_TYPE_UNKNOWN;
  }
  return array->type_id;
}

int32_t irx_arrow_array_is_nullable(const irx_arrow_array_handle* array) {
  clear_error();
  if (array == nullptr) {
    set_error(EINVAL, "array must not be NULL");
    return 0;
  }
  return array->nullable;
}

int32_t irx_arrow_array_has_validity_bitmap(
    const irx_arrow_array_handle* array) {
  clear_error();
  if (array == nullptr) {
    set_error(EINVAL, "array must not be NULL");
    return 0;
  }
  return array_has_validity_buffer(array) ? 1 : 0;
}

int32_t irx_arrow_array_can_borrow_buffer_view(
    const irx_arrow_array_handle* array) {
  clear_error();
  if (array == nullptr) {
    set_error(EINVAL, "array must not be NULL");
    return 0;
  }
  return array->buffer_view_compatible;
}

int irx_arrow_array_schema_copy(
    const irx_arrow_array_handle* array,
    irx_arrow_schema_handle** out_schema) {
  clear_error();
  try {
    if (array == nullptr || !array->array) {
      return set_error(EINVAL, "array must not be NULL");
    }
    if (out_schema == nullptr) {
      return set_error(EINVAL, "out_schema must not be NULL");
    }
    *out_schema = nullptr;

    ResolvedSchema resolved = resolved_from_arrow_type(array->array->type(), array->nullable != 0);
    if (resolved.spec == nullptr) {
      return set_error(EINVAL, "unsupported Arrow array storage type");
    }

    auto handle = std::make_unique<irx_arrow_schema_handle>();
    handle->refcount = kInitialRefcount;
    handle->field = arrow::field("", array->array->type(), resolved.nullable);
    handle->type_id = resolved.spec->type_id;
    handle->nullable = resolved.nullable ? 1 : 0;
    *out_schema = handle.release();
    return kArrowOk;
  } catch (const std::bad_alloc&) {
    return set_error(ENOMEM, "failed to allocate Arrow schema");
  } catch (const std::exception& exc) {
    return set_exception_error("irx_arrow_array_schema_copy", exc);
  }
}

int irx_arrow_array_export(
    const irx_arrow_array_handle* array,
    ArrowArray* out_array,
    ArrowSchema* out_schema) {
  clear_error();
  try {
    if (array == nullptr || !array->array) {
      return set_error(EINVAL, "array must not be NULL");
    }
    if (out_array == nullptr || out_schema == nullptr) {
      return set_error(EINVAL, "out_array and out_schema must not be NULL");
    }
    std::memset(out_array, 0, sizeof(*out_array));
    std::memset(out_schema, 0, sizeof(*out_schema));

    const arrow::Status status = arrow::ExportArray(*array->array, out_array, out_schema);
    if (!status.ok()) {
      return set_arrow_error(EINVAL, "Arrow array export failed", status);
    }
    return kArrowOk;
  } catch (const std::exception& exc) {
    return set_exception_error("irx_arrow_array_export", exc);
  }
}

int irx_arrow_array_import(
    const ArrowArray* array,
    const ArrowSchema* schema,
    irx_arrow_array_handle** out_array) {
  return irx_arrow_array_import_copy(array, schema, out_array);
}

int irx_arrow_array_import_copy(
    const ArrowArray* array,
    const ArrowSchema* schema,
    irx_arrow_array_handle** out_array) {
  clear_error();
  try {
    if (array == nullptr || schema == nullptr) {
      return set_error(EINVAL, "array and schema must not be NULL");
    }
    if (out_array == nullptr) {
      return set_error(EINVAL, "out_array must not be NULL");
    }
    *out_array = nullptr;

    ResolvedSchema resolved;
    int code = validate_supported_c_schema(schema, &resolved);
    if (code != kArrowOk) {
      return code;
    }

    std::shared_ptr<arrow::Array> copied_array;
    code = build_array_copy_from_c_data(array, resolved, &copied_array);
    if (code != kArrowOk) {
      return code;
    }

    auto handle = std::make_unique<irx_arrow_array_handle>();
    handle->refcount = kInitialRefcount;
    handle->array = std::move(copied_array);
    code = populate_array_metadata(handle.get(), resolved);
    if (code != kArrowOk) {
      return code;
    }

    *out_array = handle.release();
    return kArrowOk;
  } catch (const std::bad_alloc&) {
    return set_error(ENOMEM, "failed to allocate Arrow array");
  } catch (const std::exception& exc) {
    return set_exception_error("irx_arrow_array_import_copy", exc);
  }
}

int irx_arrow_array_import_move(
    ArrowArray* array,
    ArrowSchema* schema,
    irx_arrow_array_handle** out_array) {
  clear_error();
  try {
    if (array == nullptr || schema == nullptr) {
      return set_error(EINVAL, "array and schema must not be NULL");
    }
    if (out_array == nullptr) {
      return set_error(EINVAL, "out_array must not be NULL");
    }
    *out_array = nullptr;

    ResolvedSchema resolved;
    int code = validate_supported_c_schema(schema, &resolved);
    if (code != kArrowOk) {
      return code;
    }

    arrow::Result<std::shared_ptr<arrow::Array>> import_result =
        arrow::ImportArray(array, schema);
    if (!import_result.ok()) {
      return set_arrow_error(EINVAL, "Arrow array import failed", import_result.status());
    }

    std::shared_ptr<arrow::Array> imported = std::move(import_result).ValueUnsafe();
    ResolvedSchema imported_resolved = resolved_from_arrow_type(imported->type(), resolved.nullable);
    if (imported_resolved.spec == nullptr) {
      return set_error(EINVAL, "unsupported Arrow array storage type");
    }

    auto handle = std::make_unique<irx_arrow_array_handle>();
    handle->refcount = kInitialRefcount;
    handle->array = std::move(imported);
    code = populate_array_metadata(handle.get(), imported_resolved);
    if (code != kArrowOk) {
      return code;
    }

    *out_array = handle.release();
    return kArrowOk;
  } catch (const std::bad_alloc&) {
    return set_error(ENOMEM, "failed to allocate Arrow array");
  } catch (const std::exception& exc) {
    return set_exception_error("irx_arrow_array_import_move", exc);
  }
}

int irx_arrow_array_validity_bitmap(
    const irx_arrow_array_handle* array,
    const void** out_data,
    int64_t* out_offset_bits,
    int64_t* out_length_bits) {
  clear_error();
  if (array == nullptr || !array->array) {
    return set_error(EINVAL, "array must not be NULL");
  }
  if (out_data == nullptr || out_offset_bits == nullptr || out_length_bits == nullptr) {
    return set_error(EINVAL, "out_data, out_offset_bits, and out_length_bits must not be NULL");
  }

  *out_data = nullptr;
  *out_offset_bits = 0;
  *out_length_bits = array->array->length();

  if (array_has_validity_buffer(array)) {
    *out_data = array->array->data()->buffers[0]->data();
    *out_offset_bits = array->array->offset();
  }
  return kArrowOk;
}

int irx_arrow_array_borrow_buffer_view(
    const irx_arrow_array_handle* array,
    irx_buffer_view* out_view) {
  clear_error();
  if (array == nullptr || !array->array) {
    return set_error(EINVAL, "array must not be NULL");
  }
  if (out_view == nullptr) {
    return set_error(EINVAL, "out_view must not be NULL");
  }
  if (!array->buffer_view_compatible) {
    return set_error(
        EINVAL,
        "Arrow bool arrays use bit-packed values and cannot be exposed as plain buffer views");
  }

  int64_t offset_bytes = 0;
  int code = checked_offset_bytes(
      array->array->offset(),
      array->element_size_bytes,
      &offset_bytes);
  if (code != kArrowOk) {
    return code;
  }

  const std::shared_ptr<arrow::ArrayData>& data = array->array->data();
  void* value_data = nullptr;
  if (data && data->buffers.size() > 1 && data->buffers[1] != nullptr) {
    value_data = const_cast<uint8_t*>(data->buffers[1]->data());
  }

  std::memset(out_view, 0, sizeof(*out_view));
  out_view->data = value_data;
  out_view->owner = nullptr;
  out_view->dtype = reinterpret_cast<void*>(array->dtype_token);
  out_view->ndim = 1;
  out_view->shape = const_cast<int64_t*>(array->shape);
  out_view->strides = const_cast<int64_t*>(array->strides);
  out_view->offset_bytes = offset_bytes;
  out_view->flags = IRX_BUFFER_FLAG_BORROWED |
                    IRX_BUFFER_FLAG_READONLY |
                    IRX_BUFFER_FLAG_C_CONTIGUOUS;
  if (array_has_validity_buffer(array)) {
    out_view->flags |= IRX_BUFFER_FLAG_VALIDITY_BITMAP;
  }
  return kArrowOk;
}

int irx_arrow_array_retain(irx_arrow_array_handle* array) {
  clear_error();
  if (array == nullptr) {
    return kArrowOk;
  }
  if (array->refcount <= 0) {
    return set_error(EINVAL, "array handle is released");
  }
  array->refcount += 1;
  return kArrowOk;
}

void irx_arrow_array_release(irx_arrow_array_handle* array) {
  if (array == nullptr || array->refcount <= 0) {
    return;
  }
  array->refcount -= 1;
  if (array->refcount == 0) {
    delete array;
  }
}

int irx_arrow_tensor_builder_new(
    int32_t type_id,
    int32_t ndim,
    const int64_t* shape,
    const int64_t* strides,
    irx_arrow_tensor_builder_handle** out_builder) {
  clear_error();
  try {
    if (out_builder == nullptr) {
      return set_error(EINVAL, "out_builder must not be NULL");
    }
    *out_builder = nullptr;

    const TypeSpec* spec = type_spec_from_type_id(type_id);
    if (spec == nullptr) {
      return set_error(EINVAL, "unsupported Arrow tensor type id %d", type_id);
    }
    if (!spec->buffer_view_compatible || spec->element_size_bytes <= 0) {
      return set_error(
          EINVAL,
          "Arrow tensor builder requires a fixed-width primitive value type");
    }

    int64_t element_count = 0;
    int code = static_cast<int>(tensor_shape_extent(ndim, shape, &element_count));
    if (code != kArrowOk) {
      return code;
    }

    int64_t data_nbytes = 0;
    code = tensor_data_nbytes(element_count, spec->element_size_bytes, &data_nbytes);
    if (code != kArrowOk) {
      return code;
    }

    auto builder = std::make_unique<irx_arrow_tensor_builder_handle>();
    builder->type_id = type_id;
    builder->ndim = ndim;
    builder->element_count = element_count;
    builder->dtype_token = spec->dtype_token;
    builder->element_size_bytes = spec->element_size_bytes;
    builder->type = spec->make_type();
    builder->data.assign(static_cast<size_t>(data_nbytes), 0);

    code = copy_tensor_layout(
        ndim,
        shape,
        strides,
        spec->element_size_bytes,
        &builder->shape,
        &builder->strides);
    if (code != kArrowOk) {
      return code;
    }

    *out_builder = builder.release();
    return kArrowOk;
  } catch (const std::bad_alloc&) {
    return set_error(ENOMEM, "failed to allocate Arrow tensor builder");
  } catch (const std::exception& exc) {
    return set_exception_error("irx_arrow_tensor_builder_new", exc);
  }
}

int irx_arrow_tensor_builder_append_int(
    irx_arrow_tensor_builder_handle* builder,
    int64_t value) {
  clear_error();
  uint8_t* slot = nullptr;
  const int code = tensor_builder_require_slot(builder, &slot);
  if (code != kArrowOk) {
    return code;
  }

  switch (builder->type_id) {
    case IRX_ARROW_TYPE_INT8:
      write_tensor_slot(slot, static_cast<int8_t>(value));
      return kArrowOk;
    case IRX_ARROW_TYPE_INT16:
      write_tensor_slot(slot, static_cast<int16_t>(value));
      return kArrowOk;
    case IRX_ARROW_TYPE_INT32:
      write_tensor_slot(slot, static_cast<int32_t>(value));
      return kArrowOk;
    case IRX_ARROW_TYPE_INT64:
      write_tensor_slot(slot, static_cast<int64_t>(value));
      return kArrowOk;
    default:
      builder->values_appended -= 1;
      return set_error(EINVAL, "tensor builder expected a signed integer element type");
  }
}

int irx_arrow_tensor_builder_append_uint(
    irx_arrow_tensor_builder_handle* builder,
    uint64_t value) {
  clear_error();
  uint8_t* slot = nullptr;
  const int code = tensor_builder_require_slot(builder, &slot);
  if (code != kArrowOk) {
    return code;
  }

  switch (builder->type_id) {
    case IRX_ARROW_TYPE_UINT8:
      write_tensor_slot(slot, static_cast<uint8_t>(value));
      return kArrowOk;
    case IRX_ARROW_TYPE_UINT16:
      write_tensor_slot(slot, static_cast<uint16_t>(value));
      return kArrowOk;
    case IRX_ARROW_TYPE_UINT32:
      write_tensor_slot(slot, static_cast<uint32_t>(value));
      return kArrowOk;
    case IRX_ARROW_TYPE_UINT64:
      write_tensor_slot(slot, static_cast<uint64_t>(value));
      return kArrowOk;
    default:
      builder->values_appended -= 1;
      return set_error(EINVAL, "tensor builder expected an unsigned integer element type");
  }
}

int irx_arrow_tensor_builder_append_double(
    irx_arrow_tensor_builder_handle* builder,
    double value) {
  clear_error();
  uint8_t* slot = nullptr;
  const int code = tensor_builder_require_slot(builder, &slot);
  if (code != kArrowOk) {
    return code;
  }

  switch (builder->type_id) {
    case IRX_ARROW_TYPE_FLOAT32:
      write_tensor_slot(slot, static_cast<float>(value));
      return kArrowOk;
    case IRX_ARROW_TYPE_FLOAT64:
      write_tensor_slot(slot, static_cast<double>(value));
      return kArrowOk;
    default:
      builder->values_appended -= 1;
      return set_error(EINVAL, "tensor builder expected a floating element type");
  }
}

int irx_arrow_tensor_builder_finish(
    irx_arrow_tensor_builder_handle* builder,
    irx_arrow_tensor_handle** out_tensor) {
  clear_error();
  try {
    if (builder == nullptr) {
      return set_error(EINVAL, "tensor builder must not be NULL");
    }
    if (out_tensor == nullptr) {
      return set_error(EINVAL, "out_tensor must not be NULL");
    }
    *out_tensor = nullptr;

    if (builder->values_appended != builder->element_count) {
      return set_error(
          EINVAL,
          "tensor builder value count does not match tensor shape extent");
    }

    std::shared_ptr<arrow::Buffer> buffer = arrow::Buffer::FromVector(std::move(builder->data));
    arrow::Result<std::shared_ptr<arrow::Tensor>> tensor_result = arrow::Tensor::Make(
        builder->type,
        buffer,
        builder->shape,
        builder->strides);
    if (!tensor_result.ok()) {
      return set_arrow_error(EINVAL, "Arrow tensor construction failed", tensor_result.status());
    }

    auto tensor = std::make_unique<irx_arrow_tensor_handle>();
    tensor->refcount = kInitialRefcount;
    tensor->tensor = std::move(tensor_result).ValueUnsafe();
    tensor->shape_cache = tensor->tensor->shape();
    tensor->strides_cache = tensor->tensor->strides();
    tensor->type_id = builder->type_id;
    tensor->dtype_token = builder->dtype_token;
    tensor->element_size_bytes = builder->element_size_bytes;

    delete builder;
    *out_tensor = tensor.release();
    return kArrowOk;
  } catch (const std::bad_alloc&) {
    return set_error(ENOMEM, "failed to allocate Arrow tensor handle");
  } catch (const std::exception& exc) {
    return set_exception_error("irx_arrow_tensor_builder_finish", exc);
  }
}

void irx_arrow_tensor_builder_release(
    irx_arrow_tensor_builder_handle* builder) {
  delete builder;
}

int32_t irx_arrow_tensor_type_id(const irx_arrow_tensor_handle* tensor) {
  clear_error();
  if (tensor == nullptr) {
    set_error(EINVAL, "tensor must not be NULL");
    return IRX_ARROW_TYPE_UNKNOWN;
  }
  return tensor->type_id;
}

int32_t irx_arrow_tensor_ndim(const irx_arrow_tensor_handle* tensor) {
  clear_error();
  if (tensor == nullptr || !tensor->tensor) {
    set_error(EINVAL, "tensor must not be NULL");
    return -1;
  }
  return tensor->tensor->ndim();
}

int64_t irx_arrow_tensor_size(const irx_arrow_tensor_handle* tensor) {
  clear_error();
  if (tensor == nullptr || !tensor->tensor) {
    set_error(EINVAL, "tensor must not be NULL");
    return -1;
  }
  return tensor->tensor->size();
}

const int64_t* irx_arrow_tensor_shape(const irx_arrow_tensor_handle* tensor) {
  clear_error();
  if (tensor == nullptr) {
    set_error(EINVAL, "tensor must not be NULL");
    return nullptr;
  }
  return tensor->shape_cache.empty() ? nullptr : tensor->shape_cache.data();
}

const int64_t* irx_arrow_tensor_strides(const irx_arrow_tensor_handle* tensor) {
  clear_error();
  if (tensor == nullptr) {
    set_error(EINVAL, "tensor must not be NULL");
    return nullptr;
  }
  return tensor->strides_cache.empty() ? nullptr : tensor->strides_cache.data();
}

int irx_arrow_tensor_borrow_buffer_view(
    const irx_arrow_tensor_handle* tensor,
    irx_buffer_view* out_view) {
  clear_error();
  if (tensor == nullptr || !tensor->tensor) {
    return set_error(EINVAL, "tensor must not be NULL");
  }
  if (out_view == nullptr) {
    return set_error(EINVAL, "out_view must not be NULL");
  }

  std::memset(out_view, 0, sizeof(*out_view));
  out_view->data = const_cast<uint8_t*>(tensor->tensor->raw_data());
  out_view->owner = nullptr;
  out_view->dtype = reinterpret_cast<void*>(tensor->dtype_token);
  out_view->ndim = static_cast<int32_t>(tensor->shape_cache.size());
  out_view->shape = const_cast<int64_t*>(tensor->shape_cache.data());
  out_view->strides = const_cast<int64_t*>(tensor->strides_cache.data());
  out_view->offset_bytes = 0;
  out_view->flags = IRX_BUFFER_FLAG_BORROWED | IRX_BUFFER_FLAG_READONLY;

  if (tensor_is_c_contiguous(tensor)) {
    out_view->flags |= IRX_BUFFER_FLAG_C_CONTIGUOUS;
  }
  if (tensor_is_f_contiguous(tensor)) {
    out_view->flags |= IRX_BUFFER_FLAG_F_CONTIGUOUS;
  }
  return kArrowOk;
}

int irx_arrow_tensor_retain(irx_arrow_tensor_handle* tensor) {
  clear_error();
  if (tensor == nullptr) {
    return kArrowOk;
  }
  if (tensor->refcount <= 0) {
    return set_error(EINVAL, "tensor handle is released");
  }
  tensor->refcount += 1;
  return kArrowOk;
}

void irx_arrow_tensor_release(irx_arrow_tensor_handle* tensor) {
  if (tensor == nullptr || tensor->refcount <= 0) {
    return;
  }
  tensor->refcount -= 1;
  if (tensor->refcount == 0) {
    delete tensor;
  }
}

const char* irx_arrow_last_error(void) {
  return last_error;
}

}  // extern "C"

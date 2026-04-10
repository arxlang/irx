#ifndef IRX_BUFFER_RUNTIME_H
#define IRX_BUFFER_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IRX_BUFFER_FLAG_BORROWED (1 << 0)
#define IRX_BUFFER_FLAG_OWNED (1 << 1)
#define IRX_BUFFER_FLAG_EXTERNAL_OWNER (1 << 2)
#define IRX_BUFFER_FLAG_READONLY (1 << 3)
#define IRX_BUFFER_FLAG_WRITABLE (1 << 4)
#define IRX_BUFFER_FLAG_C_CONTIGUOUS (1 << 5)
#define IRX_BUFFER_FLAG_F_CONTIGUOUS (1 << 6)

typedef struct irx_buffer_owner_handle irx_buffer_owner_handle;
typedef void (*irx_buffer_owner_release_fn)(void* context);

typedef struct irx_buffer_view {
  void* data;
  irx_buffer_owner_handle* owner;
  void* dtype;
  int32_t ndim;
  int64_t* shape;
  int64_t* strides;
  int64_t offset_bytes;
  int32_t flags;
} irx_buffer_view;

int32_t irx_buffer_owner_external_new(
    void* context,
    irx_buffer_owner_release_fn release,
    irx_buffer_owner_handle** out_owner);
int32_t irx_buffer_owner_retain(irx_buffer_owner_handle* owner);
int32_t irx_buffer_owner_release(irx_buffer_owner_handle* owner);
int32_t irx_buffer_view_retain(const irx_buffer_view* view);
int32_t irx_buffer_view_release(irx_buffer_view* view);
const char* irx_buffer_last_error(void);

#ifdef __cplusplus
}
#endif

#endif

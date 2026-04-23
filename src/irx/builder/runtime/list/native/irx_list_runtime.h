#ifndef IRX_LIST_RUNTIME_H
#define IRX_LIST_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct irx_list {
  uint8_t* data;
  int64_t length;
  int64_t capacity;
  int64_t element_size;
} irx_list;

/* Current v1 ABI intentionally omits a destroy/release helper. */
int32_t irx_list_append(irx_list* list, const void* value);
void* irx_list_at(const irx_list* list, int64_t index);

#ifdef __cplusplus
}
#endif

#endif

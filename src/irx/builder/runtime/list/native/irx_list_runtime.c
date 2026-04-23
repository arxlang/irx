#include "irx_list_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IRX_LIST_MIN_CAPACITY 4

static void irx_list_fail(const char* message) {
  fprintf(stderr, "%s\n", message);
  exit(1);
}

static int64_t irx_list_next_capacity(int64_t current_capacity) {
  if (current_capacity < IRX_LIST_MIN_CAPACITY) {
    return IRX_LIST_MIN_CAPACITY;
  }
  return current_capacity * 2;
}

int32_t irx_list_append(irx_list* list, const void* value) {
  if (list == NULL) {
    irx_list_fail("dynamic list append requires a non-null list");
  }
  if (value == NULL) {
    irx_list_fail("dynamic list append requires a non-null value pointer");
  }
  if (list->element_size <= 0) {
    irx_list_fail("dynamic list append requires a positive element size");
  }

  if (list->length >= list->capacity) {
    int64_t new_capacity = irx_list_next_capacity(list->capacity);
    size_t new_size =
        (size_t)new_capacity * (size_t)list->element_size;
    void* new_data = realloc(list->data, new_size);
    if (new_data == NULL) {
      irx_list_fail("dynamic list append allocation failed");
    }
    list->data = (uint8_t*)new_data;
    list->capacity = new_capacity;
  }

  memcpy(
      list->data + ((size_t)list->length * (size_t)list->element_size),
      value,
      (size_t)list->element_size);
  list->length += 1;
  return 0;
}

void* irx_list_at(const irx_list* list, int64_t index) {
  if (list == NULL) {
    irx_list_fail("dynamic list indexing requires a non-null list");
  }
  if (index < 0 || index >= list->length) {
    irx_list_fail("dynamic list index out of range");
  }
  if (list->data == NULL) {
    irx_list_fail("dynamic list storage is null");
  }
  return list->data + ((size_t)index * (size_t)list->element_size);
}

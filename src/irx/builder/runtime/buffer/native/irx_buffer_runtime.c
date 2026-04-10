#include "irx_buffer_runtime.h"

#include <errno.h>
#include <stdlib.h>

struct irx_buffer_owner_handle {
  int64_t refcount;
  void* context;
  irx_buffer_owner_release_fn release;
};

static const char* irx_buffer_error = "";

static int32_t irx_buffer_set_error(const char* message) {
  irx_buffer_error = message;
  return -1;
}

const char* irx_buffer_last_error(void) { return irx_buffer_error; }

int32_t irx_buffer_owner_external_new(
    void* context,
    irx_buffer_owner_release_fn release,
    irx_buffer_owner_handle** out_owner) {
  if (out_owner == NULL) {
    return irx_buffer_set_error("out_owner must not be NULL");
  }
  *out_owner = NULL;

  irx_buffer_owner_handle* owner =
      (irx_buffer_owner_handle*)malloc(sizeof(*owner));
  if (owner == NULL) {
    return irx_buffer_set_error("failed to allocate buffer owner handle");
  }

  owner->refcount = 1;
  owner->context = context;
  owner->release = release;
  *out_owner = owner;
  irx_buffer_error = "";
  return 0;
}

int32_t irx_buffer_owner_retain(irx_buffer_owner_handle* owner) {
  if (owner == NULL) {
    irx_buffer_error = "";
    return 0;
  }
  if (owner->refcount <= 0) {
    return irx_buffer_set_error("cannot retain a released buffer owner");
  }
  owner->refcount += 1;
  irx_buffer_error = "";
  return 0;
}

int32_t irx_buffer_owner_release(irx_buffer_owner_handle* owner) {
  if (owner == NULL) {
    irx_buffer_error = "";
    return 0;
  }
  if (owner->refcount <= 0) {
    return irx_buffer_set_error("cannot release a released buffer owner");
  }
  owner->refcount -= 1;
  if (owner->refcount == 0) {
    irx_buffer_owner_release_fn release = owner->release;
    void* context = owner->context;
    free(owner);
    if (release != NULL) {
      release(context);
    }
  }
  irx_buffer_error = "";
  return 0;
}

int32_t irx_buffer_view_retain(const irx_buffer_view* view) {
  if (view == NULL) {
    return irx_buffer_set_error("view must not be NULL");
  }
  return irx_buffer_owner_retain(view->owner);
}

int32_t irx_buffer_view_release(irx_buffer_view* view) {
  if (view == NULL) {
    return irx_buffer_set_error("view must not be NULL");
  }
  int32_t code = irx_buffer_owner_release(view->owner);
  if (code == 0) {
    view->owner = NULL;
  }
  return code;
}

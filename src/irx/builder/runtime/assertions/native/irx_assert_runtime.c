#include <stdio.h>
#include <stdlib.h>

void __arx_assert_fail(
    const char* source,
    int line,
    int col,
    const char* message
) {
    const char* safe_source = source != NULL ? source : "<unknown>";
    const char* safe_message =
        message != NULL ? message : "assertion failed";

    fprintf(
        stderr,
        "ARX_ASSERT_FAIL|%s|%d|%d|%s\n",
        safe_source,
        line,
        col,
        safe_message
    );
    fflush(stderr);
    exit(EXIT_FAILURE);
}

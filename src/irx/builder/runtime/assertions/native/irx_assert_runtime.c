#include <stdio.h>
#include <stdlib.h>

static void irx_write_escaped_assert_field(FILE* stream, const char* text) {
  const unsigned char* cursor = (const unsigned char*)text;

  while (*cursor != '\0') {
    switch (*cursor) {
      case '\\':
        fputs("\\\\", stream);
        break;
      case '\n':
        fputs("\\n", stream);
        break;
      case '\r':
        fputs("\\r", stream);
        break;
      case '\t':
        fputs("\\t", stream);
        break;
      case '|':
        fputs("\\p", stream);
        break;
      default:
        fputc(*cursor, stream);
        break;
    }
    cursor++;
  }
}

void __arx_assert_fail(
    const char* source,
    int line,
    int col,
    const char* message
) {
    const char* safe_source = source != NULL ? source : "<unknown>";
    const char* safe_message =
        message != NULL ? message : "assertion failed";

    fputs("ARX_ASSERT_FAIL|", stderr);
    irx_write_escaped_assert_field(stderr, safe_source);
    fprintf(stderr, "|%d|%d|", line, col);
    irx_write_escaped_assert_field(stderr, safe_message);
    fputc('\n', stderr);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

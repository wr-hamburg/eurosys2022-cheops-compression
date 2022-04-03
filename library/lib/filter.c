#include <filter.h>
#include <settings.h>

gboolean filter_IO(size_t buf_size) {
    if (buf_size < opt_min_chunk_size)
        return FALSE;
    return TRUE;
}
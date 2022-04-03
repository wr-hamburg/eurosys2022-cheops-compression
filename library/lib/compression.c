#include <compression.h>
GArray *available_compressors;

static const char *compressor_names[] = {
    [LZ4] = "LZ4", [LZ4_FAST] = "LZ4-fast", [ZSTD] = "ZSTD", [ZLIB] = "ZLIB"};

void init_compressors() {
    available_compressors =
        g_array_new(FALSE, FALSE, sizeof(CompressionAlgorithm));
    g_array_append_val(available_compressors, IOA_lz4);
    g_array_append_val(available_compressors, IOA_lz4_fast);
    g_array_append_val(available_compressors, IOA_zstd);
    g_array_append_val(available_compressors, IOA_zlib);
}

const char *compressor_to_name(CompressionAlgorithmID id) {
    return compressor_names[id];
}

CompressionAlgorithmID name_to_compressor(char *name) {
    for (int i = 0; i < _COMPRESSOR_COUNT; i++) {
        if (strcmp(name, compressor_names[i]) == 0) {
            return i;
        }
    }
    g_printerr("Compressor not found: %s\n", name);
    exit(1);
}
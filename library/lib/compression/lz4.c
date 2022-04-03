#include <compression.h>
#include <lz4.h>
#include <lz4hc.h>

static int level_values[] = {12, 9, 6, 3, 1};

size_t bound_LZ4(size_t length) { return LZ4_compressBound(length); }

size_t compress_LZ4(void *dst, size_t dstCapacity, const void *src,
                    size_t srcSize, int compressionLevel) {
    // LZ4_compress_default
    int ret = LZ4_compress_HC(src, dst, srcSize, dstCapacity, compressionLevel);
    if (ret > 0) {
        return ret;
    }
    g_printerr("COMPRESSION-ERROR: LZ4(%d) Error: %d | srcSize: %ld\n",
               compressionLevel, ret, srcSize);
    return 0;
}

size_t decompress_LZ4(const char *src, char *dst, size_t compressedSize,
                      size_t dstCapacity) {
    int ret = LZ4_decompress_safe(src, dst, compressedSize, dstCapacity);
    if (ret > 0) {
        return ret;
    }
    g_printerr("DECOMPRESSION-ERROR: LZ4 Error: %d | compressedSize: %ld\n",
               ret, compressedSize);
    return 0;
}

CompressionAlgorithm IOA_lz4 = {
    bound_LZ4,    compress_LZ4,           decompress_LZ4,
    level_values, COUNT_OF(level_values), "LZ4",
    LZ4};

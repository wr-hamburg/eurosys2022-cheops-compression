#include <compression.h>
#include <lz4.h>
#include <lz4hc.h>

static int level_values[] = {1, 7, 17};

size_t bound_LZ4_fast(size_t length) { return LZ4_compressBound(length); }

size_t compress_LZ4_fast(void *dst, size_t dstCapacity, const void *src,
                         size_t srcSize, int compressionLevel) {
    int ret =
        LZ4_compress_fast(src, dst, srcSize, dstCapacity, compressionLevel);
    if (ret > 0) {
        return ret;
    }
    g_printerr("COMPRESSION-ERROR: LZ4-fast(%d) Error: %d | srcSize: %ld\n",
               compressionLevel, ret, srcSize);
    return 0;
}

size_t decompress_LZ4_fast(const char *src, char *dst, size_t compressedSize,
                           size_t dstCapacity) {
    int ret = LZ4_decompress_safe(src, dst, compressedSize, dstCapacity);
    if (ret > 0) {
        return ret;
    }
    g_printerr(
        "DECOMPRESSION-ERROR: LZ4-fast Error: %d | compressedSize: %ld\n", ret,
        compressedSize);
    return 0;
}

CompressionAlgorithm IOA_lz4_fast = {
    bound_LZ4_fast, compress_LZ4_fast,      decompress_LZ4_fast,
    level_values,   COUNT_OF(level_values), "LZ4-fast",
    LZ4_FAST};

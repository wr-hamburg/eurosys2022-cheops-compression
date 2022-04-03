#include <compression.h>
#include <zstd.h>

static int level_values[] = {22, 10, 3, 1};

size_t bound_ZSTD(size_t length) { return ZSTD_compressBound(length); }

size_t compress_ZSTD(void *dst, size_t dstCapacity, const void *src,
                     size_t srcSize, int compressionLevel) {
    size_t ret;
    ret = ZSTD_compress(dst, dstCapacity, src, srcSize, compressionLevel);
    if (ZSTD_isError(ret)) {
        g_printerr("COMPRESSION-ERROR: ZSTD(%d) Error: %s | srcSize: %ld\n",
                   compressionLevel, ZSTD_getErrorName(ret), srcSize);
        return 0;
    }
    return ret;
}

size_t decompress_ZSTD(const char *src, char *dst, size_t compressedSize,
                       size_t dstCapacity) {
    int ret = ZSTD_decompress(dst, dstCapacity, src, compressedSize);
    if (ZSTD_isError(ret)) {
        g_printerr(
            "DECOMPRESSION-ERROR: ZSTD, Error: %s | compressedSize: %ld\n",
            ZSTD_getErrorName(ret), compressedSize);
        return 0;
    }
    return ret;
}

CompressionAlgorithm IOA_zstd = {
    bound_ZSTD,   compress_ZSTD,          decompress_ZSTD,
    level_values, COUNT_OF(level_values), "ZSTD",
    ZSTD};

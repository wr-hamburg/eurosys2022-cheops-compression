#include <compression.h>
#include <zlib.h>

static int level_values[] = {9, 6, 3, 1};

size_t bound_ZLIB(size_t length) { return compressBound(length); }

size_t compress_ZLIB(void *dst, size_t dstCapacity, const void *src,
                     size_t srcSize, int compressionLevel) {
    int ret = compress2(dst, &dstCapacity, src, srcSize, compressionLevel);
    if (ret == Z_OK) {
        return dstCapacity;
    }
    // Error: Z_BUF_ERROR, Z_MEM_ERROR
    g_printerr("COMPRESSION-ERROR: ZLIB(%d) Error: %d | srcSize: %ld\n",
               compressionLevel, ret, srcSize);
    return 0;
}

size_t decompress_ZLIB(const char *src, char *dst, size_t compressedSize,
                       size_t dstCapacity) {
    int ret = uncompress(dst, &dstCapacity, src, compressedSize);
    if (ret == Z_OK) {
        return dstCapacity;
    }
    g_printerr("DECOMPRESSION-ERROR: ZLIB Error: %d | compressedSize: %ld\n",
               ret, compressedSize);
    return 0;
}

CompressionAlgorithm IOA_zlib = {
    bound_ZLIB,   compress_ZLIB,          decompress_ZLIB,
    level_values, COUNT_OF(level_values), "ZLIB",
    ZLIB};
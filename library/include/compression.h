#ifndef IOA_COMPRESSION_H
#define IOA_COMPRESSION_H
#include <glib.h>
#include <glib/gstdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <util.h>

typedef enum {
    LZ4,
    LZ4_FAST,
    ZSTD,
    ZLIB,
    _COMPRESSOR_COUNT
} CompressionAlgorithmID;

typedef struct {
    size_t (*bound)(size_t length);
    size_t (*compress)(void *dst, size_t dstCapacity, const void *src,
                       size_t srcSize, int compressionLevel);
    size_t (*decompress)(const char *src, char *dst, size_t compressedSize,
                         size_t dstCapacity);
    int *levels;
    int levels_count;
    const char *name;
    CompressionAlgorithmID compression_id;
} CompressionAlgorithm;

extern CompressionAlgorithm IOA_lz4;
extern CompressionAlgorithm IOA_lz4_fast;
extern CompressionAlgorithm IOA_zstd;
extern CompressionAlgorithm IOA_zlib;

typedef struct {
    CompressionAlgorithmID algorithm;
    int level;
} CompressionAlgorithm_Level;

extern GArray *available_compressors;
void init_compressors();
const char *compressor_to_name(CompressionAlgorithmID);
CompressionAlgorithmID name_to_compressor(char *name);
#endif
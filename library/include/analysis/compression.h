#ifndef IOA_ANALYSIS_COMPRESSION_H
#define IOA_ANALYSIS_COMPRESSION_H
#include <compression.h>
#include <glib.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <util.h>

typedef enum {
    METRIC_CR = 0,
    METRIC_CR_TIME,
    METRIC_COMPRESSION_SPEED,
    METRIC_DECOMPRESSION_SPEED,
    _METRIC_COUNT
} Metric_Type;

typedef struct {
    Metric_Type metric;
    CompressionAlgorithmID algorithmID;
    gint level;
    gfloat metric_value;
    long duration;
    size_t size;
    gchar *chunk_name;
} CompressionRun;

typedef struct {
    CompressionAlgorithm_Level compressor;
    Metric_Type metric;
    gfloat metric_value;
    size_t compressed_size;
} CompressionSample;

GList *test_algorithms(MPI_File fh, const void *buf, size_t buf_size,
                       MPI_Datatype datatype);

CompressionSample best_compressor(const void *buf, size_t buf_size,
                                  Metric_Type metric,
                                  CompressionAlgorithm_Level *skip);

CompressionSample evaluate(CompressionAlgorithm_Level compressor_info,
                           const void *buf, size_t buf_size);

gboolean store_training_chunk(char *name, const void *buf, size_t size,
                              MPI_Datatype datatype);

const char *metric_enum_name(Metric_Type type);

Metric_Type name_to_metric(char *name);
#endif
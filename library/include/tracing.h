#ifndef IOA_TRACING_H
#define IOA_TRACING_H
#include <analysis/compression.h>
#include <compression.h>
#include <glib.h>
#include <hdf5.h>
#include <meta.h>
#include <mpi.h>
#include <settings.h>
#include <stdio.h>
#include <stdlib.h>
#include <util.h>

extern GHashTable *trackingDB_fh;
extern GArray *trackingDB_io;
extern GArray *evaluation_ops;
extern gboolean stop_tracing;

typedef enum {
    OPERATION_TYPE_IO = 0,
    OPERATION_TYPE_META,
    OPERATION_TYPE_COMPRESSION,
    _OPERATION_TYPE_COUNT
} Operation_Type;

typedef struct {
    const char *filename;
    MPI_File *fh;
} IO_Object;

typedef struct {
    IO_Object *object;
    const char *operation_name;
    time_t time;
    long duration;
    Operation_Type type;
    int mpi_rank;
    union {
        struct {
            gchar datatype[MPI_MAX_DATAREP_STRING];
            gint count;
            gint size;
            MPI_Offset offset;
        } IO;
        struct {
            gchar datatype[MPI_MAX_DATAREP_STRING];
            gint count;
            gint size;
            MPI_Offset offset;
            CompressionAlgorithmID algorithm;
            gint level;
            Metric_Type metric;
            gfloat metric_value;
            gchar *chunk_name;
        } compression;
    };
} IO_Operation;

typedef struct {
    time_t time;
    int mpi_rank;
    size_t size;
    Metric_Type metric;
    CompressionAlgorithm_Level compressor_predicted;
    gfloat predicted_metric_value;
    size_t predicted_compressed_size;
    CompressionAlgorithm_Level compressor_tested;
    gfloat tested_metric_value;
    size_t tested_compressed_size;
} Evaluation_Operation;

void add_compression_run(void *handler, const char *type, CompressionRun run,
                         MPI_Datatype datatype, MPI_Offset offset, gint count,
                         size_t buf_size);
void add_compression_runs(void *handler, const char *type, GList *run,
                          MPI_Datatype datatype, MPI_Offset offset, gint count,
                          size_t buf_size);

void add_IO_operation(void *handler, const char *type, MPI_Datatype datatype,
                      MPI_Offset offset, gint count, size_t buf_size,
                      long duration);

void add_evaluation_operation(size_t buf_size, CompressionSample predicted,
                              CompressionSample tested);

void write_dataset();
gboolean tracing_stopped();

#endif
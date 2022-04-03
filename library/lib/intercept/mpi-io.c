#define _GNU_SOURCE
#include <dlfcn.h>
#include <filter.h>
#include <glib/gstdio.h>
#include <inferencing/compression.h>
#include <intercept/mpi-io.h>
int (*__real_PMPI_Init)(int *argc, char ***argv) = NULL;
int (*__real_PMPI_Finalize)(void) = NULL;

size_t count_to_size(int count, MPI_Datatype datatype) {
    // TODO: Long?
    int type_size;
    MPI_Type_size(datatype, &type_size);
    return count * type_size;
}

int MPI_Init(int *argc, char ***argv) {
    int ret;
    ret = PMPI_Init(argc, argv);
    return ret;
}

int PMPI_Init(int *argc, char ***argv) {
    int ret;
    __real_PMPI_Init = dlsym(RTLD_NEXT, "PMPI_Init");
    ret = __real_PMPI_Init(argc, argv);
    return ret;
}

int MPI_Finalize() {
    int ret;
    if (!tracing_stopped() &&
        (opt_test_compression || opt_tracing || opt_inferencing)) {
        stop_tracing = TRUE;
        write_dataset();
    }
    ret = PMPI_Finalize();
    return ret;
}

int PMPI_Finalize() {
    int ret;
    if (!tracing_stopped() &&
        (opt_test_compression || opt_tracing || opt_inferencing)) {
        stop_tracing = TRUE;
        write_dataset();
    }
    __real_PMPI_Finalize = dlsym(RTLD_NEXT, "PMPI_Finalize");
    ret = __real_PMPI_Finalize();
    return ret;
}

int MPI_Comm_size(MPI_Comm comm, int *size) {
    int ret;
    ret = PMPI_Comm_size(comm, size);
    /*
    TODO: Not usefull, as variable will be overwritten by
    any additional MPI_Comm
    */
    // MPI_SIZE = *size;
    return ret;
}

int MPI_Comm_rank(MPI_Comm comm, int *rank) {
    int ret;
    ret = PMPI_Comm_rank(comm, rank);
    // TODO: Same as above
    MPI_RANK = *rank;
    return ret;
}

int MPI_File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info,
                  MPI_File *fh) {
    int ret;
    ret = PMPI_File_open(comm, filename, amode, info, fh);
    if (tracing_stopped()) {
        return ret;
    }
    if (!g_hash_table_contains(trackingDB_fh, fh)) {
        IO_Object *object = g_new(IO_Object, 1);
        object->fh = (void *)*fh;
        object->filename = filename;
        g_debug("filename: %s | handler: %p", filename, object->fh);
        g_hash_table_insert(trackingDB_fh, object->fh, object);
    }
    return ret;
}

int MPI_File_write(MPI_File fh, const void *buf, int count,
                   MPI_Datatype datatype, MPI_Status *status) {

    if (tracing_stopped() || !_opt_action_required)
        return PMPI_File_write(fh, buf, count, datatype, status);

    size_t buffer_size = count_to_size(count, datatype);

    if (opt_inferencing && filter_IO(buffer_size)) {
        CompressionAlgorithm_Level prediction =
            predict_compressor(buf, buffer_size);
        CompressionSample evaluation = evaluate(prediction, buf, buffer_size);
        CompressionSample best = best_compressor(
            buf, buffer_size, opt_metric_inferencing, &evaluation.compressor);
        add_evaluation_operation(buffer_size, evaluation, best);

    } else {
        MPI_Offset offset;
        MPI_File_get_position(fh, &offset);
        if (opt_test_compression && filter_IO(buffer_size)) {
            GList *runs = test_algorithms(fh, buf, buffer_size, datatype);
            add_compression_runs(fh, __func__, runs, datatype, offset, count,
                                 buffer_size);
        }

        if (opt_tracing) {
            long s;
            long e;
            int ret;

            s = timeInMicroseconds();
            ret = PMPI_File_write(fh, buf, count, datatype, status);
            e = timeInMicroseconds() - s;
            add_IO_operation(fh, __func__, datatype, offset, count, buffer_size,
                             e);
            return ret;
        }
    }
    return PMPI_File_write(fh, buf, count, datatype, status);
}

int MPI_File_write_all(MPI_File fh, const void *buf, int count,
                       MPI_Datatype datatype, MPI_Status *status) {

    if (tracing_stopped() || !_opt_action_required)
        return PMPI_File_write_all(fh, buf, count, datatype, status);

    size_t buffer_size = count_to_size(count, datatype);

    if (opt_inferencing && filter_IO(buffer_size)) {
        CompressionAlgorithm_Level prediction =
            predict_compressor(buf, buffer_size);
        CompressionSample evaluation = evaluate(prediction, buf, buffer_size);
        CompressionSample best = best_compressor(
            buf, buffer_size, opt_metric_inferencing, &evaluation.compressor);
        add_evaluation_operation(buffer_size, evaluation, best);
    } else {
        MPI_Offset offset;
        MPI_File_get_position(fh, &offset);
        if (opt_test_compression && filter_IO(buffer_size)) {
            GList *runs = test_algorithms(fh, buf, buffer_size, datatype);
            add_compression_runs(fh, __func__, runs, datatype, offset, count,
                                 buffer_size);
        }

        if (opt_tracing) {
            long s;
            long e;
            int ret;

            s = timeInMicroseconds();
            ret = PMPI_File_write_all(fh, buf, count, datatype, status);
            e = timeInMicroseconds() - s;
            add_IO_operation(fh, __func__, datatype, offset, count, buffer_size,
                             e);
            return ret;
        }
    }
    return PMPI_File_write_all(fh, buf, count, datatype, status);
}

int MPI_File_write_at(MPI_File fh, MPI_Offset offset, const void *buf,
                      int count, MPI_Datatype datatype, MPI_Status *status) {

    if (tracing_stopped() || !_opt_action_required)
        return PMPI_File_write_at(fh, offset, buf, count, datatype, status);

    size_t buffer_size = count_to_size(count, datatype);

    if (opt_inferencing && filter_IO(buffer_size)) {
        CompressionAlgorithm_Level prediction =
            predict_compressor(buf, buffer_size);
        CompressionSample evaluation = evaluate(prediction, buf, buffer_size);
        CompressionSample best = best_compressor(
            buf, buffer_size, opt_metric_inferencing, &evaluation.compressor);
        add_evaluation_operation(buffer_size, evaluation, best);
    } else {
        if (opt_test_compression && filter_IO(buffer_size)) {
            GList *runs = test_algorithms(fh, buf, buffer_size, datatype);
            add_compression_runs(fh, __func__, runs, datatype, offset, count,
                                 buffer_size);
        }

        if (opt_tracing) {
            long s;
            long e;
            int ret;

            s = timeInMicroseconds();
            ret = PMPI_File_write_at(fh, offset, buf, count, datatype, status);
            e = timeInMicroseconds() - s;
            add_IO_operation(fh, __func__, datatype, offset, count, buffer_size,
                             e);
            return ret;
        }
    }
    return PMPI_File_write_at(fh, offset, buf, count, datatype, status);
}

int MPI_File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf,
                          int count, MPI_Datatype datatype,
                          MPI_Status *status) {

    if (tracing_stopped() || !_opt_action_required)
        return PMPI_File_write_at_all(fh, offset, buf, count, datatype, status);

    size_t buffer_size = count_to_size(count, datatype);

    if (opt_inferencing && filter_IO(buffer_size)) {
        CompressionAlgorithm_Level prediction =
            predict_compressor(buf, buffer_size);
        CompressionSample evaluation = evaluate(prediction, buf, buffer_size);
        CompressionSample best = best_compressor(
            buf, buffer_size, opt_metric_inferencing, &evaluation.compressor);
        add_evaluation_operation(buffer_size, evaluation, best);
    } else {
        if (opt_test_compression && filter_IO(buffer_size)) {
            GList *runs = test_algorithms(fh, buf, buffer_size, datatype);
            add_compression_runs(fh, __func__, runs, datatype, offset, count,
                                 buffer_size);
        }

        if (opt_tracing) {
            long s;
            long e;
            int ret;

            s = timeInMicroseconds();
            ret = PMPI_File_write_at_all(fh, offset, buf, count, datatype,
                                         status);
            e = timeInMicroseconds() - s;
            add_IO_operation(fh, __func__, datatype, offset, count, buffer_size,
                             e);
            return ret;
        }
    }
    return PMPI_File_write_at_all(fh, offset, buf, count, datatype, status);
}

int MPI_File_iwrite(MPI_File fh, const void *buf, int count,
                    MPI_Datatype datatype, MPI_Request *request) {

    if (tracing_stopped() || !_opt_action_required)
        return PMPI_File_iwrite(fh, buf, count, datatype, request);

    size_t buffer_size = count_to_size(count, datatype);

    if (opt_inferencing && filter_IO(buffer_size)) {
        CompressionAlgorithm_Level prediction =
            predict_compressor(buf, buffer_size);
        CompressionSample evaluation = evaluate(prediction, buf, buffer_size);
        CompressionSample best = best_compressor(
            buf, buffer_size, opt_metric_inferencing, &evaluation.compressor);
        add_evaluation_operation(buffer_size, evaluation, best);
    } else {
        MPI_Offset offset;
        MPI_File_get_position(fh, &offset);
        if (opt_test_compression && filter_IO(buffer_size)) {
            GList *runs = test_algorithms(fh, buf, buffer_size, datatype);
            add_compression_runs(fh, __func__, runs, datatype, offset, count,
                                 buffer_size);
        }

        if (opt_tracing) {
            long s;
            long e;
            int ret;

            s = timeInMicroseconds();
            ret = PMPI_File_iwrite(fh, buf, count, datatype, request);
            e = timeInMicroseconds() - s;
            add_IO_operation(fh, __func__, datatype, offset, count, buffer_size,
                             e);
            return ret;
        }
    }
    return PMPI_File_iwrite(fh, buf, count, datatype, request);
}

int MPI_File_iwrite_all(MPI_File fh, const void *buf, int count,
                        MPI_Datatype datatype, MPI_Request *request) {

    if (tracing_stopped() || !_opt_action_required)
        return PMPI_File_iwrite_all(fh, buf, count, datatype, request);

    size_t buffer_size = count_to_size(count, datatype);

    if (opt_inferencing && filter_IO(buffer_size)) {
        CompressionAlgorithm_Level prediction =
            predict_compressor(buf, buffer_size);
        CompressionSample evaluation = evaluate(prediction, buf, buffer_size);
        CompressionSample best = best_compressor(
            buf, buffer_size, opt_metric_inferencing, &evaluation.compressor);
        add_evaluation_operation(buffer_size, evaluation, best);
    } else {
        MPI_Offset offset;
        MPI_File_get_position(fh, &offset);
        if (opt_test_compression && filter_IO(buffer_size)) {
            GList *runs = test_algorithms(fh, buf, buffer_size, datatype);
            add_compression_runs(fh, __func__, runs, datatype, offset, count,
                                 buffer_size);
        }

        if (opt_tracing) {
            long s;
            long e;
            int ret;

            s = timeInMicroseconds();
            ret = PMPI_File_iwrite_all(fh, buf, count, datatype, request);
            e = timeInMicroseconds() - s;
            add_IO_operation(fh, __func__, datatype, offset, count, buffer_size,
                             e);
            return ret;
        }
    }
    return PMPI_File_iwrite_all(fh, buf, count, datatype, request);
}

int MPI_File_iwrite_at(MPI_File fh, MPI_Offset offset, const void *buf,
                       int count, MPI_Datatype datatype,
                       MPIO_Request *request) {
    if (tracing_stopped() || !_opt_action_required)
        return PMPI_File_iwrite_at(fh, offset, buf, count, datatype, request);

    size_t buffer_size = count_to_size(count, datatype);

    if (opt_inferencing && filter_IO(buffer_size)) {
        CompressionAlgorithm_Level prediction =
            predict_compressor(buf, buffer_size);
        CompressionSample evaluation = evaluate(prediction, buf, buffer_size);
        CompressionSample best = best_compressor(
            buf, buffer_size, opt_metric_inferencing, &evaluation.compressor);
        add_evaluation_operation(buffer_size, evaluation, best);
    } else {
        if (opt_test_compression && filter_IO(buffer_size)) {
            GList *runs = test_algorithms(fh, buf, buffer_size, datatype);
            add_compression_runs(fh, __func__, runs, datatype, offset, count,
                                 buffer_size);
        }

        if (opt_tracing) {
            long s;
            long e;
            int ret;

            s = timeInMicroseconds();
            ret =
                PMPI_File_iwrite_at(fh, offset, buf, count, datatype, request);
            e = timeInMicroseconds() - s;
            add_IO_operation(fh, __func__, datatype, offset, count, buffer_size,
                             e);
            return ret;
        }
    }
    return PMPI_File_iwrite_at(fh, offset, buf, count, datatype, request);
}

int MPI_File_iwrite_at_all(MPI_File fh, MPI_Offset offset, const void *buf,
                           int count, MPI_Datatype datatype,
                           MPIO_Request *request) {
    if (tracing_stopped() || !_opt_action_required)
        return PMPI_File_iwrite_at_all(fh, offset, buf, count, datatype,
                                       request);

    size_t buffer_size = count_to_size(count, datatype);

    if (opt_inferencing && filter_IO(buffer_size)) {
        CompressionAlgorithm_Level prediction =
            predict_compressor(buf, buffer_size);
        CompressionSample evaluation = evaluate(prediction, buf, buffer_size);
        CompressionSample best = best_compressor(
            buf, buffer_size, opt_metric_inferencing, &evaluation.compressor);
        add_evaluation_operation(buffer_size, evaluation, best);
    } else {
        if (opt_test_compression && filter_IO(buffer_size)) {
            GList *runs = test_algorithms(fh, buf, buffer_size, datatype);
            add_compression_runs(fh, __func__, runs, datatype, offset, count,
                                 buffer_size);
        }

        if (opt_tracing) {
            long s;
            long e;
            int ret;

            s = timeInMicroseconds();
            ret = PMPI_File_iwrite_at_all(fh, offset, buf, count, datatype,
                                          request);
            e = timeInMicroseconds() - s;
            add_IO_operation(fh, __func__, datatype, offset, count, buffer_size,
                             e);
            return ret;
        }
    }
    return PMPI_File_iwrite_at_all(fh, offset, buf, count, datatype, request);
}

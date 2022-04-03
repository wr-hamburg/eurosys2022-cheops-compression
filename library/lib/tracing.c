#include <mpi.h>
#include <tracing.h>

GHashTable *trackingDB_fh;
GArray *trackingDB_io;
GArray *evaluation_ops;
gboolean stop_tracing = FALSE;

gboolean tracing_stopped() { return stop_tracing; }

void add_compression_run(void *handler, const char *type, CompressionRun run,
                         MPI_Datatype datatype, MPI_Offset offset, gint count,
                         size_t buf_size) {
    char datatype_name[MPI_MAX_DATAREP_STRING];
    int len;

    IO_Operation operation;
    IO_Object *object = g_hash_table_lookup(trackingDB_fh, handler);
    operation.object = object;
    operation.operation_name = g_strdup(type);
    operation.time = time(NULL);
    operation.duration = run.duration;
    operation.type = OPERATION_TYPE_COMPRESSION;
    operation.mpi_rank = MPI_RANK;

    MPI_Type_get_name(datatype, datatype_name, &len);
    if (len <= 0) {
        strcpy(operation.compression.datatype, "NA");
    } else {
        strcpy(operation.compression.datatype, datatype_name);
    }
    operation.compression.algorithm = run.algorithmID;
    operation.compression.metric = run.metric;
    operation.compression.metric_value = run.metric_value;
    operation.compression.level = run.level;
    operation.compression.count = count;
    operation.compression.size = buf_size;
    operation.compression.offset = offset;
    operation.compression.chunk_name = run.chunk_name;
    g_array_append_val(trackingDB_io, operation);
}

void add_compression_runs(void *handler, const char *type, GList *runs,
                          MPI_Datatype datatype, MPI_Offset offset, gint count,
                          size_t buf_size) {
    GList *l;
    for (l = runs; l != NULL; l = l->next) {
        add_compression_run(handler, type, *(CompressionRun *)(l->data),
                            datatype, offset, count, buf_size);
    }
}

void add_IO_operation(void *handler, const char *type, MPI_Datatype datatype,
                      MPI_Offset offset, gint count, size_t buf_size,
                      long duration) {
    char datatype_name[MPI_MAX_DATAREP_STRING];
    int len;
    IO_Operation operation;
    IO_Object *object = g_hash_table_lookup(trackingDB_fh, handler);
    operation.object = object;
    operation.operation_name = g_strdup(type);
    operation.time = time(NULL);
    operation.duration = duration;
    operation.type = OPERATION_TYPE_IO;
    operation.mpi_rank = MPI_RANK;

    MPI_Type_get_name(datatype, datatype_name, &len);
    // TODO: Might be empty "empty string if not such name exists" ??
    // TODO: Filter?
    // g_debug("Datatype: %s, len: %d", datatype_name, len);
    if (len <= 0) {
        strcpy(operation.IO.datatype, "NA");
    } else {
        strcpy(operation.IO.datatype, datatype_name);
    }
    operation.IO.count = count;
    operation.IO.size = buf_size;
    operation.IO.offset = offset;
    g_array_append_val(trackingDB_io, operation);
}

void add_evaluation_operation(size_t buf_size, CompressionSample predicted,
                              CompressionSample tested) {
    Evaluation_Operation operation;
    operation.size = buf_size;
    operation.mpi_rank = MPI_RANK;
    operation.time = time(NULL);

    operation.metric = predicted.metric;
    operation.compressor_predicted = predicted.compressor;
    operation.predicted_metric_value = predicted.metric_value;
    operation.predicted_compressed_size = predicted.compressed_size;

    // Only store additional compressor if it performed better than the
    // predicted one
    if (tested.metric_value > predicted.metric_value) {
        operation.compressor_tested = tested.compressor;
        operation.tested_metric_value = tested.metric_value;
        operation.tested_compressed_size = tested.compressed_size;
    } else {
        // Something to test for
        operation.compressor_tested.algorithm = _COMPRESSOR_COUNT;
    }
    g_array_append_val(evaluation_ops, operation);
}

void write_dataset() {
    int ret;
    hid_t file, memtype_IO, memtype_compression, memtype_evaluation, space,
        dset_io, dset_compression, dset_evaluation, slabmemspace;
    herr_t status;
    hid_t plist_id;
    hsize_t dims_io[1];
    hsize_t dims_compression[1];
    hsize_t dims_evaluation[1];
    hid_t operation_type, datatype_type, compressor_type, metric_type,
        chunk_name_type;
    int count_io_ops = 0, count_compression_ops = 0, count_evaluation_ops = 0;
    IO_Operation *io;
    Evaluation_Operation *eo;

    typedef struct io_op_t {
        // Note: variable-length string datatype not possible in parallel
        gchar operation_name[100];
        time_t time;
        long duration;
        gchar datatype[MPI_MAX_DATAREP_STRING];
        long long mpi_offset;
        int mpi_rank;
        int count;
        int size;
    } io_op_t;

    typedef struct io_compression_t {
        // Note: variable-length string datatype not possible in parallel
        gchar operation_name[100];
        time_t time;
        long duration;
        gchar datatype[MPI_MAX_DATAREP_STRING];
        long long mpi_offset;
        int mpi_rank;
        int count;
        int size;
        gchar compressor[100];
        int level;
        gchar metric_name[100];
        gfloat metric_value;
        gchar chunk_name[100];
    } io_compression_t;

    typedef struct io_evaluation_t {
        time_t time;
        int mpi_rank;
        int size;
        gchar metric_name[100];
        gchar compressor_predicted[100];
        int compressor_predicted_level;
        gfloat predicted_metric_value;
        long compressed_size;
        gchar compressor_tested[100];
        int compressor_tested_level;
        gfloat tested_metric_value;
        long tested_size;
    } io_evaluation_t;

    // Count number of items per operation type and process
    for (int i = 0; i < trackingDB_io->len; ++i) {
        io = &g_array_index(trackingDB_io, IO_Operation, i);
        if (io->type == OPERATION_TYPE_IO)
            ++count_io_ops;
        if (io->type == OPERATION_TYPE_COMPRESSION)
            ++count_compression_ops;
    }
    count_evaluation_ops = evaluation_ops->len;

    PMPI_Barrier(MPI_COMM_WORLD);

    ret = PMPI_Comm_size(MPI_COMM_WORLD, &MPI_SIZE);
    if (ret != MPI_SUCCESS)
        PMPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    g_debug("MPI_SIZE: %d", MPI_SIZE);

    int *offsets_io = (int *)calloc(MPI_SIZE, sizeof(int));
    PMPI_Allgather(&count_io_ops, 1, MPI_INT, offsets_io, 1, MPI_INT,
                   MPI_COMM_WORLD);
    int *offsets_compression = (int *)calloc(MPI_SIZE, sizeof(int));
    PMPI_Allgather(&count_compression_ops, 1, MPI_INT, offsets_compression, 1,
                   MPI_INT, MPI_COMM_WORLD);
    int *offsets_evaluation = (int *)calloc(MPI_SIZE, sizeof(int));
    PMPI_Allgather(&count_evaluation_ops, 1, MPI_INT, offsets_evaluation, 1,
                   MPI_INT, MPI_COMM_WORLD);

    dims_io[0] = 0;
    for (int i = 0; i < MPI_SIZE; ++i) {
        dims_io[0] += offsets_io[i];
    }
    dims_compression[0] = 0;
    for (int i = 0; i < MPI_SIZE; ++i) {
        dims_compression[0] += offsets_compression[i];
    }
    dims_evaluation[0] = 0;
    for (int i = 0; i < MPI_SIZE; ++i) {
        dims_evaluation[0] += offsets_evaluation[i];
    }

    /*
     * Set up file access property list with parallel I/O access
     */
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    // TODO: Check MPI-IO availability
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL); // MPI_INFO_NULL
    /*
     * Create a new file using the default properties.
     */
    file = H5Fcreate(opt_meta_data_path, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    // BEGIN: TRACING
    /*
     * Create the compound datatype for memory.
     */
    operation_type = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(operation_type, 100);

    datatype_type = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(datatype_type, MPI_MAX_DATAREP_STRING);

    memtype_IO = H5Tcreate(H5T_COMPOUND, sizeof(io_op_t));

    status = H5Tinsert(memtype_IO, "Operation name",
                       HOFFSET(io_op_t, operation_name), operation_type);
    status = H5Tinsert(memtype_IO, "Timestamp", HOFFSET(io_op_t, time),
                       H5T_NATIVE_LONG);
    status = H5Tinsert(memtype_IO, "Duration [µs]", HOFFSET(io_op_t, duration),
                       H5T_NATIVE_LONG);
    status = H5Tinsert(memtype_IO, "MPI Datatype", HOFFSET(io_op_t, datatype),
                       datatype_type);
    status = H5Tinsert(memtype_IO, "MPI Offset", HOFFSET(io_op_t, mpi_offset),
                       H5T_NATIVE_LLONG);
    status = H5Tinsert(memtype_IO, "Variable Count", HOFFSET(io_op_t, count),
                       H5T_NATIVE_INT);
    status =
        H5Tinsert(memtype_IO, "Size", HOFFSET(io_op_t, size), H5T_NATIVE_INT);
    status = H5Tinsert(memtype_IO, "MPI Rank", HOFFSET(io_op_t, mpi_rank),
                       H5T_NATIVE_INT);

    /*
     * Create dataspace and datasets
     */
    space = H5Screate_simple(1, dims_io, NULL);
    dset_io = H5Dcreate(file, "IO-Trace", memtype_IO, space, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);

    // BEGIN: COMPRESSION
    memtype_compression = H5Tcreate(H5T_COMPOUND, sizeof(io_compression_t));

    compressor_type = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(compressor_type, 100);

    metric_type = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(metric_type, 100);

    chunk_name_type = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(chunk_name_type, 100);

    status =
        H5Tinsert(memtype_compression, "Operation name",
                  HOFFSET(io_compression_t, operation_name), operation_type);
    status = H5Tinsert(memtype_compression, "Timestamp",
                       HOFFSET(io_compression_t, time), H5T_NATIVE_LONG);
    status = H5Tinsert(memtype_compression, "Chunk Name",
                       HOFFSET(io_compression_t, chunk_name), chunk_name_type);
    status = H5Tinsert(memtype_compression, "Duration [µs]",
                       HOFFSET(io_compression_t, duration), H5T_NATIVE_LONG);
    status = H5Tinsert(memtype_compression, "MPI Datatype",
                       HOFFSET(io_compression_t, datatype), datatype_type);
    status = H5Tinsert(memtype_compression, "MPI Offset",
                       HOFFSET(io_compression_t, mpi_offset), H5T_NATIVE_LLONG);
    status = H5Tinsert(memtype_compression, "Variable Count",
                       HOFFSET(io_compression_t, count), H5T_NATIVE_INT);
    status = H5Tinsert(memtype_compression, "Size",
                       HOFFSET(io_compression_t, size), H5T_NATIVE_INT);
    status = H5Tinsert(memtype_compression, "MPI Rank",
                       HOFFSET(io_compression_t, mpi_rank), H5T_NATIVE_INT);
    status = H5Tinsert(memtype_compression, "Compressor name",
                       HOFFSET(io_compression_t, compressor), compressor_type);
    status = H5Tinsert(memtype_compression, "Compressor Level",
                       HOFFSET(io_compression_t, level), H5T_NATIVE_INT);
    status = H5Tinsert(memtype_compression, "Metric Name",
                       HOFFSET(io_compression_t, metric_name), metric_type);
    status =
        H5Tinsert(memtype_compression, "Metric Measurement",
                  HOFFSET(io_compression_t, metric_value), H5T_NATIVE_FLOAT);

    space = H5Screate_simple(1, dims_compression, NULL);
    dset_compression = H5Dcreate(file, "Compression-Trace", memtype_compression,
                                 space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // BEGIN: EVALUATION
    memtype_evaluation = H5Tcreate(H5T_COMPOUND, sizeof(io_evaluation_t));

    status = H5Tinsert(memtype_evaluation, "Timestamp",
                       HOFFSET(io_evaluation_t, time), H5T_NATIVE_LONG);
    status = H5Tinsert(memtype_evaluation, "MPI Rank",
                       HOFFSET(io_evaluation_t, mpi_rank), H5T_NATIVE_INT);
    status = H5Tinsert(memtype_evaluation, "Size",
                       HOFFSET(io_evaluation_t, size), H5T_NATIVE_INT);
    status = H5Tinsert(memtype_evaluation, "Metric Name",
                       HOFFSET(io_evaluation_t, metric_name), metric_type);
    status = H5Tinsert(memtype_evaluation, "Predicted Compressor",
                       HOFFSET(io_evaluation_t, compressor_predicted),
                       compressor_type);
    status = H5Tinsert(memtype_evaluation, "Predicted Level",
                       HOFFSET(io_evaluation_t, compressor_predicted_level),
                       H5T_NATIVE_INT);
    status = H5Tinsert(
        memtype_evaluation, "Predicted Compressor: Metric Measurement",
        HOFFSET(io_evaluation_t, predicted_metric_value), H5T_NATIVE_FLOAT);
    status =
        H5Tinsert(memtype_evaluation, "Predicted Compressor: Size",
                  HOFFSET(io_evaluation_t, compressed_size), H5T_NATIVE_LONG);
    status =
        H5Tinsert(memtype_evaluation, "Ideal Compressor",
                  HOFFSET(io_evaluation_t, compressor_tested), compressor_type);
    status = H5Tinsert(memtype_evaluation, "Ideal Level",
                       HOFFSET(io_evaluation_t, compressor_tested_level),
                       H5T_NATIVE_INT);
    status = H5Tinsert(
        memtype_evaluation, "Ideal Compressor: Metric Measurement",
        HOFFSET(io_evaluation_t, tested_metric_value), H5T_NATIVE_FLOAT);
    status = H5Tinsert(memtype_evaluation, "Ideal Compressor: Size",
                       HOFFSET(io_evaluation_t, tested_size), H5T_NATIVE_LONG);

    space = H5Screate_simple(1, dims_evaluation, NULL);
    dset_evaluation = H5Dcreate(file, "Evaluation", memtype_evaluation, space,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* END: Dataset Description */

    hsize_t count_compression[2] = {count_compression_ops, 1};
    hsize_t offset_compression[2] = {0, 0};
    for (int r = 0; r < MPI_RANK; ++r) {
        offset_compression[0] += offsets_compression[r];
    }
    free(offsets_compression);
    io_compression_t *data_compression =
        malloc(sizeof(io_compression_t) * count_compression_ops);
    int data_compression_index = 0;

    hsize_t count_io[2] = {count_io_ops, 1};
    hsize_t offset_io[2] = {0, 0};
    for (int r = 0; r < MPI_RANK; ++r) {
        offset_io[0] += offsets_io[r];
    }
    free(offsets_io);
    io_op_t *data_io = malloc(sizeof(io_op_t) * count_io_ops);
    int data_io_index = 0;

    hsize_t count_evaluation[2] = {count_evaluation_ops, 1};
    hsize_t offset_evaluation[2] = {0, 0};
    for (int r = 0; r < MPI_RANK; ++r) {
        offset_evaluation[0] += offsets_evaluation[r];
    }
    free(offsets_evaluation);
    io_evaluation_t *data_evaluation =
        malloc(sizeof(io_evaluation_t) * count_evaluation_ops);

    g_debug("Available Tracking Data: %d", trackingDB_io->len);
    g_debug("count_io_ops: %d", count_io_ops);
    g_debug("count_compression_ops: %d", count_compression_ops);
    g_debug("count_evaluation_ops: %d", count_evaluation_ops);
    // Available Tracking Data
    for (int i = 0; i < trackingDB_io->len; ++i) {
        io = &g_array_index(trackingDB_io, IO_Operation, i);
        if (io->type == OPERATION_TYPE_IO) {
            g_stpcpy(data_io[data_io_index].operation_name, io->operation_name);
            data_io[data_io_index].time = io->time;
            data_io[data_io_index].duration = io->duration;
            data_io[data_io_index].count = io->IO.count;
            data_io[data_io_index].size = io->IO.size;
            data_io[data_io_index].mpi_rank = io->mpi_rank;
            data_io[data_io_index].mpi_offset = io->IO.offset;

            strcpy(data_io[data_io_index].datatype, io->IO.datatype);
            ++data_io_index;
        } else if (io->type == OPERATION_TYPE_COMPRESSION) {
            g_stpcpy(data_compression[data_compression_index].operation_name,
                     io->operation_name);
            data_compression[data_compression_index].time = io->time;
            g_stpcpy(data_compression[data_compression_index].chunk_name,
                     io->compression.chunk_name);
            data_compression[data_compression_index].duration = io->duration;
            data_compression[data_compression_index].count =
                io->compression.count;
            data_compression[data_compression_index].size =
                io->compression.size;
            data_compression[data_compression_index].mpi_rank = io->mpi_rank;
            data_compression[data_compression_index].mpi_offset =
                io->compression.offset;

            g_stpcpy(data_compression[data_compression_index].compressor,
                     compressor_to_name(io->compression.algorithm));
            data_compression[data_compression_index].level =
                io->compression.level;
            strcpy(data_compression[data_compression_index].metric_name,
                   metric_enum_name(io->compression.metric));

            data_compression[data_compression_index].metric_value =
                io->compression.metric_value;

            strcpy(data_compression[data_compression_index].datatype,
                   io->compression.datatype);
            ++data_compression_index;
        }
    }

    for (int i = 0; i < evaluation_ops->len; ++i) {
        eo = &g_array_index(evaluation_ops, Evaluation_Operation, i);
        data_evaluation[i].time = eo->time;
        data_evaluation[i].mpi_rank = eo->mpi_rank;
        data_evaluation[i].size = eo->size;
        strcpy(data_evaluation[i].metric_name, metric_enum_name(eo->metric));

        g_stpcpy(data_evaluation[i].compressor_predicted,
                 compressor_to_name(eo->compressor_predicted.algorithm));
        data_evaluation[i].compressor_predicted_level =
            eo->compressor_predicted.level;
        data_evaluation[i].predicted_metric_value = eo->predicted_metric_value;
        data_evaluation[i].compressed_size = eo->predicted_compressed_size;

        if (eo->compressor_tested.algorithm != _COMPRESSOR_COUNT) {
            g_stpcpy(data_evaluation[i].compressor_tested,
                     compressor_to_name(eo->compressor_tested.algorithm));
            data_evaluation[i].compressor_tested_level =
                eo->compressor_tested.level;
            data_evaluation[i].tested_metric_value = eo->tested_metric_value;
            data_evaluation[i].tested_size = eo->tested_compressed_size;
        } else {
            g_stpcpy(data_evaluation[i].compressor_tested, "");
            data_evaluation[i].compressor_tested_level = 0;
            data_evaluation[i].tested_metric_value = 0;
            data_evaluation[i].tested_size = 0;
        }
    }

    /* Write: IO-Traces */
    space = H5Dget_space(dset_io);
    status = H5Sselect_hyperslab(space, H5S_SELECT_SET, offset_io, NULL,
                                 count_io, NULL);
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

    /* Create memory space for slab writes */

    slabmemspace = H5Screate_simple(1, count_io, NULL);
    status =
        H5Dwrite(dset_io, memtype_IO, slabmemspace, space, plist_id, data_io);

    status = H5Sclose(space);
    status = H5Pclose(plist_id);
    status = H5Dclose(dset_io);
    status = H5Sclose(slabmemspace);
    status = H5Tclose(memtype_IO);

    /* Write: Compression-Traces */
    space = H5Dget_space(dset_compression);
    status = H5Sselect_hyperslab(space, H5S_SELECT_SET, offset_compression,
                                 NULL, count_compression, NULL);
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

    /* Create memory space for slab writes */
    slabmemspace = H5Screate_simple(1, count_compression, NULL);
    status = H5Dwrite(dset_compression, memtype_compression, slabmemspace,
                      space, plist_id, data_compression);

    status = H5Sclose(space);
    status = H5Pclose(plist_id);
    status = H5Dclose(dset_compression);
    status = H5Sclose(slabmemspace);
    status = H5Tclose(memtype_compression);

    /* Write: Evaluation-Traces */
    space = H5Dget_space(dset_evaluation);
    status = H5Sselect_hyperslab(space, H5S_SELECT_SET, offset_evaluation, NULL,
                                 count_evaluation, NULL);
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

    /* Create memory space for slab writes */
    slabmemspace = H5Screate_simple(1, count_evaluation, NULL);
    status = H5Dwrite(dset_evaluation, memtype_evaluation, slabmemspace, space,
                      plist_id, data_evaluation);

    status = H5Sclose(space);
    status = H5Pclose(plist_id);
    status = H5Dclose(dset_evaluation);
    status = H5Sclose(slabmemspace);
    status = H5Tclose(memtype_evaluation);

    free(data_compression);
    free(data_io);
    free(data_evaluation);
    PMPI_Barrier(MPI_COMM_WORLD);
    status = H5Fclose(file);
    if (status < 0) {
        g_debug("HDF5 Error...");
    }
}

#include <analysis/compression.h>
#include <settings.h>

const char *const metric_type_name[] = {
    [METRIC_CR] = "Compression Rate",
    [METRIC_CR_TIME] = "Compression Rate per Time",
    [METRIC_COMPRESSION_SPEED] = "Compression Speed",
    [METRIC_DECOMPRESSION_SPEED] = "Decompression Speed",
};

GList *test_algorithms(MPI_File fh, const void *buf, size_t buf_size,
                       MPI_Datatype datatype) {

    GList *compressor_list = NULL;
    char *chunk_name = g_strdup_printf("%s.data", g_uuid_string_random());

    CompressionAlgorithm *compressor;
    for (int i = 0; i < available_compressors->len; ++i) {
        size_t max_bound, compressed_size;
        char *compressed_data;
        compressor =
            &g_array_index(available_compressors, CompressionAlgorithm, i);

        int *compression_levels = compressor->levels;
        for (int l = 0; l < compressor->levels_count; ++l) {
            gint level = compression_levels[l];

            max_bound = compressor->bound(buf_size);
            compressed_data = g_malloc(max_bound);
            long time_average = 0;
            for (int t = 0; t < opt_repeat_measurements; ++t) {
                long s = timeInMicroseconds();
                compressed_size = compressor->compress(
                    compressed_data, max_bound, buf, buf_size, level);
                // Compressor Error Handling
                if (compressed_size == 0) {
                    // g_free(compressed_data);
                    continue;
                }
                time_average += (timeInMicroseconds() - s);
            }
            time_average = time_average / opt_repeat_measurements;

            gfloat cr = (gfloat)buf_size / (gfloat)compressed_size;
            gfloat cr_time = cr / (time_average / 1000000.0);
            // Throughput per Second
            gfloat compression_throughput =
                buf_size / (time_average / 1000000.0);

            for (int m = 0; m < _METRIC_COUNT; ++m) {
                if (m == METRIC_DECOMPRESSION_SPEED && opt_decompression) {
                    long time_decomp_average = 0;
                    for (int t = 0; t < opt_repeat_measurements; ++t) {
                        long s_decomp = timeInMicroseconds();
                        // TODO: Too risky? Uses original buffer to decompress
                        size_t decompressed_size = compressor->decompress(
                            compressed_data, buf, compressed_size, buf_size);
                        time_decomp_average +=
                            (timeInMicroseconds() - s_decomp);
                    }
                    time_decomp_average =
                        time_decomp_average / opt_repeat_measurements;
                    gfloat decompression_speed =
                        buf_size / (time_decomp_average / 1000000.0);

                    CompressionRun *run = g_malloc(sizeof(CompressionRun));
                    run->algorithmID = compressor->compression_id;
                    run->level = level;
                    run->duration = time_decomp_average;
                    run->size = buf_size;
                    run->metric = m;
                    run->metric_value = decompression_speed;
                    run->chunk_name = chunk_name;
                    compressor_list = g_list_prepend(compressor_list, run);
                } else {
                    CompressionRun *run = g_malloc(sizeof(CompressionRun));
                    run->algorithmID = compressor->compression_id;
                    run->level = level;
                    run->duration = time_average;
                    run->size = buf_size;
                    run->metric = m;
                    run->chunk_name = chunk_name;

                    if (m == METRIC_CR)
                        run->metric_value = cr;
                    else if (m == METRIC_CR_TIME)
                        run->metric_value = cr_time;
                    else if (m == METRIC_COMPRESSION_SPEED)
                        run->metric_value = compression_throughput;
                    compressor_list = g_list_prepend(compressor_list, run);
                }
            }
            g_free(compressed_data);
        }
    }

    if (opt_store_chunks)
        store_training_chunk(chunk_name, buf, buf_size, datatype);

    return compressor_list;
}

CompressionSample best_compressor(const void *buf, size_t buf_size,
                                  Metric_Type metric,
                                  CompressionAlgorithm_Level *skip) {

    CompressionSample best;
    best.metric = metric;

    CompressionAlgorithm *compressor;
    for (int i = 0; i < available_compressors->len; ++i) {
        size_t max_bound, compressed_size;
        char *compressed_data;
        compressor =
            &g_array_index(available_compressors, CompressionAlgorithm, i);

        int *compression_levels = compressor->levels;
        for (int l = 0; l < compressor->levels_count; ++l) {
            gint level = compression_levels[l];
            // Might want to skip a compressor that has been tested before
            if (skip && skip->algorithm == compressor->compression_id &&
                skip->level == level)
                continue;

            max_bound = compressor->bound(buf_size);
            compressed_data = g_malloc(max_bound);
            long s = timeInMicroseconds();
            compressed_size = compressor->compress(compressed_data, max_bound,
                                                   buf, buf_size, level);
            // Compressor Error Handling
            if (compressed_size == 0) {
                g_free(compressed_data);
                continue;
            }
            long e = timeInMicroseconds() - s;
            gfloat cr = (gfloat)buf_size / (gfloat)compressed_size;
            gfloat cr_time = cr / (e / 1000000.0);
            // Throughput per Second
            gfloat compression_throughput = buf_size / (e / 1000000.0);

            gboolean winner = FALSE;
            if (metric == METRIC_DECOMPRESSION_SPEED) {
                long s_decomp = timeInMicroseconds();
                size_t decompressed_size = compressor->decompress(
                    compressed_data, buf, compressed_size, buf_size);
                long e_decomp = timeInMicroseconds() - s_decomp;
                gfloat decompression_speed = buf_size / (e_decomp / 1000000.0);

                if (decompression_speed > best.metric_value) {
                    winner = TRUE;
                    best.metric_value = decompression_speed;
                }
            } else if (metric == METRIC_CR && cr > best.metric_value) {
                winner = TRUE;
                best.metric_value = cr;
            } else if (metric == METRIC_CR_TIME &&
                       cr_time > best.metric_value) {
                winner = TRUE;
                best.metric_value = cr_time;
            } else if (metric == METRIC_COMPRESSION_SPEED &&
                       compression_throughput > best.metric_value) {
                winner = TRUE;
                best.metric_value = compression_throughput;
            }

            if (winner) {
                best.compressor.algorithm = compressor->compression_id;
                best.compressor.level = level;
                best.compressed_size = compressed_size;
            }
            g_free(compressed_data);
        }
    }
    return best;
}

CompressionSample evaluate(CompressionAlgorithm_Level compressor_info,
                           const void *buf, size_t buf_size) {
    size_t max_bound, compressed_size;
    char *compressed_data;
    CompressionSample run;
    CompressionAlgorithm *compressor = &g_array_index(
        available_compressors, CompressionAlgorithm, compressor_info.algorithm);

    max_bound = compressor->bound(buf_size);
    compressed_data = g_malloc(max_bound);
    // Compress
    long s = timeInMicroseconds();
    compressed_size = compressor->compress(compressed_data, max_bound, buf,
                                           buf_size, compressor_info.level);
    long e = timeInMicroseconds() - s;

    gfloat cr = (gfloat)buf_size / (gfloat)compressed_size;
    gfloat cr_time = cr / (e / 1000000.0);
    gfloat compression_throughput = buf_size / (e / 1000000.0);
    g_debug(
        "Predicted Compressor: %s(%d) - CR: %.6f | Input: %ld - Output: %ld",
        compressor->name, compressor_info.level, cr, buf_size, compressed_size);

    if (opt_metric_inferencing == METRIC_DECOMPRESSION_SPEED) {
        long s_decomp = timeInMicroseconds();
        size_t decompressed_size = compressor->decompress(
            compressed_data, buf, compressed_size, buf_size);
        long e_decomp = timeInMicroseconds() - s_decomp;
        gfloat decompression_speed = buf_size / (e_decomp / 1000000.0);

        run.metric_value = decompression_speed;
    } else if (opt_metric_inferencing == METRIC_CR) {
        run.metric_value = cr;
    } else if (opt_metric_inferencing == METRIC_CR_TIME) {
        run.metric_value = cr_time;
    } else if (opt_metric_inferencing == METRIC_COMPRESSION_SPEED) {
        run.metric_value = compression_throughput;
    }

    g_free(compressed_data);
    run.metric = opt_metric_inferencing;
    run.compressor = compressor_info;
    run.compressed_size = compressed_size;
    return run;
}

gboolean store_training_chunk(char *name, const void *buf, size_t size,
                              MPI_Datatype datatype) {
    gboolean ret = TRUE;

    char *path = g_build_filename(opt_chunk_path, name, NULL);
    g_file_set_contents(path, buf, size, NULL);

    g_free(path);
    return ret;
}

const char *metric_enum_name(Metric_Type type) {
    return metric_type_name[type];
}

Metric_Type name_to_metric(char *name) {
    for (int i = 0; i < _METRIC_COUNT; i++) {
        if (strcmp(name, metric_type_name[i]) == 0) {
            return i;
        }
    }
    g_printerr("Metric not found: %s\n", name);
    exit(EXIT_FAILURE);
}

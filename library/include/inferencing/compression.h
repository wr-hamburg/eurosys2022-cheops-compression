#ifndef IOA_INFERENCING_COMPRESSION_H
#define IOA_INFERENCING_COMPRESSION_H
#include <analysis/compression.h>
#include <compression.h>
#include <inferencing/onnxruntime_c_api.h>
#include <stddef.h>

CompressionAlgorithm_Level *parse_labels(char *path);
Metric_Type parse_metric(char *path);
int model_input_size(char *path);
void init_ml(char *model_path, char *settings_path);
void cleanup_ml();
CompressionAlgorithm_Level predict_compressor(const void *data, size_t length);

#endif
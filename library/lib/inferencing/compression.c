#include <assert.h>
#include <inferencing/compression.h>
#include <math.h>
#include <settings.h>
#include <util.h>

const OrtApi *onnx_api = NULL;
OrtSession *onnx_session = NULL;
OrtEnv *onnx_env = NULL;
OrtSessionOptions *onnx_session_options = NULL;

const CompressionAlgorithm_Level *labels;
int input_size;
int total_elements;
const size_t ELEMENT_SIZE = sizeof(float);
size_t total_size;

#define ORT_ABORT_ON_ERROR(expr)                                               \
    do {                                                                       \
        OrtStatus *onnx_status = (expr);                                       \
        if (onnx_status != NULL) {                                             \
            const char *msg = onnx_api->GetErrorMessage(onnx_status);          \
            g_printerr("%s\n", msg);                                           \
            onnx_api->ReleaseStatus(onnx_status);                              \
            abort();                                                           \
        }                                                                      \
    } while (0);

CompressionAlgorithm_Level *parse_labels(char *path) {
    gsize length;
    char *content;
    if (!g_file_get_contents(path, &content, &length, NULL)) {
        g_printerr("Can't open settings file: %s\n", path);
        exit(1);
    }

    gchar **settings_raw = g_strsplit(content, "\n", 0);
    gchar **ptr;

    // Count valid label rows (ignore empty rows, skip first two entries)
    gint valid_labels = 0;
    gint row_id = 0;
    for (ptr = settings_raw; *ptr; ptr++) {
        if (!g_strcmp0(*ptr, "") == 0 && row_id > 1)
            ++valid_labels;
        ++row_id;
    }

    CompressionAlgorithm_Level *labels =
        malloc(valid_labels * sizeof(CompressionAlgorithm_Level));

    row_id = 0;
    gint label_id = -1;
    for (ptr = settings_raw; *ptr; ptr++) {
        if (!g_strcmp0(*ptr, "") == 0 && row_id > 1) {
            ++label_id;
            gchar **label_level = g_strsplit(*ptr, ":", 0);
            // g_debug("Compressor: %s - Level: %d", label_level[0],
            //        atoi(label_level[1]));
            labels[label_id].algorithm = name_to_compressor(label_level[0]);
            labels[label_id].level = atoi(label_level[1]);
            // g_print("name_to_compressor(%s): %d\n", label_level[0],
            //         name_to_compressor(label_level[0]));
            g_strfreev(label_level);
        }
        ++row_id;
    }
    g_strfreev(settings_raw);
    return labels;
}

Metric_Type parse_metric(char *path) {
    gsize length;
    char *content;
    if (!g_file_get_contents(path, &content, &length, NULL)) {
        g_printerr("Can't open settings file: %s\n", path);
        exit(1);
    }

    gchar **settings_raw = g_strsplit(content, "\n", 2);
    Metric_Type metric = name_to_metric(settings_raw[0]);
    g_strfreev(settings_raw);
    return metric;
}

int model_input_size(char *path) {
    gsize length;
    char *content;
    if (!g_file_get_contents(path, &content, &length, NULL)) {
        g_printerr("Can't open settings file: %s\n", path);
        exit(1);
    }

    gchar **settings_raw = g_strsplit(content, "\n", 3);
    int input_size = atoi(settings_raw[1]);
    g_strfreev(settings_raw);
    return input_size;
}

void init_ml(char *model_path, char *settings_path) {
    onnx_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!onnx_api) {
        g_printerr("Failed to init ONNX Runtime engine.\n");
        abort();
    } else {
        g_debug("Initialized ONNX Runtime version %s\n",
                OrtGetApiBase()->GetVersionString());
    }

    ORT_ABORT_ON_ERROR(
        onnx_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &onnx_env));
    ORT_ABORT_ON_ERROR(onnx_api->CreateSessionOptions(&onnx_session_options));
    ORT_ABORT_ON_ERROR(onnx_api->CreateSession(
        onnx_env, model_path, onnx_session_options, &onnx_session));

    labels = parse_labels(settings_path);
    opt_metric_inferencing = parse_metric(settings_path);
    input_size = model_input_size(settings_path);

    total_elements = 1 * input_size;
    total_size = total_elements * ELEMENT_SIZE;
}

void cleanup_ml() {
    onnx_api->ReleaseSessionOptions(onnx_session_options);
    onnx_api->ReleaseSession(onnx_session);
    onnx_api->ReleaseEnv(onnx_env);
    free((void *)labels);
}

CompressionAlgorithm_Level predict_compressor(const void *data, size_t length) {
    size_t model_input_ele_count;
    // g_debug("Length: %ld", length);
    if (length > total_size)
        length = total_size;

    model_input_ele_count = length / ELEMENT_SIZE;
    // g_debug("model_input_ele_count: %ld\n", model_input_ele_count);

    float *parsed = g_malloc(length);
    memcpy(parsed, data, length);
#pragma omp parallel for
    for (int i = 0; i < model_input_ele_count; i++) {
        if (isnan(parsed[i]))
            parsed[i] = 0.0;
    }

    OrtMemoryInfo *memory_info;
    ORT_ABORT_ON_ERROR(onnx_api->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    int64_t input_shape[] = {1, 1, model_input_ele_count};
    size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);

    OrtValue *input_tensor = NULL;
    ORT_ABORT_ON_ERROR(onnx_api->CreateTensorWithDataAsOrtValue(
        memory_info, parsed, length, input_shape, input_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    assert(input_tensor != NULL);
    int is_tensor;
    ORT_ABORT_ON_ERROR(onnx_api->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);

    onnx_api->ReleaseMemoryInfo(memory_info);
    const char *input_names[] = {"input_1"};
    const char *output_names[] = {"output_1"};
    OrtValue *output_tensor = NULL;
    ORT_ABORT_ON_ERROR(onnx_api->Run(onnx_session, NULL, input_names,
                                     (const OrtValue *const *)&input_tensor, 1,
                                     output_names, 1, &output_tensor));
    assert(output_tensor != NULL);
    ORT_ABORT_ON_ERROR(onnx_api->IsTensor(output_tensor, &is_tensor));
    assert(is_tensor);

    // Get result
    struct OrtTensorTypeAndShapeInfo *shape_info;
    ORT_ABORT_ON_ERROR(
        onnx_api->GetTensorTypeAndShape(output_tensor, &shape_info));
    size_t class_dim_count;
    ORT_ABORT_ON_ERROR(
        onnx_api->GetTensorShapeElementCount(shape_info, &class_dim_count));

    // ['GZIP', 'LZ4', 'ZSTD']
    float *f;
    ORT_ABORT_ON_ERROR(
        onnx_api->GetTensorMutableData(output_tensor, (void **)&f));

    float class_probabilities[class_dim_count];
    softmax(f, class_dim_count, class_probabilities);
    /*
    g_debug("Softmax Probabilities");
    for (int i = 0; i < class_dim_count; ++i) {
        g_debug("[%d]: %.6f (%.6f)\n", i, class_probabilities[i], f[i]);
    }
    */
    int winning_index = max_value_index(class_probabilities, class_dim_count);
    // g_debug("Winner Index: %d", winning_index);

    onnx_api->ReleaseValue(output_tensor);
    onnx_api->ReleaseValue(input_tensor);
    g_free(parsed);

    return labels[winning_index];
}

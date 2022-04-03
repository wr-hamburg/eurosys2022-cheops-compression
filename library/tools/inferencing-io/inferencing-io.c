#include <assert.h>
#include <glib.h>
#include <include/inferencing/compression.h>
#include <inferencing-io.h>
#include <math.h>
#include <stdio.h>
#include <util.h>
/*
This tool is currently following the example shown in [1].

[1]
https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/fns_candy_style_transfer/fns_candy_style_transfer.c

Model Visualization: https://netron.app/

./builddir/inferencing-demo
/home/julius/Documents/Studium/MA/ML/CompressionML/models/icon.onnx
/home/julius/Documents/Studium/MA/ML/CompressionML/training_data_transformed/icon/ZSTD/fff6dd90-272a-4fab-a18f-0089021f8508.data.jpg
*/
#define W 4096
#define H 1

const int WIDTH = W;
const int HEIGHT = H;
const int TOTAL_ELEMENTS = W * H;
const size_t ELEMENT_SIZE = sizeof(float);
const size_t TOTAL_SIZE = TOTAL_ELEMENTS * ELEMENT_SIZE;

const OrtApi *g_ort = NULL;
const CompressionAlgorithm_Level *labels;
Metric_Type metric_inferencing;
int input_size;

#define ORT_ABORT_ON_ERROR(expr)                                               \
    do {                                                                       \
        OrtStatus *onnx_status = (expr);                                       \
        if (onnx_status != NULL) {                                             \
            const char *msg = g_ort->GetErrorMessage(onnx_status);             \
            fprintf(stderr, "%s\n", msg);                                      \
            g_ort->ReleaseStatus(onnx_status);                                 \
            abort();                                                           \
        }                                                                      \
    } while (0);

static void show_class(OrtValue *tensor) {
    struct OrtTensorTypeAndShapeInfo *shape_info;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(tensor, &shape_info));
    size_t class_dim_count;
    ORT_ABORT_ON_ERROR(
        g_ort->GetTensorShapeElementCount(shape_info, &class_dim_count));
    g_print("class_dim_count: %ld\n", class_dim_count);

    int64_t layer_dims;
    ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(shape_info, &layer_dims));
    g_print("layer_dims: %ld\n", layer_dims);

    ONNXTensorElementDataType onnxTypeEnum;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorElementType(shape_info, &onnxTypeEnum));
    g_print("GetTensorElementType: %d\n", onnxTypeEnum);

    int64_t dims[layer_dims];
    ORT_ABORT_ON_ERROR(g_ort->GetDimensions(shape_info, dims, layer_dims));

    for (int i = 0; i < layer_dims; ++i) {
        g_print("dims[%d]: %ld\n", i, dims[i]);
    }

    float *f;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(tensor, (void **)&f));

    float class_probabilities[class_dim_count];
    softmax(f, class_dim_count, class_probabilities);
    g_print("Softmax Probabilities\n");
    for (int i = 0; i < class_dim_count; ++i) {
        g_print("[%d]: %.6f (%.6f)\n", i, class_probabilities[i], f[i]);
    }
    int winning_index = max_value_index(class_probabilities, class_dim_count);
    CompressionAlgorithm *compressor =
        &g_array_index(available_compressors, CompressionAlgorithm,
                       labels[winning_index].algorithm);
    g_print("Winner Index: %d -> Compressor: %s | Level: %d\n", winning_index,
            compressor->name, labels[winning_index].level);
}

// Source: https://stackoverflow.com/a/3974138
void printBits(size_t const size, void const *const ptr) {
    unsigned char *b = (unsigned char *)ptr;
    unsigned char byte;
    int i, j;

    for (i = size - 1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    puts("");
}

int run_inference(OrtSession *session, const ORTCHAR_T *input_file) {
    size_t model_input_ele_count;

    gsize length;
    float *content;
    if (!g_file_get_contents(input_file, &content, &length, NULL)) {
        g_printerr("Can't open file\n");
        return -1;
    }
    g_print("Length: %ld\n", length);
    if (length > TOTAL_SIZE)
        length = TOTAL_SIZE;

    model_input_ele_count = length / ELEMENT_SIZE; //  *sizeof(float);
    g_print("model_input_ele_count: %ld\n", model_input_ele_count);

    // printBits(sizeof(float), &content[5]);

    // TODO: Better way of handling NaN values (e.g. Layer within NN)
#pragma omp parallel for
    for (int i = 0; i < model_input_ele_count; i++) {
        if (isnan(content[i]))
            content[i] = 0.0;
        // g_print("%.6f\n", content[i]);
        // g_print("isnan: %d\n", isnan(content[i]));
    }

    OrtMemoryInfo *memory_info;
    ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    const int64_t input_shape[] = {1, HEIGHT, model_input_ele_count};
    const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
    g_print("input_shape_len %ld\n", input_shape_len);

    // Input length: model_input_ele_count * sizeof(float)

    OrtValue *input_tensor = NULL;
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, content, length, input_shape, input_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    assert(input_tensor != NULL);
    int is_tensor;
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    const char *input_names[] = {"input_1"};
    const char *output_names[] = {"output_1"};
    OrtValue *output_tensor = NULL;
    ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names,
                                  (const OrtValue *const *)&input_tensor, 1,
                                  output_names, 1, &output_tensor));
    assert(output_tensor != NULL);
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
    assert(is_tensor);
    int ret = 0;
    show_class(output_tensor);
    /*
    if (write_tensor_to_png_file(output_tensor, output_file_p) != 0) {
        ret = -1;
    }
    */
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    // free(model_input);
    g_print("done\n");

    g_free(content);
    return ret;
}

void verify_input_output_count(OrtSession *session) {
    size_t count;
    ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
    assert(count == 1);
    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
    assert(count == 1);
}

int main(int argc, char **argv) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        g_printerr("Failed to init ONNX Runtime engine.\n");
        return -1;
    } else {
        g_print("Initialized ONNX Runtime version %s\n",
                OrtGetApiBase()->GetVersionString());
    }

    ORTCHAR_T *model_path = argv[1];
    ORTCHAR_T *input_file = argv[2];
    char *settings_file = argv[3];

    labels = parse_labels(settings_file);
    metric_inferencing = parse_metric(settings_file);
    input_size = model_input_size(settings_file);
    g_print("Metric: %d, Name: %s, Tensor Size: %d\n", metric_inferencing,
            metric_enum_name(metric_inferencing), input_size);

    OrtEnv *env;
    ORT_ABORT_ON_ERROR(
        g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
    assert(env != NULL);
    int ret = 0;
    OrtSessionOptions *session_options;
    ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

    OrtSession *session;
    ORT_ABORT_ON_ERROR(
        g_ort->CreateSession(env, model_path, session_options, &session));
    verify_input_output_count(session);

    if (g_file_test(input_file, G_FILE_TEST_IS_REGULAR))
        ret = run_inference(session, input_file);
    else if (g_file_test(input_file, G_FILE_TEST_IS_DIR)) {
        GDir *dir;
        GError *error;
        const gchar *filename;

        dir = g_dir_open(input_file, 0, &error);
        while ((filename = g_dir_read_name(dir))) {
            filename = g_strconcat(input_file, filename, NULL);
            g_print("%s\n", filename);
            ret = run_inference(session, filename);
        }
    }
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseEnv(env);

    return 0;
}

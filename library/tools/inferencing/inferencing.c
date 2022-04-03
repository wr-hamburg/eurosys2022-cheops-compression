#include <assert.h>
#include <glib.h>
#include <inferencing.h>
#include <png.h>
#include <stdio.h>

/*
This tool is currently following the example shown in [1].

[1]
https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/fns_candy_style_transfer/fns_candy_style_transfer.c

Model Visualization: https://netron.app/

./builddir/inferencing-demo
/home/julius/Documents/Studium/MA/ML/CompressionML/models/icon.onnx
/home/julius/Documents/Studium/MA/ML/CompressionML/training_data_transformed/icon/ZSTD/fff6dd90-272a-4fab-a18f-0089021f8508.data.jpg
*/
#define W 64
#define H 64
#define C 3

const int WIDTH = W;
const int HEIGHT = H;
const int CHANNELS = C;
const int TOTAL_ELEMENTS = W * H * C;

const OrtApi *g_ort = NULL;

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

/**
 * convert input from HWC format to CHW format
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
static void hwc_to_chw(const png_byte *input, size_t h, size_t w,
                       float **output, size_t *output_count) {
    size_t stride = h * w;
    *output_count = stride * 3;
    float *output_data = (float *)malloc(*output_count * sizeof(float));
    for (size_t i = 0; i != stride; ++i) {
        for (size_t c = 0; c != 3; ++c) {
            output_data[c * stride + i] = input[i * 3 + c];
        }
    }
    *output = output_data;
}

static int show_class(OrtValue *tensor) {
    struct OrtTensorTypeAndShapeInfo *shape_info;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(tensor, &shape_info));
    size_t class_dim_count;
    ORT_ABORT_ON_ERROR(
        g_ort->GetTensorShapeElementCount(shape_info, &class_dim_count));
    g_print("class_dim_count: %d\n", class_dim_count);

    int64_t layer_dims;
    ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(shape_info, &layer_dims));
    g_print("layer_dims: %d\n", layer_dims);

    ONNXTensorElementDataType onnxTypeEnum;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorElementType(shape_info, &onnxTypeEnum));
    g_print("GetTensorElementType: %d\n", onnxTypeEnum);

    int64_t dims[layer_dims];
    ORT_ABORT_ON_ERROR(g_ort->GetDimensions(shape_info, dims, layer_dims));

    for (int i = 0; i < layer_dims; ++i) {
        g_print("dims[%d]: %d\n", i, dims[i]);
    }

    float *f;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(tensor, (void **)&f));
    for (int i = 0; i < class_dim_count; ++i) {
        g_print("[%d]: %.6f\n", i, f[i]);
    }
}
/**
 * \param out should be freed by caller after use
 * \param output_count Array length of the `out` param
 */
static int read_png_file(const char *input_file, size_t *height, size_t *width,
                         float **out, size_t *output_count) {
    png_image image; /* The control structure used by libpng */
    /* Initialize the 'png_image' structure. */
    memset(&image, 0, (sizeof image));
    image.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_file(&image, input_file) == 0) {
        return -1;
    }
    png_bytep buffer;
    image.format = PNG_FORMAT_BGR;
    size_t input_data_length = PNG_IMAGE_SIZE(image);
    if (input_data_length != TOTAL_ELEMENTS) {
        printf("input_data_length:%zd\n", input_data_length);
        return -1;
    }
    buffer = (png_bytep)malloc(input_data_length);
    memset(buffer, 0, input_data_length);
    if (png_image_finish_read(&image, NULL /*background*/, buffer,
                              0 /*row_stride*/, NULL /*colormap*/) == 0) {
        return -1;
    }
    hwc_to_chw(buffer, image.height, image.width, out, output_count);
    free(buffer);
    *width = image.width;
    *height = image.height;
    return 0;
}

int run_inference(OrtSession *session, const ORTCHAR_T *input_file) {
    size_t input_height;
    size_t input_width;
    float *model_input;
    size_t model_input_ele_count;

    const char *input_file_p = input_file;

    if (read_png_file(input_file_p, &input_height, &input_width, &model_input,
                      &model_input_ele_count) != 0) {
        return -1;
    }
    if (input_height != HEIGHT || input_width != WIDTH) {
        g_printerr("please resize to image to %dx%d\n", HEIGHT, WIDTH);
        free(model_input);
        return -1;
    }
    OrtMemoryInfo *memory_info;
    ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    const int64_t input_shape[] = {1, WIDTH, HEIGHT, 3};
    const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
    const size_t model_input_len = model_input_ele_count * sizeof(float);

    OrtValue *input_tensor = NULL;
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, model_input, model_input_len, input_shape, input_shape_len,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
    assert(input_tensor != NULL);
    int is_tensor;
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    const char *input_names[] = {"rescaling_1_input"};
    const char *output_names[] = {"dense_1"};
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
    free(model_input);
    g_print("done\n");
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
    ret = run_inference(session, input_file);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseEnv(env);

    return 0;
}

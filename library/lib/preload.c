#define _GNU_SOURCE
#define G_LOG_DOMAIN ((gchar *)"IOA")

#include <compression.h>
#include <dlfcn.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <inferencing/compression.h>
#include <meta.h>
#include <settings.h>
#include <stdio.h>
#include <stdlib.h>
#include <tracing.h>

int init_main(int argc, char **argv, char **env) {

    // g_debug("Application Parameters:");
    // INFO: Used to identify application
    // TODO: Hash/Versionize application? How to identify globaly?
    // for (int i = 0; i < argc; ++i)
    //{
    //        g_debug("%s", argv[i]);
    //}

    return 0;
}
// TODO: Better way to do this!?
#if defined(__APPLE__) && defined(__MACH__)
#define INIT_MAIN                                                              \
    __attribute__((section("__INITA,__init_array"), used)) static void *main = \
        &init_main;

#else
#define INIT_MAIN                                                              \
    __attribute__((section(".init_array"), used)) static void *main =          \
        &init_main;

#endif

INIT_MAIN

void show_help(GOptionContext *context) {
    g_autofree gchar *help = NULL;
    help = g_option_context_get_help(context, TRUE, NULL);
    g_print("Setup options by specifying them in the environment IOA_OPTIONS, "
            "e.g. export IOA_OPTIONS=\"--meta-path=/tmp/meta.h5\"\n");
    g_print("%s", help);
    exit(1);
}

static void init() __attribute__((constructor));
void init() {
    char *cli_options = g_strdup_printf("IOA %s", getenv("IOA_OPTIONS"));
    int argc;
    char **argv;

    g_shell_parse_argv(cli_options, &argc, &argv, NULL);

    GError *error = NULL;
    g_autoptr(GOptionContext) context = NULL;
    static GOptionEntry entries[] = {
        {"min-size", 'm', 0, G_OPTION_ARG_INT, &opt_min_chunk_size,
         "Min size of chunks to analyze in bytes", "9"},
        {"repeat", 'r', 0, G_OPTION_ARG_INT, &opt_repeat_measurements,
         "Number of times to repeat measurements", "3"},
        {"meta-path", 'p', 0, G_OPTION_ARG_STRING, &opt_meta_data_path,
         "Path for metadata storage", "/tmp/meta.h5"},
        {"tracing", 't', 0, G_OPTION_ARG_NONE, &opt_tracing,
         "Activates tracing of MPI-Calls"},
        {"store-chunks", 's', 0, G_OPTION_ARG_NONE, &opt_store_chunks,
         "Activates chunk storage"},
        {"chunk-path", 'c', 0, G_OPTION_ARG_STRING, &opt_chunk_path,
         "Storage path of chunks for later analysis", "/tmp/chunks/"},
        {"test-compression", 'e', 0, G_OPTION_ARG_NONE, &opt_test_compression,
         "Activates compression tests according to metrics"},
        {"model-path", 'x', 0, G_OPTION_ARG_STRING, &opt_model_path,
         "Path to exported ONNX model"},
        {"settings-path", 'o', 0, G_OPTION_ARG_STRING, &opt_setting_path,
         "Path to exported ONNX settings"},
        {"inferencing", 'i', 0, G_OPTION_ARG_NONE, &opt_inferencing,
         "Run inferencing"},
        {"decompression", 'd', 0, G_OPTION_ARG_NONE, &opt_decompression,
         "Measure decompression"},
        {"verbose", 'v', 0, G_OPTION_ARG_NONE, &opt_verbose, "Verbose", NULL},
        {NULL}};

    context = g_option_context_new(NULL);
    g_option_context_add_main_entries(context, entries, NULL);

    if (!g_option_context_parse(context, &argc, &argv, &error)) {
        if (error) {
            g_printerr("CLI Error:%s\n", error->message);
            g_error_free(error);
        }
        exit(1);
    }

    // Validate required options
    if ((opt_tracing || opt_test_compression || opt_inferencing) &&
        opt_meta_data_path == NULL) {
        // TODO: Check if path exists
        g_print("--meta-path has to be specified\n");
        show_help(context);
    }

    if (opt_chunk_path != NULL &&
        !g_file_test(opt_chunk_path, G_FILE_TEST_IS_DIR)) {
        g_print("If --chunk-path is given, make sure that the path exists\n");
        show_help(context);
    } else
        opt_store_chunks = TRUE;

    if (opt_model_path != NULL &&
        !g_file_test(opt_model_path, G_FILE_TEST_IS_REGULAR)) {
        g_print("If --model-path is given, make sure that the file exists\n");
        show_help(context);
    }

    if (opt_setting_path != NULL &&
        !g_file_test(opt_setting_path, G_FILE_TEST_IS_REGULAR)) {
        g_print(
            "If --settings-path is given, make sure that the file exists\n");
        show_help(context);
    }

    if (opt_inferencing &&
        (opt_model_path == NULL || opt_setting_path == NULL)) {
        g_print("If inferencing is required, provide a model and a label path "
                "(--model-path, --settings-path)\n");
        show_help(context);
    }

    if (opt_tracing || opt_test_compression || opt_inferencing)
        _opt_action_required = TRUE;

    // g_log_set_handler(G_LOG_DOMAIN, G_LOG_LEVEL_MASK | G_LOG_LEVEL_DEBUG
    // | G_LOG_FLAG_RECURSION, g_log_default_handler, NULL);
    meta_storage = g_hash_table_new(g_direct_hash, g_direct_equal);
    trackingDB_fh = g_hash_table_new(g_direct_hash, g_direct_equal);
    trackingDB_io = g_array_new(FALSE, FALSE, sizeof(IO_Operation));
    evaluation_ops = g_array_new(FALSE, FALSE, sizeof(Evaluation_Operation));
    init_compressors();

    if (opt_inferencing)
        init_ml(opt_model_path, opt_setting_path);
}

static void fin() __attribute__((destructor));
void fin() {
    g_hash_table_destroy(trackingDB_fh);
    g_array_free(trackingDB_io, TRUE);
    if (opt_inferencing)
        cleanup_ml();
    g_debug("...done");
}

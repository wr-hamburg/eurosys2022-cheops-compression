#include <settings.h>

gboolean opt_verbose = FALSE;
gboolean opt_store_chunks = FALSE;
gboolean opt_tracing = FALSE;
gboolean opt_test_compression = FALSE;
gboolean opt_inferencing = FALSE;
gboolean opt_decompression = FALSE;
gboolean _opt_action_required = FALSE;

gint opt_min_chunk_size = 0;
gint opt_repeat_measurements = 1;
gchar const *opt_meta_data_path = NULL;
gchar const *opt_chunk_path = NULL;
gchar const *opt_model_path = NULL;
gchar const *opt_setting_path = NULL;
Metric_Type opt_metric_inferencing;

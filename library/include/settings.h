#ifndef IOA_SETTINGS_H
#define IOA_SETTINGS_H
#include <analysis/compression.h>
#include <glib.h>

extern gboolean opt_verbose;
extern gboolean opt_store_chunks;
extern gboolean opt_tracing;
extern gboolean opt_inferencing;
extern gboolean opt_test_compression;
extern gboolean opt_decompression;
extern gboolean _opt_action_required;

extern gint opt_min_chunk_size;
extern gint opt_repeat_measurements;
extern gchar const *opt_meta_data_path;
extern gchar const *opt_chunk_path;
extern gchar const *opt_model_path;
extern gchar const *opt_setting_path;
extern Metric_Type opt_metric_inferencing;

#endif
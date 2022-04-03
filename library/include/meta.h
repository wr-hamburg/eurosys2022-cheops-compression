#ifndef IOA_META_H
#define IOA_META_H
#include <glib.h>
#include <mpi.h>

extern GHashTable *meta_storage;

extern int MPI_SIZE;
extern int MPI_RANK;

void meta_mpi_insert(char *key, int value);

#endif
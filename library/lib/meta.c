#include <meta.h>

GHashTable *meta_storage;
int MPI_SIZE;
int MPI_RANK;

void meta_mpi_insert(char *key, int value) {
    // TODO: Keep track of comm changes and which one is actually used
    /*
    char comm_name[MPI_MAX_OBJECT_NAME];
    int len;
    MPI_Comm_get_name(comm, comm_name, &len);
    g_hash_table_insert(meta_storage, comm_name, value);
    */
}
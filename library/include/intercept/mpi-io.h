#ifndef IOA_MPI_IO_H
#define IOA_MPI_IO_H
#include <analysis/compression.h>
#include <glib.h>
#include <meta.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <tracing.h>

size_t count_to_size(int count, MPI_Datatype datatype);

int MPI_Init(int *argc, char ***argv);
int PMPI_Init(int *argc, char ***argv);
int MPI_Finalize();
int PMPI_Finalize();

int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);

int MPI_File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info,
                  MPI_File *fh);
int MPI_File_write(MPI_File fh, const void *buf, int count,
                   MPI_Datatype datatype, MPI_Status *status);
int MPI_File_write_all(MPI_File fh, const void *buf, int count,
                       MPI_Datatype datatype, MPI_Status *status);
int MPI_File_write_at(MPI_File fh, MPI_Offset offset, const void *buf,
                      int count, MPI_Datatype datatype, MPI_Status *status);
int MPI_File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf,
                          int count, MPI_Datatype datatype, MPI_Status *status);

int MPI_File_iwrite(MPI_File fh, const void *buf, int count,
                    MPI_Datatype datatype, MPI_Request *request);
int MPI_File_iwrite_all(MPI_File fh, const void *buf, int count,
                        MPI_Datatype datatype, MPI_Request *request);
int MPI_File_iwrite_at(MPI_File fh, MPI_Offset offset, const void *buf,
                       int count, MPI_Datatype datatype, MPIO_Request *request);
int MPI_File_iwrite_at_all(MPI_File fh, MPI_Offset offset, const void *buf,
                           int count, MPI_Datatype datatype,
                           MPI_Request *request);
#endif

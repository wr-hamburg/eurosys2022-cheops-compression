#ifndef IOA_UTIL_H
#define IOA_UTIL_H
#include <sys/time.h>
#include <time.h>

#define COUNT_OF(x)                                                            \
    ((sizeof(x) / sizeof(0 [x])) / ((size_t)(!(sizeof(x) % sizeof(0 [x])))))

long long timeInMilliseconds();
long timeInMicroseconds();

void softmax(float *input, int elem, float *out);
int max_value_index(float *array, int size);

#endif
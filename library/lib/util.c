#include <float.h>
#include <math.h>
#include <stddef.h>
#include <util.h>

long long timeInMilliseconds() {
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (((long long)tv.tv_sec) * 1000) + (tv.tv_usec / 1000);
}

long timeInMicroseconds() {
    struct timespec ts;
    // TODO: Prepare @ Setup
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
    else
        return 0;
}

void softmax(float *input, int elem, float *out) {
    float sum = 0.0;
    for (int i = 0; i < elem; ++i) {
        sum += exp(input[i]);
    }

    for (int i = 0; i < elem; ++i) {
        out[i] = exp(input[i]) / sum;
    }
}

int max_value_index(float *array, int size) {
    float max_value = FLT_MIN;
    int index = 0;
    for (int i = 0; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
            index = i;
        }
    }
    return index;
}
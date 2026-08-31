/* The same patience/binary-search LIS as lis.ppy, over the same values. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static long long longest_increasing(const long long *values, long long size, long long *tails) {
    long long length = 0;
    for (long long i = 0; i < size; i++) {
        long long value = values[i];
        long long low = 0;
        long long high = length;
        while (low < high) {
            long long middle = (low + high) / 2;
            if (tails[middle] < value) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        tails[low] = value;
        if (low == length) {
            length += 1;
        }
    }
    return length;
}

int main(void) {
    const long long size = 2000000;
    long long *values = malloc((size_t)size * sizeof(long long));
    long long *tails = calloc((size_t)size, sizeof(long long));
    for (long long i = 0; i < size; i++) {
        values[i] = (i * 7919 + (i % 977) * 104729) % 1000003;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long length = longest_increasing(values, size, tails);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("lis 2e6%17.1f ms   -> %lld\n", ms, length);
    return 0;
}

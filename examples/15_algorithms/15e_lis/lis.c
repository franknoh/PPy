/* The same problem as lis.ppy, reading the judge's input with scanf. */
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

static double since(struct timespec start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start.tv_sec) * 1000.0 + (now.tv_nsec - start.tv_nsec) / 1e6;
}

int main(void) {
    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);

    long long size = 0;
    long long *values = NULL;
    if (scanf("%lld", &size) == 1) {
        values = malloc((size_t)size * sizeof(long long));
        for (long long i = 0; i < size; i++) {
            if (scanf("%lld", &values[i]) != 1) {
                return 1;
            }
        }
    } else {
        /* Nothing piped in: the same stand-in the .ppy generates. */
        size = 1000000;
        values = malloc((size_t)size * sizeof(long long));
        for (long long i = 0; i < size; i++) {
            values[i] = (i * 7919 + (i % 977) * 104729) % 1000003;
        }
    }
    double read_ms = since(started);

    long long *tails = calloc((size_t)size, sizeof(long long));
    clock_gettime(CLOCK_MONOTONIC, &started);
    long long answer = longest_increasing(values, size, tails);
    double solve_ms = since(started);

    printf("%lld\n", answer);
    printf("# n=%lld read %.1f ms   solve %.1f ms\n", size, read_ms, solve_ms);
    return 0;
}

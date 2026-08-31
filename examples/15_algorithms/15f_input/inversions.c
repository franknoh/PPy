/* The same Fenwick inversion count as inversions.ppy, same generated data. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LIMIT 1000004

static void add(long long *tree, long long position, long long delta) {
    long long index = position + 1;
    while (index < LIMIT) {
        tree[index] += delta;
        index += index & -index;
    }
}

static long long prefix(const long long *tree, long long position) {
    long long total = 0;
    long long index = position + 1;
    while (index > 0) {
        total += tree[index];
        index -= index & -index;
    }
    return total;
}

static long long count_inversions(const long long *values, long long size, long long *tree) {
    long long total = 0;
    for (long long i = size - 1; i >= 0; i--) {
        total += prefix(tree, values[i] - 1);
        add(tree, values[i], 1);
    }
    return total;
}

int main(void) {
    const long long size = 500000;
    long long *values = malloc((size_t)size * sizeof(long long));
    long long *tree = calloc(LIMIT, sizeof(long long));
    for (long long i = 0; i < size; i++) {
        values[i] = (i * 7919) % 1000003;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long answer = count_inversions(values, size, tree);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("count inversions%8.1f ms   -> %lld\n", ms, answer);
    return 0;
}

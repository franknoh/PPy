/* The same problem as inversions.ppy, reading the judge's input with scanf. */
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
        size = 500000;
        values = malloc((size_t)size * sizeof(long long));
        for (long long i = 0; i < size; i++) {
            values[i] = (i * 7919) % 1000003;
        }
    }
    double read_ms = since(started);

    long long *tree = calloc(LIMIT, sizeof(long long));
    clock_gettime(CLOCK_MONOTONIC, &started);
    long long answer = count_inversions(values, size, tree);
    double solve_ms = since(started);

    printf("%lld\n", answer);
    printf("# n=%lld read %.1f ms   solve %.1f ms\n", size, read_ms, solve_ms);
    return 0;
}

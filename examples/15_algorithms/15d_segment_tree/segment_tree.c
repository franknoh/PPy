/* The same iterative segment tree as segment_tree.ppy, same workload. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static long long build(long long *tree, long long size) {
    for (long long i = size - 1; i > 0; i--) {
        tree[i] = tree[2 * i] + tree[2 * i + 1];
    }
    return tree[1];
}

static long long update(long long *tree, long long size, long long position, long long value) {
    long long index = position + size;
    tree[index] = value;
    index /= 2;
    while (index >= 1) {
        tree[index] = tree[2 * index] + tree[2 * index + 1];
        index /= 2;
    }
    return tree[1];
}

static long long query(const long long *tree, long long size, long long left, long long right) {
    long long total = 0;
    long long low = left + size;
    long long high = right + size;
    while (low < high) {
        if (low % 2 == 1) {
            total += tree[low];
            low += 1;
        }
        if (high % 2 == 1) {
            high -= 1;
            total += tree[high];
        }
        low /= 2;
        high /= 2;
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

    long long size = 0, rounds = 0, queries = 0;
    long long *leaves = NULL;
    long long *commands = NULL;
    int piped = scanf("%lld %lld %lld", &size, &rounds, &queries) == 3;
    if (piped) {
        leaves = malloc((size_t)size * sizeof(long long));
        for (long long i = 0; i < size; i++) {
            if (scanf("%lld", &leaves[i]) != 1) {
                return 1;
            }
        }
        commands = malloc((size_t)(rounds * 6) * sizeof(long long));
        for (long long i = 0; i < rounds * 6; i++) {
            if (scanf("%lld", &commands[i]) != 1) {
                return 1;
            }
        }
    } else {
        /* Nothing piped in: the same stand-in the .ppy generates. */
        size = 1 << 18;
        rounds = 200000;
        leaves = malloc((size_t)size * sizeof(long long));
        for (long long i = 0; i < size; i++) {
            leaves[i] = (i * 7919) % 1000;
        }
        commands = malloc((size_t)(rounds * 6) * sizeof(long long));
        for (long long step = 0; step < rounds; step++) {
            long long left = (step * 104729) % size;
            long long right = left + (step % 512) + 1;
            right = right > size ? size : right;
            long long *at = &commands[step * 6];
            at[0] = 1;
            at[1] = (step * 7919) % size;
            at[2] = (step * 31) % 1000;
            at[3] = 2;
            at[4] = left;
            at[5] = right;
        }
    }
    double read_ms = since(started);

    long long *tree = calloc((size_t)(2 * size), sizeof(long long));
    for (long long i = 0; i < size; i++) {
        tree[size + i] = leaves[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &started);
    build(tree, size);
    long long checksum = 0;
    for (long long step = 0; step < rounds; step++) {
        const long long *at = &commands[step * 6];
        update(tree, size, at[1], at[2]);
        checksum += query(tree, size, at[4], at[5]);
    }
    double solve_ms = since(started);

    printf("%lld\n", checksum);
    printf("# n=%lld q=%lld read %.1f ms   solve %.1f ms\n", size, rounds, read_ms, solve_ms);
    return 0;
}

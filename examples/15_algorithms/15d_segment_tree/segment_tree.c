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

static long long workload(long long *tree, long long size, long long rounds) {
    long long checksum = 0;
    for (long long step = 0; step < rounds; step++) {
        long long position = (step * 7919) % size;
        update(tree, size, position, (step * 31) % 1000);
        long long left = (step * 104729) % size;
        long long right = left + (step % 512) + 1;
        right = right > size ? size : right;
        checksum += query(tree, size, left, right);
    }
    return checksum;
}

int main(void) {
    const long long size = 1 << 18;
    long long *tree = calloc((size_t)(2 * size), sizeof(long long));
    for (long long i = 0; i < size; i++) {
        tree[size + i] = (i * 7919) % 1000;
    }
    build(tree, size);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long checksum = workload(tree, size, 200000);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("segment tree 2e5%8.1f ms   -> %lld\n", ms, checksum);
    return 0;
}

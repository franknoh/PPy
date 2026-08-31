/* The same heap and the same graph as dijkstra.ppy, for the same answer. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INFINITY_DISTANCE (1LL << 60)

static void sift_down(long long *keys, long long *nodes, long long size, long long start) {
    long long position = start;
    while (1) {
        long long left = 2 * position + 1;
        if (left >= size) {
            break;
        }
        long long best = left;
        long long right = left + 1;
        if (right < size && keys[right] < keys[left]) {
            best = right;
        }
        if (keys[position] <= keys[best]) {
            break;
        }
        long long key = keys[position];
        long long node = nodes[position];
        keys[position] = keys[best];
        nodes[position] = nodes[best];
        keys[best] = key;
        nodes[best] = node;
        position = best;
    }
}

static void sift_up(long long *keys, long long *nodes, long long start) {
    long long position = start;
    while (position > 0) {
        long long parent = (position - 1) / 2;
        if (keys[parent] <= keys[position]) {
            break;
        }
        long long key = keys[position];
        long long node = nodes[position];
        keys[position] = keys[parent];
        nodes[position] = nodes[parent];
        keys[parent] = key;
        nodes[parent] = node;
        position = parent;
    }
}

static long long dijkstra(const long long *offsets, const long long *targets,
                          const long long *weights, long long *distance, long long *keys,
                          long long *nodes, long long count) {
    for (long long i = 0; i < count; i++) {
        distance[i] = INFINITY_DISTANCE;
    }
    distance[0] = 0;
    keys[0] = 0;
    nodes[0] = 0;
    long long size = 1;
    while (size > 0) {
        long long best = keys[0];
        long long node = nodes[0];
        size -= 1;
        keys[0] = keys[size];
        nodes[0] = nodes[size];
        sift_down(keys, nodes, size, 0);
        if (best <= distance[node]) {
            for (long long edge = offsets[node]; edge < offsets[node + 1]; edge++) {
                long long neighbour = targets[edge];
                long long relaxed = best + weights[edge];
                if (relaxed < distance[neighbour]) {
                    distance[neighbour] = relaxed;
                    keys[size] = relaxed;
                    nodes[size] = neighbour;
                    size += 1;
                    sift_up(keys, nodes, size - 1);
                }
            }
        }
    }
    long long total = 0;
    for (long long i = 0; i < count; i++) {
        if (distance[i] < INFINITY_DISTANCE) {
            total += distance[i];
        }
    }
    return total;
}

int main(void) {
    const long long count = 200000;
    const long long degree = 6;
    long long *offsets = malloc((size_t)(count + 1) * sizeof(long long));
    long long *targets = malloc((size_t)(count * degree) * sizeof(long long));
    long long *weights = malloc((size_t)(count * degree) * sizeof(long long));
    long long *distance = malloc((size_t)count * sizeof(long long));
    long long *keys = malloc((size_t)(count * degree + 1) * sizeof(long long));
    long long *nodes = malloc((size_t)(count * degree + 1) * sizeof(long long));
    for (long long i = 0; i <= count; i++) {
        offsets[i] = i * degree;
    }
    for (long long i = 0; i < count; i++) {
        for (long long k = 0; k < degree; k++) {
            long long edge = i * degree + k;
            targets[edge] = (i * 7919 + k * 104729 + 1) % count;
            weights[edge] = (i * 31 + k * 17) % 1000 + 1;
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long total = dijkstra(offsets, targets, weights, distance, keys, nodes, count);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("dijkstra 2e5%12.1f ms   -> %lld\n", ms, total);
    return 0;
}

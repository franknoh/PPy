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
                          long long *nodes, long long count, long long source) {
    for (long long i = 0; i < count; i++) {
        distance[i] = INFINITY_DISTANCE;
    }
    distance[source] = 0;
    keys[0] = 0;
    nodes[source] = source;
    nodes[0] = source;
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

static double since(struct timespec start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start.tv_sec) * 1000.0 + (now.tv_nsec - start.tv_nsec) / 1e6;
}

int main(void) {
    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);

    long long count = 0, edges = 0, source = 0;
    long long *heads = NULL, *tails = NULL, *costs = NULL;
    if (scanf("%lld %lld %lld", &count, &edges, &source) == 3) {
        heads = malloc((size_t)edges * sizeof(long long));
        tails = malloc((size_t)edges * sizeof(long long));
        costs = malloc((size_t)edges * sizeof(long long));
        for (long long e = 0; e < edges; e++) {
            if (scanf("%lld %lld %lld", &heads[e], &tails[e], &costs[e]) != 3) {
                return 1;
            }
        }
    } else {
        /* Nothing piped in: the same stand-in graph the .ppy generates. */
        const long long degree = 6;
        count = 200000;
        edges = count * degree;
        source = 0;
        heads = malloc((size_t)edges * sizeof(long long));
        tails = malloc((size_t)edges * sizeof(long long));
        costs = malloc((size_t)edges * sizeof(long long));
        for (long long i = 0; i < count; i++) {
            for (long long k = 0; k < degree; k++) {
                long long e = i * degree + k;
                heads[e] = i;
                tails[e] = (i * 7919 + k * 104729 + 1) % count;
                costs[e] = (i * 31 + k * 17) % 1000 + 1;
            }
        }
    }
    double read_ms = since(started);

    long long *offsets = calloc((size_t)(count + 1), sizeof(long long));
    long long *cursor = calloc((size_t)count, sizeof(long long));
    long long *targets = malloc((size_t)edges * sizeof(long long));
    long long *weights = malloc((size_t)edges * sizeof(long long));
    long long *distance = malloc((size_t)count * sizeof(long long));
    long long *keys = malloc((size_t)(edges + 1) * sizeof(long long));
    long long *nodes = malloc((size_t)(edges + 1) * sizeof(long long));

    clock_gettime(CLOCK_MONOTONIC, &started);
    for (long long e = 0; e < edges; e++) {
        offsets[heads[e] + 1] += 1;
    }
    for (long long i = 0; i < count; i++) {
        offsets[i + 1] += offsets[i];
        cursor[i] = offsets[i];
    }
    for (long long e = 0; e < edges; e++) {
        long long at = cursor[heads[e]]++;
        targets[at] = tails[e];
        weights[at] = costs[e];
    }
    long long total = dijkstra(offsets, targets, weights, distance, keys, nodes, count, source);
    double solve_ms = since(started);

    printf("%lld\n", total);
    printf("# v=%lld e=%lld read %.1f ms   solve %.1f ms\n", count, edges, read_ms, solve_ms);
    return 0;
}

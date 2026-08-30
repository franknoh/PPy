/* The same kernels as algorithms.ppy, hand-written in C as the reference
 * the native backend is measured against. Same workloads, same answers,
 * same output format; 64-bit integers and doubles throughout. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static void run(const char *label, double started, long long answer) {
    printf("%-18s %9.1f ms   -> %lld\n", label, now_ms() - started, answer);
}

static long long sieve_count(int64_t *flags, long long limit) {
    for (long long i = 0; i < limit; i++) flags[i] = 1;
    flags[0] = 0;
    if (limit > 1) flags[1] = 0;
    for (long long p = 2; p * p < limit; p++) {
        if (flags[p] == 1) {
            for (long long multiple = p * p; multiple < limit; multiple += p) flags[multiple] = 0;
        }
    }
    long long total = 0;
    for (long long i = 0; i < limit; i++) total += flags[i];
    return total;
}

static long long collatz_longest(long long limit) {
    long long best = 0;
    for (long long start = 1; start < limit; start++) {
        long long n = start;
        long long steps = 0;
        while (n != 1) {
            n = n % 2 == 0 ? n / 2 : 3 * n + 1;
            steps += 1;
        }
        if (steps > best) best = steps;
    }
    return best;
}

static long long knapsack(const int64_t *weights, const int64_t *values, int64_t *table,
                          long long items, long long capacity) {
    for (long long i = 0; i <= capacity; i++) table[i] = 0;
    for (long long item = 0; item < items; item++) {
        long long weight = weights[item];
        long long value = values[item];
        for (long long room = capacity; room >= weight; room--) {
            long long candidate = table[room - weight] + value;
            if (candidate > table[room]) table[room] = candidate;
        }
    }
    return table[capacity];
}

static long long edit_distance(const int64_t *a, long long la, const int64_t *b, long long lb,
                               int64_t *row) {
    for (long long j = 0; j <= lb; j++) row[j] = j;
    for (long long i = 0; i < la; i++) {
        long long previous = row[0];
        row[0] = i + 1;
        for (long long j = 0; j < lb; j++) {
            long long current = row[j + 1];
            long long cost = a[i] != b[j] ? 1 : 0;
            long long best = previous + cost;
            if (row[j] + 1 < best) best = row[j] + 1;
            if (current + 1 < best) best = current + 1;
            row[j + 1] = best;
            previous = current;
        }
    }
    return row[lb];
}

static long long floyd_warshall(int64_t *dist, long long n) {
    for (long long k = 0; k < n; k++) {
        for (long long i = 0; i < n; i++) {
            long long through = dist[i * n + k];
            if (through < 1000000000) {
                for (long long j = 0; j < n; j++) {
                    long long candidate = through + dist[k * n + j];
                    if (candidate < dist[i * n + j]) dist[i * n + j] = candidate;
                }
            }
        }
    }
    long long total = 0;
    for (long long i = 0; i < n * n; i++) {
        if (dist[i] < 1000000000) total += dist[i];
    }
    return total;
}

static double matmul(const double *a, const double *b, double *out, long long n) {
    for (long long i = 0; i < n; i++) {
        for (long long j = 0; j < n; j++) {
            double total = 0.0;
            for (long long k = 0; k < n; k++) total += a[i * n + k] * b[k * n + j];
            out[i * n + j] = total;
        }
    }
    return out[n * n - 1];
}

static long long union_find(int64_t *parent, const int64_t *edges, long long pairs, long long n) {
    for (long long i = 0; i < n; i++) parent[i] = i;
    for (long long e = 0; e < pairs; e++) {
        long long a = edges[e * 2];
        long long b = edges[e * 2 + 1];
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        while (parent[b] != b) {
            parent[b] = parent[parent[b]];
            b = parent[b];
        }
        if (a != b) parent[a] = b;
    }
    long long components = 0;
    for (long long i = 0; i < n; i++) {
        if (parent[i] == i) components += 1;
    }
    return components;
}

static long long modpow(long long base, long long exponent, long long modulus) {
    long long result = 1;
    long long b = base % modulus;
    for (long long e = exponent; e > 0; e /= 2) {
        if (e % 2 == 1) result = result * b % modulus;
        b = b * b % modulus;
    }
    return result;
}

static long long count_primes_fermat(long long limit) {
    long long found = 0;
    for (long long n = 3; n < limit; n += 2) {
        if (modpow(2, n - 1, n) == 1) found += 1;
    }
    return found;
}

int main(void) {
    double started;

    int64_t *flags = malloc(2000000 * sizeof(int64_t));
    started = now_ms();
    long long primes = sieve_count(flags, 2000000);
    run("sieve 2e6", started, primes);
    free(flags);

    started = now_ms();
    run("collatz 3e5", started, collatz_longest(300000));

    int64_t weights[400], values[400];
    for (long long i = 0; i < 400; i++) {
        weights[i] = (i * 7919) % 97 + 1;
        values[i] = (i * 104729) % 1000 + 1;
    }
    int64_t *table = malloc(20001 * sizeof(int64_t));
    started = now_ms();
    long long best = knapsack(weights, values, table, 400, 20000);
    run("knapsack 400x2e4", started, best);
    free(table);

    int64_t *a = malloc(2000 * sizeof(int64_t));
    int64_t *b = malloc(2000 * sizeof(int64_t));
    int64_t *row = malloc(2001 * sizeof(int64_t));
    for (long long i = 0; i < 2000; i++) {
        a[i] = (i * 31) % 26;
        b[i] = (i * 17) % 26;
    }
    started = now_ms();
    long long distance = edit_distance(a, 2000, b, 2000, row);
    run("edit 2000x2000", started, distance);
    free(a);
    free(b);
    free(row);

    long long size = 220;
    int64_t *dist = malloc(size * size * sizeof(int64_t));
    for (long long i = 0; i < size; i++) {
        for (long long j = 0; j < size; j++) {
            dist[i * size + j] = i == j ? 0 : (i * 7 + j * 13) % 100 + 1;
        }
    }
    started = now_ms();
    long long total = floyd_warshall(dist, size);
    run("floyd 220", started, total);
    free(dist);

    long long n = 220;
    double *left = malloc(n * n * sizeof(double));
    double *right = malloc(n * n * sizeof(double));
    double *out = malloc(n * n * sizeof(double));
    for (long long i = 0; i < n * n; i++) {
        left[i] = (double)((i * 31) % 17);
        right[i] = (double)((i * 13) % 23);
    }
    started = now_ms();
    double corner = matmul(left, right, out, n);
    run("matmul 220", started, (long long)corner);
    free(left);
    free(right);
    free(out);

    long long nodes = 500000;
    int64_t *parent = malloc(nodes * sizeof(int64_t));
    int64_t *edges = malloc(2000000 * sizeof(int64_t));
    for (long long i = 0; i < 2000000; i++) edges[i] = (i * 7919) % nodes;
    started = now_ms();
    long long components = union_find(parent, edges, 1000000, nodes);
    run("union-find 5e5", started, components);
    free(parent);
    free(edges);

    started = now_ms();
    long long fermat = count_primes_fermat(60000);
    run("fermat 6e4", started, fermat);

    return 0;
}

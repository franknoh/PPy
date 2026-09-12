/* The same kernels in C, the loops alone: what the machine does with no call boundary. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double total(const double *values, int64_t n) {
    double result = 0.0;
    for (int64_t i = 0; i < n; i++) result += values[i];
    return result;
}

static double dot(const double *a, const double *b, int64_t n) {
    double result = 0.0;
    for (int64_t i = 0; i < n; i++) result += a[i] * b[i];
    return result;
}

static int64_t digest(const int64_t *values, int64_t n, int64_t modulus) {
    int64_t result = 0;
    for (int64_t i = 0; i < n; i++) result += values[i] % modulus;
    return result;
}

/* `round(v, places)` printed the way Python prints it: no trailing zeros. */
static const char *shortest(char *out, double value, int places) {
    snprintf(out, 64, "%.*f", places, value);
    char *end = out + strlen(out) - 1;
    while (*end == '0') *end-- = 0;
    if (*end == '.') *end = 0;
    return out;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

#define TIMED(label, expr, rounds, sink)                                           \
    do {                                                                           \
        double best = 1e9;                                                         \
        for (int r = 0; r < 5; r++) {                                              \
            double started = now();                                                \
            for (int i = 0; i < (rounds); i++) { sink = (expr); asm volatile("" : : "g"(&sink) : "memory"); } \
            double took = (now() - started) / (rounds);                            \
            if (took < best) best = took;                                          \
        }                                                                          \
        printf("# %s: %.4f ms\n", label, best * 1000.0);                           \
    } while (0)

int main(void) {
    const int64_t n = 8192;
    double *x = malloc(n * sizeof *x), *y = malloc(n * sizeof *y);
    int64_t *counts = malloc(n * sizeof *counts);
    for (int64_t i = 0; i < n; i++) {
        x[i] = (double)i * 0.001;
        y[i] = (double)i * 0.002;
        counts[i] = i * 7919;
    }
    volatile double d;
    volatile int64_t k;
    TIMED("total", total(x, n), 2000, d);
    TIMED("dot", dot(x, y, n), 2000, d);
    TIMED("dot_relaxed", dot(x, y, n), 2000, d);
    TIMED("digest", digest(counts, n, 1000003), 2000, k);
    char a[64], b[64], c[64];
    printf("%s %s %s %lld\n", shortest(a, total(x, n), 6), shortest(b, dot(x, y, n), 6),
           shortest(c, dot(x, y, n), 3), (long long)digest(counts, n, 1000003));
    free(x); free(y); free(counts);
    return 0;
}

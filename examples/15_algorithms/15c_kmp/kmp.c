/* The same failure table and scan as kmp.ppy, over the same text. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void build_failure(const long long *pattern, long long length, long long *failure) {
    failure[0] = 0;
    long long span = 0;
    for (long long i = 1; i < length; i++) {
        while (span > 0 && pattern[i] != pattern[span]) {
            span = failure[span - 1];
        }
        if (pattern[i] == pattern[span]) {
            span += 1;
        }
        failure[i] = span;
    }
}

static long long count_matches(const long long *text, long long size, const long long *pattern,
                               long long length, const long long *failure) {
    long long found = 0;
    long long span = 0;
    for (long long i = 0; i < size; i++) {
        while (span > 0 && text[i] != pattern[span]) {
            span = failure[span - 1];
        }
        if (text[i] == pattern[span]) {
            span += 1;
        }
        if (span == length) {
            found += 1;
            span = failure[span - 1];
        }
    }
    return found;
}

int main(void) {
    const long long size = 4000000;
    const long long length = 12;
    long long *text = malloc((size_t)size * sizeof(long long));
    long long pattern[12];
    long long failure[12];
    for (long long i = 0; i < size; i++) {
        text[i] = (i * 7919) % 4;
    }
    for (long long i = 0; i < length; i++) {
        pattern[i] = (i * 7919) % 4;
    }
    build_failure(pattern, length, failure);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long found = count_matches(text, size, pattern, length, failure);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("kmp 4e6%17.1f ms   -> %lld\n", ms, found);
    return 0;
}

/* The same problem as kmp.ppy, reading the judge's two lines with scanf. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_TEXT 4000064
#define MAX_PATTERN 4096

static void build_failure(const char *pattern, long long length, long long *failure) {
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

static long long count_matches(const char *text, long long size, const char *pattern,
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

static double since(struct timespec start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start.tv_sec) * 1000.0 + (now.tv_nsec - start.tv_nsec) / 1e6;
}

int main(void) {
    static char text[MAX_TEXT];
    static char pattern[MAX_PATTERN];
    static long long failure[MAX_PATTERN];

    struct timespec started;
    clock_gettime(CLOCK_MONOTONIC, &started);
    if (scanf("%4000063s", text) != 1 || scanf("%4095s", pattern) != 1) {
        /* Nothing piped in: the same stand-in the .ppy generates. */
        for (long long i = 0; i < 4000000; i++) {
            text[i] = (char)('a' + (i * 7919) % 4);
        }
        text[4000000] = '\0';
        memcpy(pattern, text, 12);
        pattern[12] = '\0';
    }
    long long size = (long long)strlen(text);
    long long length = (long long)strlen(pattern);
    double read_ms = since(started);

    clock_gettime(CLOCK_MONOTONIC, &started);
    build_failure(pattern, length, failure);
    long long found = count_matches(text, size, pattern, length, failure);
    double solve_ms = since(started);

    printf("%lld\n", found);
    printf("# |t|=%lld |p|=%lld read %.1f ms   solve %.1f ms\n", size, length, read_ms, solve_ms);
    return 0;
}

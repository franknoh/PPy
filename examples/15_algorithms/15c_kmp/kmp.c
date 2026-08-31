/* The same problem as kmp.ppy, reading the judge's two lines with scanf. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(void) {
    static char text[MAX_TEXT];
    static char pattern[MAX_PATTERN];
    static long long failure[MAX_PATTERN];

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

    build_failure(pattern, length, failure);
    long long found = count_matches(text, size, pattern, length, failure);

    printf("%lld\n", found);
    return 0;
}

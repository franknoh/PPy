/* The same problem as lis.ppy, reading the judge's input with scanf. */
#include <stdio.h>
#include <stdlib.h>

static long long longest_increasing(const long long *values, long long size, long long *tails) {
    long long length = 0;
    for (long long i = 0; i < size; i++) {
        long long value = values[i];
        long long low = 0;
        long long high = length;
        while (low < high) {
            long long middle = (low + high) / 2;
            if (tails[middle] < value) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        tails[low] = value;
        if (low == length) {
            length += 1;
        }
    }
    return length;
}

int main(void) {

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
        /* Nothing piped in: the same stand-in the .ppy generates. */
        size = 1000000;
        values = malloc((size_t)size * sizeof(long long));
        for (long long i = 0; i < size; i++) {
            values[i] = (i * 7919 + (i % 977) * 104729) % 1000003;
        }
    }

    long long *tails = calloc((size_t)size, sizeof(long long));
    long long answer = longest_increasing(values, size, tails);

    printf("%lld\n", answer);
    return 0;
}

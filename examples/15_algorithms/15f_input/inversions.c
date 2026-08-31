/* The same problem as inversions.ppy, reading the judge's input with scanf. */
#include <stdio.h>
#include <stdlib.h>

#define LIMIT 1000004

static void add(long long *tree, long long position, long long delta) {
    long long index = position + 1;
    while (index < LIMIT) {
        tree[index] += delta;
        index += index & -index;
    }
}

static long long prefix(const long long *tree, long long position) {
    long long total = 0;
    long long index = position + 1;
    while (index > 0) {
        total += tree[index];
        index -= index & -index;
    }
    return total;
}

static long long count_inversions(const long long *values, long long size, long long *tree) {
    long long total = 0;
    for (long long i = size - 1; i >= 0; i--) {
        total += prefix(tree, values[i] - 1);
        add(tree, values[i], 1);
    }
    return total;
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
        size = 500000;
        values = malloc((size_t)size * sizeof(long long));
        for (long long i = 0; i < size; i++) {
            values[i] = (i * 7919) % 1000003;
        }
    }

    long long *tree = calloc(LIMIT, sizeof(long long));
    long long answer = count_inversions(values, size, tree);

    printf("%lld\n", answer);
    return 0;
}

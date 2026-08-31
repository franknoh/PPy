/* The same bitmask backtracking as nqueens.ppy, for the same answer. */
#include <stdio.h>
#include <time.h>

static long long solve(long long full, long long columns, long long diagonal,
                       long long antidiagonal) {
    if (columns == full) {
        return 1;
    }
    long long total = 0;
    long long available = full & ~(columns | diagonal | antidiagonal);
    while (available != 0) {
        long long bit = available & -available;
        available -= bit;
        total += solve(full, columns | bit, ((diagonal | bit) << 1) & full,
                       (antidiagonal | bit) >> 1);
    }
    return total;
}

int main(void) {
    int n = 12;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long long answer = solve((1LL << n) - 1, 0, 0, 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("n-queens n=%d%12.1f ms   -> %lld\n", n, ms, answer);
    return 0;
}

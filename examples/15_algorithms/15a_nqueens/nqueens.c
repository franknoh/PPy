/* The same problem as nqueens.ppy, reading the judge's input with scanf. */
#include <stdio.h>

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
    int n = 0;
    if (scanf("%d", &n) != 1) {
        n = 12;
    }

    long long answer = solve((1LL << n) - 1, 0, 0, 0);

    printf("%lld\n", answer);
    return 0;
}

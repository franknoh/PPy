/* The same three questions put to C: signed overflow is undefined, division truncates. */
#include <stdio.h>

static long long may_overflow(long long n) {
    long long result = 1;
    for (long long i = 1; i <= n; i++) result *= i;
    return result;
}

static long long floor_semantics(long long a, long long b) { return a / b; }
static long long modulo_semantics(long long a, long long b) { return a % b; }

int main(void) {
    printf("%lld\n", may_overflow(20));
    printf("%lld\n", may_overflow(30));
    printf("%lld %lld\n", floor_semantics(-7, 2), floor_semantics(7, 2));
    printf("%lld %lld\n", modulo_semantics(-7, 2), modulo_semantics(7, -2));
    return 0;
}

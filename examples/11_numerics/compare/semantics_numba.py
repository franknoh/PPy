"""The three questions of `numerics.ppy` put to Numba: `@njit` on a machine integer."""

from numba import njit


@njit
def may_overflow(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


@njit
def floor_semantics(a, b):
    return a // b


@njit
def modulo_semantics(a, b):
    return a % b


print(may_overflow(20))
print(may_overflow(30))
print(floor_semantics(-7, 2), floor_semantics(7, 2))
print(modulo_semantics(-7, 2), modulo_semantics(7, -2))

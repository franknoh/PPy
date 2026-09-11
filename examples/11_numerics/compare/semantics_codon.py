"""The same three questions put to Codon: Python syntax, a 64-bit `int`."""


def may_overflow(n: int) -> int:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def floor_semantics(a: int, b: int) -> int:
    return a // b


def modulo_semantics(a: int, b: int) -> int:
    return a % b


print(may_overflow(20))
print(may_overflow(30))
print(floor_semantics(-7, 2), floor_semantics(7, 2))
print(modulo_semantics(-7, 2), modulo_semantics(7, -2))

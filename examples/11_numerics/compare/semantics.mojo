"""The same three questions put to Mojo 1.0: `Int` is a machine word."""


def may_overflow(n: Int) -> Int:
    var result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def floor_semantics(a: Int, b: Int) -> Int:
    return a // b


def modulo_semantics(a: Int, b: Int) -> Int:
    return a % b


def main():
    print(may_overflow(20))
    print(may_overflow(30))
    print(floor_semantics(-7, 2), floor_semantics(7, 2))
    print(modulo_semantics(-7, 2), modulo_semantics(7, -2))

"""N-Queens: how many ways N queens fit on an N by N board.

Input:  N on the first line.
Output: how many ways N queens can be placed without attacking each other.
"""


def solve(full, columns, diagonal, antidiagonal):
    if columns == full:
        return 1
    total = 0
    available = full & ~(columns | diagonal | antidiagonal)
    while available != 0:
        bit = available & -available
        available -= bit
        total += solve(
            full,
            columns | bit,
            ((diagonal | bit) << 1) & full,
            (antidiagonal | bit) >> 1,
        )
    return total


def read_size():
    """The judge's input, or the stand-in this example uses on its own."""
    try:
        return int(input())
    except EOFError:
        return 12


def main():
    n = read_size()
    print(solve((1 << n) - 1, 0, 0, 0))


main()

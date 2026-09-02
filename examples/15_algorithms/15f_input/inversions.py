"""Counting inversions with a Fenwick tree.

Input:  N on the first line, then N integers.
Output: how many pairs are out of order.
"""

import array

LIMIT = 1000004


def add(tree, position, delta):
    index = position + 1
    while index < LIMIT:
        tree[index] += delta
        index += index & -index
    return index


def prefix(tree, position):
    total = 0
    index = position + 1
    while index > 0:
        total += tree[index]
        index -= index & -index
    return total


def count_inversions(values, tree):
    total = 0
    for i in range(len(values) - 1, -1, -1):
        total += prefix(tree, values[i] - 1)
        add(tree, values[i], 1)
    return total


def read_values():
    """The judge's input, or a generated stand-in when nothing is piped in."""
    try:
        count = int(input())
    except EOFError:
        return array.array("q", [(i * 7919) % 1000003 for i in range(500000)])
    values = array.array("q", [0] * count)
    for i in range(count):
        values[i] = int(input())
    return values


def main():
    values = read_values()
    tree = array.array("q", [0] * LIMIT)
    print(count_inversions(values, tree))


main()

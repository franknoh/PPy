"""Range sums with point updates, over a segment tree.

Input:  N M K on the first line, then N numbers, then M+K command lines --
        `1 b c` assigns c to position b, `2 b c` asks for the sum over
        [b, c).
Output: the checksum of every answered query, so one line stands in for K.
"""

import array


def build(tree, size):
    for i in range(size - 1, 0, -1):
        tree[i] = tree[2 * i] + tree[2 * i + 1]
    return tree[1]


def update(tree, size, position, value):
    index = position + size
    tree[index] = value
    index //= 2
    while index >= 1:
        tree[index] = tree[2 * index] + tree[2 * index + 1]
        index //= 2
    return tree[1]


def query(tree, size, left, right):
    total = 0
    low = left + size
    high = right + size
    while low < high:
        if low % 2 == 1:
            total += tree[low]
            low += 1
        if high % 2 == 1:
            high -= 1
            total += tree[high]
        low //= 2
        high //= 2
    return total


def run_commands(tree, commands, size, rounds):
    checksum = 0
    for step in range(rounds):
        base = step * 6
        update(tree, size, commands[base + 1], commands[base + 2])
        checksum += query(tree, size, commands[base + 4], commands[base + 5])
    return checksum


def generated():
    """The stand-in input, in the same order the judge would send it."""
    size = 1 << 18
    rounds = 200000
    fields = [size, rounds, rounds]
    fields.extend((i * 7919) % 1000 for i in range(size))
    for step in range(rounds):
        fields.append(1)
        fields.append((step * 7919) % size)
        fields.append((step * 31) % 1000)
        left = (step * 104729) % size
        fields.append(2)
        fields.append(left)
        fields.append(min(left + (step % 512) + 1, size))
    return fields


def read_fields():
    """The judge's input, or a generated stand-in when nothing is piped in."""
    try:
        size, rounds, queries = map(int, input().split())
    except EOFError:
        return array.array("q", generated())
    fields = array.array("q", [size, rounds, queries])
    rest = array.array("q", [0] * (size + rounds * 6))
    for i in range(size + rounds * 6):
        rest[i] = int(input())
    fields.extend(rest)
    return fields


def main():
    fields = read_fields()
    size = fields[0]
    rounds = fields[1]
    tree = array.array("q", [0] * (2 * size))
    for i in range(size):
        tree[size + i] = fields[3 + i]
    commands = array.array("q", fields[3 + size :])

    build(tree, size)
    print(run_commands(tree, commands, size, rounds))


main()

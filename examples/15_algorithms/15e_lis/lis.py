"""Longest increasing subsequence: https://www.acmicpc.net/problem/12015

Input:  N on the first line, then N integers.
Output: the length of the longest strictly increasing subsequence.
"""

import array


def longest_increasing(values, tails):
    length = 0
    for i in range(len(values)):
        value = values[i]
        low = 0
        high = length
        while low < high:
            middle = (low + high) // 2
            if tails[middle] < value:
                low = middle + 1
            else:
                high = middle
        tails[low] = value
        if low == length:
            length += 1
    return length


def read_values():
    """The judge's input, or a generated stand-in when nothing is piped in."""
    try:
        count = int(input())
    except EOFError:
        return array.array("q", [(i * 7919 + (i % 977) * 104729) % 1000003 for i in range(1000000)])
    values = array.array("q", [0] * count)
    for i in range(count):
        values[i] = int(input())
    return values


def main():
    values = read_values()
    tails = array.array("q", [0] * len(values))
    print(longest_increasing(values, tails))


main()

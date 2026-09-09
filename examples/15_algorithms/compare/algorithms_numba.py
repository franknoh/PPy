"""The eight kernels of `algorithms.ppy` under Numba's `@njit`: the same loops over NumPy arrays."""

import time

import numpy as np
from numba import njit


@njit
def sieve_count(flags, limit):
    for i in range(limit):
        flags[i] = 1
    flags[0] = 0
    if limit > 1:
        flags[1] = 0
    p = 2
    while p * p < limit:
        if flags[p] == 1:
            multiple = p * p
            while multiple < limit:
                flags[multiple] = 0
                multiple += p
        p += 1
    total = 0
    for i in range(limit):
        total += flags[i]
    return total


@njit
def collatz_longest(limit):
    best = 0
    for start in range(1, limit):
        n = start
        steps = 0
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
        best = max(best, steps)
    return best


@njit
def knapsack(weights, values, table, capacity):
    for i in range(capacity + 1):
        table[i] = 0
    for item in range(len(weights)):
        weight = weights[item]
        value = values[item]
        room = capacity
        while room >= weight:
            candidate = table[room - weight] + value
            if candidate > table[room]:
                table[room] = candidate
            room -= 1
    return table[capacity]


@njit
def edit_distance(a, b, row):
    width = len(b)
    for j in range(width + 1):
        row[j] = j
    for i in range(len(a)):
        previous = row[0]
        row[0] = i + 1
        for j in range(width):
            current = row[j + 1]
            cost = 0
            if a[i] != b[j]:
                cost = 1
            best = previous + cost
            best = min(best, row[j] + 1)
            best = min(best, current + 1)
            row[j + 1] = best
            previous = current
    return row[width]


@njit
def floyd_warshall(dist, n):
    for k in range(n):
        for i in range(n):
            through = dist[i * n + k]
            if through < 1000000000:
                for j in range(n):
                    candidate = through + dist[k * n + j]
                    if candidate < dist[i * n + j]:
                        dist[i * n + j] = candidate
    total = 0
    for i in range(n * n):
        if dist[i] < 1000000000:
            total += dist[i]
    return total


@njit
def matmul(a, b, out, n):
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a[i * n + k] * b[k * n + j]
            out[i * n + j] = total
    return out[n * n - 1]


@njit
def union_find(parent, edges, n):
    for i in range(n):
        parent[i] = i
    pairs = len(edges) // 2
    for e in range(pairs):
        a = edges[e * 2]
        b = edges[e * 2 + 1]
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a != b:
            parent[a] = b
    components = 0
    for i in range(n):
        if parent[i] == i:
            components += 1
    return components


@njit
def modpow(base, exponent, modulus):
    result = 1
    b = base % modulus
    e = exponent
    while e > 0:
        if e % 2 == 1:
            result = result * b % modulus
        b = b * b % modulus
        e = e // 2
    return result


@njit
def count_primes_fermat(limit):
    found = 0
    for n in range(3, limit, 2):
        if modpow(2, n - 1, n) == 1:
            found += 1
    return found


def run(label, seconds, answer):
    print(f"{label:<18s} {seconds * 1000.0:9.1f} ms   -> {answer}")


def timed(label, call):
    call()  # compile
    start = time.perf_counter()
    answer = call()
    run(label, time.perf_counter() - start, answer)


def main():
    flags = np.zeros(2000000, dtype=np.int64)
    timed("sieve 2e6", lambda: sieve_count(flags, 2000000))
    timed("collatz 3e5", lambda: collatz_longest(300000))
    weights = np.array([(i * 7919) % 97 + 1 for i in range(400)], dtype=np.int64)
    values = np.array([(i * 104729) % 1000 + 1 for i in range(400)], dtype=np.int64)
    table = np.zeros(20001, dtype=np.int64)
    timed("knapsack 400x2e4", lambda: knapsack(weights, values, table, 20000))
    a = np.array([(i * 31) % 26 for i in range(2000)], dtype=np.int64)
    b = np.array([(i * 17) % 26 for i in range(2000)], dtype=np.int64)
    row = np.zeros(2001, dtype=np.int64)
    timed("edit 2000x2000", lambda: edit_distance(a, b, row))
    size = 220
    dist = np.zeros(size * size, dtype=np.int64)
    for i in range(size):
        for j in range(size):
            dist[i * size + j] = 0 if i == j else (i * 7 + j * 13) % 100 + 1
    fresh = dist.copy()
    floyd_warshall(fresh, size)  # compile on a copy: the kernel relaxes in place
    start = time.perf_counter()
    total = floyd_warshall(dist, size)
    run("floyd 220", time.perf_counter() - start, total)
    n = 220
    left = np.array([float((i * 31) % 17) for i in range(n * n)])
    right = np.array([float((i * 13) % 23) for i in range(n * n)])
    out = np.zeros(n * n)
    timed("matmul 220", lambda: int(matmul(left, right, out, n)))
    nodes = 500000
    parent = np.zeros(nodes, dtype=np.int64)
    edges = np.array([(i * 7919) % nodes for i in range(2000000)], dtype=np.int64)
    timed("union-find 5e5", lambda: union_find(parent, edges, nodes))
    timed("fermat 6e4", lambda: count_primes_fermat(60000))


main()

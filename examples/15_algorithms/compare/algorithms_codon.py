"""The eight kernels of `algorithms.ppy` under Codon: the same loops over typed lists."""

from time import time


def sieve_count(flags: List[int], limit: int) -> int:
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


def collatz_longest(limit: int) -> int:
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


def knapsack(weights: List[int], values: List[int], table: List[int], capacity: int) -> int:
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


def edit_distance(a: List[int], b: List[int], row: List[int]) -> int:
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


def floyd_warshall(dist: List[int], n: int) -> int:
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


def matmul(a: List[float], b: List[float], out: List[float], n: int) -> float:
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a[i * n + k] * b[k * n + j]
            out[i * n + j] = total
    return out[n * n - 1]


def union_find(parent: List[int], edges: List[int], n: int) -> int:
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


def modpow(base: int, exponent: int, modulus: int) -> int:
    result = 1
    b = base % modulus
    e = exponent
    while e > 0:
        if e % 2 == 1:
            result = result * b % modulus
        b = b * b % modulus
        e = e // 2
    return result


def count_primes_fermat(limit: int) -> int:
    found = 0
    for n in range(3, limit, 2):
        if modpow(2, n - 1, n) == 1:
            found += 1
    return found


def run(label: str, seconds: float, answer: int):
    print(label, round(seconds * 1000.0, 1), "ms   ->", answer)


def main():
    flags = [0 for _ in range(2000000)]
    start = time()
    primes = sieve_count(flags, 2000000)
    run("sieve 2e6", time() - start, primes)

    start = time()
    longest = collatz_longest(300000)
    run("collatz 3e5", time() - start, longest)

    weights = [(i * 7919) % 97 + 1 for i in range(400)]
    values = [(i * 104729) % 1000 + 1 for i in range(400)]
    table = [0 for _ in range(20001)]
    start = time()
    best = knapsack(weights, values, table, 20000)
    run("knapsack 400x2e4", time() - start, best)

    a = [(i * 31) % 26 for i in range(2000)]
    b = [(i * 17) % 26 for i in range(2000)]
    row = [0 for _ in range(2001)]
    start = time()
    distance = edit_distance(a, b, row)
    run("edit 2000x2000", time() - start, distance)

    size = 220
    dist = [0 for _ in range(size * size)]
    for i in range(size):
        for j in range(size):
            dist[i * size + j] = 0 if i == j else (i * 7 + j * 13) % 100 + 1
    start = time()
    total = floyd_warshall(dist, size)
    run("floyd 220", time() - start, total)

    n = 220
    left = [float((i * 31) % 17) for i in range(n * n)]
    right = [float((i * 13) % 23) for i in range(n * n)]
    out = [0.0 for _ in range(n * n)]
    start = time()
    corner = matmul(left, right, out, n)
    run("matmul 220", time() - start, int(corner))

    nodes = 500000
    parent = [0 for _ in range(nodes)]
    edges = [(i * 7919) % nodes for i in range(2000000)]
    start = time()
    components = union_find(parent, edges, nodes)
    run("union-find 5e5", time() - start, components)

    start = time()
    fermat = count_primes_fermat(60000)
    run("fermat 6e4", time() - start, fermat)


main()

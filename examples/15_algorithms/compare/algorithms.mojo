"""The eight kernels of `algorithms.ppy` in Mojo: the same loops over Lists, typed by hand."""

from std.time import perf_counter_ns


def sieve_count(mut flags: List[Int64], limit: Int) -> Int64:
    for i in range(limit):
        flags[i] = 1
    flags[0] = 0
    if limit > 1:
        flags[1] = 0
    var p = 2
    while p * p < limit:
        if flags[p] == 1:
            var multiple = p * p
            while multiple < limit:
                flags[multiple] = 0
                multiple += p
        p += 1
    var total: Int64 = 0
    for i in range(limit):
        total += flags[i]
    return total


def collatz_longest(limit: Int) -> Int64:
    var best: Int64 = 0
    for start in range(1, limit):
        var n = Int64(start)
        var steps: Int64 = 0
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
        best = max(best, steps)
    return best


def knapsack(weights: List[Int64], values: List[Int64], mut table: List[Int64], capacity: Int) -> Int64:
    for i in range(capacity + 1):
        table[i] = 0
    for item in range(len(weights)):
        var weight = Int(weights[item])
        var value = values[item]
        var room = capacity
        while room >= weight:
            var candidate = table[room - weight] + value
            if candidate > table[room]:
                table[room] = candidate
            room -= 1
    return table[capacity]


def edit_distance(a: List[Int64], b: List[Int64], mut row: List[Int64]) -> Int64:
    var width = len(b)
    for j in range(width + 1):
        row[j] = Int64(j)
    for i in range(len(a)):
        var previous = row[0]
        row[0] = Int64(i + 1)
        for j in range(width):
            var current = row[j + 1]
            var cost: Int64 = 0
            if a[i] != b[j]:
                cost = 1
            var best = previous + cost
            best = min(best, row[j] + 1)
            best = min(best, current + 1)
            row[j + 1] = best
            previous = current
    return row[width]


def floyd_warshall(mut dist: List[Int64], n: Int) -> Int64:
    for k in range(n):
        for i in range(n):
            var through = dist[i * n + k]
            if through < 1000000000:
                for j in range(n):
                    var candidate = through + dist[k * n + j]
                    if candidate < dist[i * n + j]:
                        dist[i * n + j] = candidate
    var total: Int64 = 0
    for i in range(n * n):
        if dist[i] < 1000000000:
            total += dist[i]
    return total


def matmul(a: List[Float64], b: List[Float64], mut out: List[Float64], n: Int) -> Float64:
    for i in range(n):
        for j in range(n):
            var total: Float64 = 0.0
            for k in range(n):
                total += a[i * n + k] * b[k * n + j]
            out[i * n + j] = total
    return out[n * n - 1]


def union_find(mut parent: List[Int64], edges: List[Int64], n: Int) -> Int64:
    for i in range(n):
        parent[i] = Int64(i)
    var pairs = len(edges) // 2
    for e in range(pairs):
        var a = Int(edges[e * 2])
        var b = Int(edges[e * 2 + 1])
        while Int(parent[a]) != a:
            parent[a] = parent[Int(parent[a])]
            a = Int(parent[a])
        while Int(parent[b]) != b:
            parent[b] = parent[Int(parent[b])]
            b = Int(parent[b])
        if a != b:
            parent[a] = Int64(b)
    var components: Int64 = 0
    for i in range(n):
        if Int(parent[i]) == i:
            components += 1
    return components


def modpow(base: Int64, exponent: Int64, modulus: Int64) -> Int64:
    var result: Int64 = 1
    var b = base % modulus
    var e = exponent
    while e > 0:
        if e % 2 == 1:
            result = result * b % modulus
        b = b * b % modulus
        e = e // 2
    return result


def count_primes_fermat(limit: Int) -> Int64:
    var found: Int64 = 0
    for n in range(3, limit, 2):
        if modpow(2, Int64(n - 1), Int64(n)) == 1:
            found += 1
    return found


def report(label: String, started: Int, answer: Int64):
    var ms = Float64(perf_counter_ns() - started) / 1e6
    print(label, ms, "ms   ->", answer)


def main():
    var flags = List[Int64](length=2000000, fill=0)
    var started = perf_counter_ns()
    var primes = sieve_count(flags, 2000000)
    report("sieve 2e6", started, primes)

    started = perf_counter_ns()
    var longest = collatz_longest(300000)
    report("collatz 3e5", started, longest)

    var weights = List[Int64](capacity=400)
    var values = List[Int64](capacity=400)
    for i in range(400):
        weights.append(Int64((i * 7919) % 97 + 1))
        values.append(Int64((i * 104729) % 1000 + 1))
    var table = List[Int64](length=20001, fill=0)
    started = perf_counter_ns()
    var best = knapsack(weights, values, table, 20000)
    report("knapsack 400x2e4", started, best)

    var a = List[Int64](capacity=2000)
    var b = List[Int64](capacity=2000)
    for i in range(2000):
        a.append(Int64((i * 31) % 26))
        b.append(Int64((i * 17) % 26))
    var row = List[Int64](length=2001, fill=0)
    started = perf_counter_ns()
    var distance = edit_distance(a, b, row)
    report("edit 2000x2000", started, distance)

    var size = 220
    var dist = List[Int64](length=size * size, fill=0)
    for i in range(size):
        for j in range(size):
            if i == j:
                dist[i * size + j] = 0
            else:
                dist[i * size + j] = Int64((i * 7 + j * 13) % 100 + 1)
    started = perf_counter_ns()
    var total = floyd_warshall(dist, size)
    report("floyd 220", started, total)

    var n = 220
    var left = List[Float64](capacity=n * n)
    var right = List[Float64](capacity=n * n)
    for i in range(n * n):
        left.append(Float64((i * 31) % 17))
        right.append(Float64((i * 13) % 23))
    var out = List[Float64](length=n * n, fill=0.0)
    started = perf_counter_ns()
    var corner = matmul(left, right, out, n)
    report("matmul 220", started, Int64(corner))

    var nodes = 500000
    var parent = List[Int64](length=nodes, fill=0)
    var edges = List[Int64](capacity=2000000)
    for i in range(2000000):
        edges.append(Int64((i * 7919) % nodes))
    started = perf_counter_ns()
    var components = union_find(parent, edges, nodes)
    report("union-find 5e5", started, components)

    started = perf_counter_ns()
    var fermat = count_primes_fermat(60000)
    report("fermat 6e4", started, fermat)

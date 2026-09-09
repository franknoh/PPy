"""The four kernels in Mojo: `parallelize` over indices, partial sums by hand; timed warm."""

from max.algorithm import parallelize
from std.time import perf_counter_ns

comptime N = 8_000_000
comptime WORKERS = 64


def squares(mut out: List[Int64]) -> Int64:
    var p = out.unsafe_ptr()

    @parameter
    def one(i: Int):
        p[i] = Int64(i) * Int64(i)

    parallelize[one](N)
    return out[N - 1]


def dot(a: List[Float64], b: List[Float64]) -> Float64:
    var total: Float64 = 0.0
    for i in range(N):
        total += a[i] * b[i]
    return total


def dot_relaxed(a: List[Float64], b: List[Float64]) -> Float64:
    var partial = List[Float64](length=WORKERS, fill=0.0)
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var pp = partial.unsafe_ptr()
    var chunk = (N + WORKERS - 1) // WORKERS

    @parameter
    def piece(w: Int):
        var total: Float64 = 0.0
        var stop = min(N, (w + 1) * chunk)
        for i in range(w * chunk, stop):
            total += ap[i] * bp[i]
        pp[w] = total

    parallelize[piece](WORKERS)
    var total: Float64 = 0.0
    for w in range(WORKERS):
        total += partial[w]
    return total


def count_odd(xs: List[Int64], scale: Int64) -> Int64:
    var partial = List[Int64](length=WORKERS, fill=0)
    var xp = xs.unsafe_ptr()
    var pp = partial.unsafe_ptr()
    var chunk = (N + WORKERS - 1) // WORKERS

    @parameter
    def piece(w: Int):
        var hits: Int64 = 0
        var stop = min(N, (w + 1) * chunk)
        for i in range(w * chunk, stop):
            if xp[i] % 2 == 1:
                hits += scale
        pp[w] = hits

    parallelize[piece](WORKERS)
    var hits: Int64 = 0
    for w in range(WORKERS):
        hits += partial[w]
    return hits


def fill(mut out: List[Float64], base: Float64) -> Float64:
    for i in range(N):
        out[i] = base + Float64(i)
    var acc: Float64 = 0.0
    for j in range(N):
        acc += out[j]
    return acc


def main():
    var a = List[Float64](capacity=N)
    var b = List[Float64](capacity=N)
    var ints = List[Int64](capacity=N)
    for i in range(N):
        a.append(0.001 * Float64(i))
        b.append(1.0 / Float64(1 + i))
        ints.append(Int64(i % 3))
    var room = List[Int64](length=N, fill=0)
    var floats = List[Float64](length=N, fill=0.0)
    print(squares(room), dot(a, b), dot_relaxed(a, b), count_odd(ints, 3), fill(floats, 0.5))
    var labels = List[String]()
    labels.append("squares")
    labels.append("dot")
    labels.append("dot_relaxed")
    labels.append("count_odd")
    labels.append("fill")
    for k in range(5):
        var best: Float64 = 1e18
        for _ in range(6):
            var started = perf_counter_ns()
            if k == 0:
                _ = squares(room)
            elif k == 1:
                _ = dot(a, b)
            elif k == 2:
                _ = dot_relaxed(a, b)
            elif k == 3:
                _ = count_odd(ints, 3)
            else:
                _ = fill(floats, 0.5)
            var elapsed = Float64(perf_counter_ns() - started) / 1e6
            if elapsed < best:
                best = elapsed
        print("# " + labels[k] + ": " + String(best) + " ms")

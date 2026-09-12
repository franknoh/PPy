# cython: language_level=3, boundscheck=False, wraparound=False
"""The same kernels in Cython: typed memoryviews and C locals, built by `cythonize`."""

import time


cpdef double total(double[::1] values):
    cdef double result = 0.0
    cdef Py_ssize_t i
    for i in range(values.shape[0]):
        result += values[i]
    return result


cpdef double dot(double[::1] a, double[::1] b):
    cdef double result = 0.0
    cdef Py_ssize_t i
    for i in range(a.shape[0]):
        result += a[i] * b[i]
    return result


cpdef long long digest(long long[::1] values, long long modulus):
    cdef long long result = 0
    cdef Py_ssize_t i
    for i in range(values.shape[0]):
        result += values[i] % modulus
    return result


def timed(label, run, rounds):
    best = 1e9
    answer = None
    for _ in range(5):
        started = time.perf_counter()
        for _i in range(rounds):
            answer = run()
        best = min(best, (time.perf_counter() - started) / rounds)
    print(f"# {label}: {best * 1000:.4f} ms")
    return answer


def main():
    import array
    size = 8192
    x = array.array("d", [float(i) * 0.001 for i in range(size)])
    y = array.array("d", [float(i) * 0.002 for i in range(size)])
    counts = array.array("q", [i * 7919 for i in range(size)])
    timed("total", lambda: total(x), 2000)
    timed("dot", lambda: dot(x, y), 2000)
    timed("dot_relaxed", lambda: dot(x, y), 2000)
    timed("digest", lambda: digest(counts, 1000003), 2000)
    print(round(total(x), 6), round(dot(x, y), 6), round(dot(x, y), 3), digest(counts, 1000003))


if __name__ == "__main__":
    main()

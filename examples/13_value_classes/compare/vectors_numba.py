"""The same work under Numba: `@jitclass`es of float fields, `@njit` functions over them."""

import time

from numba import float64, njit
from numba.experimental import jitclass


@jitclass([("x", float64), ("y", float64), ("z", float64)])
class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def norm2(self):
        return self.x * self.x + self.y * self.y + self.z * self.z


@njit
def distance2(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz


@jitclass([("origin", float64), ("direction", float64)])
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


@njit
def travel(ray, count):
    position = ray.origin
    total = 0.0
    for i in range(count):
        position = position * 0.999999 + ray.direction * (i % 3)
        total += position
    return total


def main():
    a = Vec3(1.0, 2.0, 3.0)
    b = Vec3(0.5, 0.25, 0.125)
    distance2(a, b)
    ray = Ray(1.0, 0.001)
    travel(ray, 1000)
    best_calls = 1e9
    best_loop = 1e9
    for _ in range(5):
        started = time.perf_counter()
        acc = 0.0
        for _i in range(1_000_000):
            acc += distance2(a, b)
        best_calls = min(best_calls, (time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        total = travel(ray, 8_000_000)
        best_loop = min(best_loop, (time.perf_counter() - started) * 1000.0)
    print(f"# distance2, a million calls from Python: {best_calls:.2f} ms")
    print(f"# travel, eight million steps over a Ray natively: {best_loop:.2f} ms")
    print(f"{acc:.6f} {total:.3f}")


main()

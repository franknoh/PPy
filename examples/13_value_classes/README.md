# Value classes

An all-scalar `@dataclass` has no boxed form in native code. `Vec3` is
three floats, and native code treats it that way: `distance2(a, b)` passes
six doubles, not two object pointers, and `a.norm2()` is a native function
whose `self` is three scalars.

## Flattened at the ABI, guarded on the class

```python
@dataclass
class Vec3:
    x: float
    y: float
    z: float

    @ppy.pure
    @ppy.opt(3)
    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z
```

The generated boundary reads the fields, checks the class is exactly
`Vec3`, and calls the native function with scalars. `distance2(Tracked(1.0,
2.0, 3.0), b)` shows the guard: `Tracked` is a subclass, the exact-class
check refuses it, and the Python body answers. A class too wide for the ABI
stays boxed, and `ppy explain` says so. Eight float operations cost about
50 ns per call from Python here, most of it the boundary; `steps` runs a
200-iteration loop over a `Ray`'s two fields at about 200 ns per call, with
the `Ray` never leaving registers.

## Operators dispatch statically

Inside native code `a + b` on a value class calls the class's own
`__add__`, lowered like any other native function. There is no fallback to
Python's dynamic dispatch and no `__radd__` search; a class without a native
operator is refused with the reason.

## Compared with Numba's `@jitclass`

A million `distance2` calls from Python, and a native loop over a `Ray`'s
two fields, in [`compare/`](compare/):
[`vectors_bench.ppy`](compare/vectors_bench.ppy) -- under `ppy run` and,
the same file, under `python` -- and
[`vectors_numba.py`](compare/vectors_numba.py). Milliseconds, best of five,
over five processes.

**PPY** is a `@dataclass` and functions over it; Numba's nearest thing is a
`@jitclass` with a typed field list, whose instances are Numba's own
objects rather than Python ones:

```python
@dataclass
class Ray:
    origin: float
    direction: float


@ppy.pure
@ppy.opt(3)
def travel(ray: Ray, count: int) -> float:
    position: float = ray.origin
    total: float = 0.0
    for i in range(count):
        position = position * 0.999999 + ray.direction * (i % 3)
        total += position
    return total
```

```python
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
```

<!-- compare:start -->
| | PPY `ppy run` | CPython, the same file | Numba `@jitclass` |
|---|---:|---:|---:|
| distance2, a million calls from Python | **62.18 ± 0.47** | 63.13 ± 0.71 | 349.84 ± 7.47 |
| travel, eight million steps over a Ray natively | 11.16 ± 0.19 | 262.98 ± 2.15 | **10.97 ± 0.17** |
<!-- compare:end -->

Inside a native loop the two are the same code: the `Ray` is two doubles
in registers on both. The difference is the boundary. Eight float
operations are too little work to see past it: a PPY value class is still
a Python dataclass, the generated boundary reads its fields and passes
scalars, and a million calls cost what CPython takes to run the body
itself; a `@jitclass` instance crosses into an `@njit` function through
Numba's dispatcher and its own object layout, several times that per
call. The native loop is where the work is, and there the two agree. Constructing a `Vec3` inside a native loop is where PPY
stops: a value class built in the loop keeps the function in Python (`ppy
explain` says `boxed: Vec3 has no native lowering`), so the loop here
takes the class as a parameter.

Intel Core Ultra 9 386H; Numba 0.67.0 on CPython 3.12.13, PPY on CPython
3.14.5, from a checkout on a native filesystem.

## Run it

```bash
python  value_classes.ppy
ppy     value_classes.ppy
ppy run value_classes.ppy
```

<!-- outputs:start -->
## What it prints

**`python  value_classes.ppy`**

```text
14.0 25.0 25.0 7.0
25.0
# native: False
distance2 (8 float ops)      58.8 ns/call
steps     (200 iters)      4551.9 ns/call
```

**`ppy     value_classes.ppy`**

```text
14.0 25.0 25.0 7.0
25.0
# native: False
distance2 (8 float ops)      60.7 ns/call
steps     (200 iters)      4451.3 ns/call
```

**`ppy run value_classes.ppy`**

```text
14.0 25.0 25.0 7.0
25.0
# native: False
distance2 (8 float ops)      58.4 ns/call
steps     (200 iters)       322.6 ns/call
```

<!-- outputs:end -->

Read on: [Generics](../40_generics/README.md) ·
[Native data](../08_native_data/README.md)

`value_classes.ppy` is hand-written; there is no `.py` source and no
conversion step.

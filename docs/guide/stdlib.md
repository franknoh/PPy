# The standard library natively

A function that calls `random`, `math`, `heapq`, `bisect`, `itertools`, or
`collections.deque`, or reads a `string` constant, lowers to native code under `ppy run`, in a
standalone binary, and in emitted C and C++. It gives CPython's answer on
every path: the same random numbers from the same seed, the same heap after
the same pushes, the same last bit of a `math.fsum`.

```python
import bisect
import heapq
import math
import random


def simulate(n: int) -> str:
    random.seed(7)
    rolls = [random.randint(1, 6) for _ in range(n)]
    random.shuffle(rolls)
    picked = random.sample(rolls, 3)
    weighted = random.choices(["a", "b", "c"], [1, 5, 2], k=4)
    noise = random.normalvariate(0.0, 1.0) + random.expovariate(2.0)
    return f"{rolls[:5]} {picked} {weighted} {noise!r}"


def shortest(n: int) -> list[int]:
    dist = [1000000000] * n
    dist[0] = 0
    queue: list[tuple[int, int]] = [(0, 0)]
    while queue:
        d, u = heapq.heappop(queue)
        if d > dist[u]:
            continue
        for step in [1, 3, 7]:
            v = (u * step + 1) % n
            w = (u + v) % 11 + 1
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(queue, (d + w, v))
    return dist


def ranks(n: int, x: int) -> int:
    ordered = sorted([i * 37 % 11 for i in range(n)])
    bisect.insort(ordered, x)
    return bisect.bisect_left(ordered, x) + math.comb(len(ordered), 2)
```

## `random`

The generator is CPython's own: the Mersenne Twister, seeded the way
`random.seed` seeds it from an integer, and `random.py`'s algorithms on top
of it, draw for draw. A program seeded with `random.seed(42)` prints the same
numbers on every path.

| | |
|---|---|
| `seed(n)`, `seed()` | an integer seed, or one from the operating system |
| `random()`, `getrandbits(k)` | `k` up to 63 |
| `randrange`, `randint` | one, two, or three arguments, with CPython's errors |
| `choice(xs)`, `shuffle(xs)` | over a list |
| `sample(xs, k)` | over a list or a `range(...)` |
| `choices(xs, weights, k=)` | with `weights` or `cum_weights` of ints or floats |
| `uniform`, `triangular`, `normalvariate`, `lognormvariate` | |
| `expovariate`, `paretovariate`, `weibullvariate` | |
| `gammavariate`, `betavariate` | |
| `gauss` | in a standalone binary and emitted C; under `ppy run` it stays in Python |

Under `ppy run` native code draws from the generator Python's `random` uses,
in place. A native function and Python code can take turns drawing, and the
sequence is the one CPython would give. When a native call falls back to
Python after it drew (an integer grew past 64 bits, say), the generator is put
back as it was before the call, so the Python rerun draws the same numbers.
`gauss` keeps a value back between calls in a Python attribute, which is why
it stays in Python under `ppy run`.

A program that never seeds draws different numbers on each run, on every
path, as in CPython.

`random.Random(seed)` objects, `randbytes`, `getstate`, `setstate`, a string
seed, `vonmisesvariate`, and `binomialvariate` stay in Python.

## `math`

| | |
|---|---|
| `gcd`, `lcm`, `isqrt`, `comb`, `perm`, `factorial` | integers; a result past 64 bits falls back to Python |
| `fsum`, `prod` | over a list of ints or floats; `fsum` is CPython's exact algorithm |
| `hypot`, `dist` | two or three coordinates, or `dist` of two lists; CPython's correctly rounded norm |
| `isclose` | with `rel_tol=` and `abs_tol=` |
| `log(x, base)`, `copysign`, `atan2`, `fmod`, `degrees`, `radians` | |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh` | |
| `asinh`, `acosh`, `atanh`, `log1p`, `expm1`, `erf`, `erfc`, `cbrt`, `exp2` | |
| `pi`, `e`, `tau`, `inf`, `nan` | as `math.pi` or `from math import pi` |

`math.sqrt(-1.0)`, `math.log(0.0)`, `math.exp(1000.0)`, and the like raise
what CPython raises. Under `ppy run` the call falls back and Python raises.
A standalone binary prints the exception. On Python 3.14, which puts the
argument in the message ("expected a nonnegative input, got -1.0"), the
binary prints "math domain error" instead.

## `heapq` and `bisect`

`heappush`, `heappop`, `heapify`, `heappushpop`, `heapreplace`, `nlargest`,
and `nsmallest` work on a list of numbers, strings, or tuples of them.
`bisect_left`, `bisect_right`, `insort_left`, and `insort_right` take `lo`
and `hi`. Both make the same comparisons as CPython, in the same order, so a
heap of tuples leaves the list in the same state. `key=` stays in Python, and
so does popping from a heap whose elements hold strings.

## `itertools`

`product`, `permutations(xs, r)`, `combinations(xs, r)`,
`combinations_with_replacement(xs, r)`, `pairwise`, `chain`, `islice`,
`accumulate` (with `+` and `initial=`), and `repeat(x, n)` lower where a
`for` loop, `list`, `sum`, `sorted`, or a comprehension consumes them. They
take lists and `range(...)`, and the tuples they hand out hold numbers.
The lowering makes the whole sequence into a list first, in CPython's order.
`itertools` takes its inputs when it is called, so the loop sees the same
items. `r` is a constant. `count`, `cycle`, and `groupby` stay in Python.

## `string`

`ascii_letters`, `ascii_lowercase`, `ascii_uppercase`, `digits`,
`hexdigits`, `octdigits`, `punctuation`, `whitespace`, and `printable` are
string literals in native code.

## `collections.deque`

A deque whose annotation says what it holds (`q: deque[int] = deque()`, or
`deque([...])` given to an annotated name) is the runtime's `Deque`:
`append`, `appendleft`, `pop`, `popleft`, `extend`, `extendleft`, `rotate`,
`clear`, `copy`, `len`, `in`, a `for` loop, and `q[i]` counted from either
end. Popping an empty deque raises CPython's `IndexError`. `maxlen`,
`q[i] = x`, and `list(q)` stay in Python.

## What stays in Python

`defaultdict`, `Counter`, and `OrderedDict`, `functools.reduce`,
`functools.lru_cache` and `cache`, and `operator`'s functions are not
lowered yet. A function that calls them runs in Python, and `ppy explain
--summary` names the call.

# The standard library natively

A function that calls `random`, `math`, `heapq`, `bisect`, `itertools`,
`functools`, or `operator`, uses `collections`' `deque`, `defaultdict`,
`Counter`, or `OrderedDict`, or reads a `string` constant, lowers to native
code under `ppy run`, in a standalone binary, and in emitted C and C++. It
gives CPython's answer on every path: the same random numbers from the same
seed, the same heap after the same pushes, the same last bit of a
`math.fsum`, the same `repr` of a `Counter`.

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

The containers of `collections`, a cache, and the functions of `functools`
and `operator` used as values:

```python
import random
from collections import Counter, defaultdict
from functools import cache, reduce
from operator import itemgetter, mul


@cache
def paths(rows: int, cols: int) -> int:
    if rows == 0 or cols == 0:
        return 1
    return paths(rows - 1, cols) + paths(rows, cols - 1)


def tally(words: list[str]) -> str:
    counts = Counter(words)
    lengths: defaultdict[int, list[str]] = defaultdict(list)
    for word in counts:
        lengths[len(word)].append(word)
    top = [f"{word}={n}" for word, n in counts.most_common(2)]
    return f"{counts} {lengths} {top}"


def dice(seed: int) -> str:
    rng = random.Random(seed)
    rolls = [(i, rng.randint(1, 6)) for i in range(5)]
    best = max(rolls, key=itemgetter(1))
    return f"{rolls} {best[0]} {reduce(mul, [r for _, r in rolls], 1)}"
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

`randbytes`, `getstate`, `setstate`, a string seed, `vonmisesvariate`, and
`binomialvariate` stay in Python.

### `random.Random`

`random.Random(seed)` and `random.Random()` are generators of their own,
seeded as CPython seeds them, with every method the module's functions
above have: `r.randint(1, 6)` draws what `random.Random(seed).randint(1, 6)`
draws in CPython, whatever the module's generator is doing. `r.seed(n)`
reseeds the instance, and `r.gauss` holds its second value back in the
instance, so it is native under `ppy run` as well. An instance can be passed
to another native function; it does not cross into Python, so a function
that takes one from Python, or hands one back, stays in Python.

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

A deque is the runtime's `Deque`: `append`, `appendleft`, `pop`, `popleft`,
`extend`, `extendleft`, `rotate`, `clear`, `copy`, `len`, `in`, a `for`
loop, `list(q)`, `sorted(q)`, and `q[i]` and `q[i] = x` counted from either
end. What it holds comes from its annotation (`q: deque[int] = deque()`), or
from what it is made of (`deque([1, 2])`, `deque(range(n))`), or from the
first `append` to an empty one. `print(q)` and `repr(q)` give
`deque([1, 2])`. Popping an empty deque raises CPython's `IndexError`.
`maxlen` stays in Python.

## `defaultdict`, `Counter`, and `OrderedDict`

Each is a dict natively, in insertion order, so a subscript, `in`, `len`, a
loop over it or its `keys()`, `values()`, and `items()`, `get`, `pop`,
`setdefault`, `update`, `copy`, and `dict(d)` work as they do on a dict.
Their own behavior:

| | |
|---|---|
| `defaultdict(int)`, `defaultdict(list)`, `defaultdict(set)`, ... | a missing key gets the factory's value and is added before the read |
| `defaultdict(lambda: -1)`, `defaultdict(f)` | the function is called for each missing key, as in CPython |
| `Counter(xs)`, `Counter("text")`, `Counter(d)` | counts an iterable's elements, or copies a mapping's counts |
| `c[k]` | 0 for a missing key, which is not added |
| `c.most_common()`, `c.most_common(n)` | the counts from highest, equal counts in insertion order |
| `c.update(xs)`, `c.subtract(xs)` | an iterable counted, or a mapping's counts added or taken away |
| `c.elements()`, `c.total()` | |
| `a + b`, `a - b`, `a \| b`, `a & b` | `Counter`'s operators: only positive counts are kept |
| `od.move_to_end(k)`, `od.move_to_end(k, last=False)` | |
| `od.popitem()`, `od.popitem(last=False)` | for keys and values that are numbers |

`repr` is CPython's: `defaultdict(<class 'int'>, {0: 3})`,
`Counter({'l': 3, 'o': 2})` in `most_common` order, and `OrderedDict({1:
2})` (`OrderedDict([(1, 2)])` before Python 3.12). A `defaultdict` whose
factory is a function is shown by Python, which names the function by its
address. `==` between two `Counter`s or two `OrderedDict`s stays in Python,
since the first ignores counts of zero and the second compares order. A
string-keyed `most_common` is walked by a loop, a comprehension, or `print`;
kept as a list, it stays in Python, since a list of `(str, int)` tuples has
no native form.

Like a deque, these are passed between native functions by handle and do
not cross into Python.

## `functools`

| | |
|---|---|
| `@cache`, `@lru_cache(maxsize=None)` | every result kept |
| `@lru_cache`, `@lru_cache(maxsize=n)` | at most `n` (128 by default), the one used longest ago let go first |
| `reduce(f, xs)`, `reduce(f, xs, start)` | with `f` a lambda, a function of the program, `max`, `min`, or an `operator` function |
| `partial(f, a, ...)` | where a function value is used, with constant arguments or ones that keep one value |
| `cmp_to_key(f)` | as a `sort` or `sorted` key, over numbers, strings, and objects |

A cached function lowers to its body and an entry that looks its arguments
up in a table first. A call in the body to the function itself is a call to
the entry, so a dynamic program computes each subproblem once, natively.
Under `ppy run` the entry is what Python's callers call too, so they share
the table. The arguments are numbers, bools, strings, and tuples of
numbers, and so is the result. A cached function defined inside another
gets a new table each time its `def` runs, as CPython makes a new cache, and
the closure holds it. A function whose `cache_info`, `cache_clear`, or
`__wrapped__` the module reads keeps CPython's cache and stays in Python, so
the counts it reports are CPython's. A cached method stays in Python, and
so does a cached function with default arguments, since `f(5)` and
`f(5, 2)` are two entries in CPython's cache.

## `operator`

`add`, `sub`, `mul`, `truediv`, `floordiv`, `mod`, `pow`, the bitwise and
shift operators, `neg`, `pos`, `invert`, `not_`, `truth`, `abs`, the
comparisons, `contains`, `getitem`, `itemgetter`, and `attrgetter` are the
lambdas they stand for wherever a function value is used: a `key=`, `map`,
`filter`, `reduce`, or a `Callable` parameter. `itemgetter(1)` is
`lambda x: x[1]`, and `attrgetter("weight")` is `lambda x: x.weight`.

## What stays in Python

`ChainMap`, `namedtuple`, `deque(maxlen=...)`, `functools.singledispatch`,
`cached_property`, and `operator.methodcaller` are not lowered. A function
that uses them runs in Python, and `ppy explain --summary` names the call.
